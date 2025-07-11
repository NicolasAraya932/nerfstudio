# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Semantic NeRF-W implementation which should be fast enough to view in the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Literal

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn import CrossEntropyLoss

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.semantic_is_enough_field import SemanticIEField
from nerfstudio.model_components.losses import distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class SemanticIEModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SemanticIEModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""

    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 1024
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""

    num_semantic_classes: int = 2
    """Number of semantic classes."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""

    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""

    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""

    average_init_density: float = 1.0
    """Average initial density output from MLP. """

    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""

class SemanticIEModel(Model):
    """Semantic NeRF model for binary semantics and density."""

    config: SemanticIEModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # Initialize the field
        self.field = SemanticIEField(
            aabb = self.scene_box.aabb,
            num_levels = self.config.num_levels,
            base_res = self.config.base_res,
            max_res = self.config.max_res,
            log2_hashmap_size = self.config.log2_hashmap_size,
            features_per_level = self.config.features_per_level,
            average_init_density = self.config.average_init_density,
            implementation = self.config.implementation
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        for _ in range(self.config.num_proposal_iterations):
            network = HashMLPDensityField(
                self.scene_box.aabb,
                num_levels=self.config.num_levels,
                base_res=self.config.base_res,
                max_res=self.config.max_res,
                log2_hashmap_size=self.config.log2_hashmap_size,
                features_per_level=self.config.features_per_level,
                average_init_density=self.config.average_init_density,
            )
            self.proposal_networks.append(network)
        
        # Populate density functions
        self.density_fns = [network.density_fn for network in self.proposal_networks]

        # Collider
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
        ) 
        
        # Proposal sampler
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
        )
        
        # Renderers
        self.renderer_semantics = SemanticRenderer()

        # Losses
        self.semantic_loss = CrossEntropyLoss()
        self.interlevel_loss = interlevel_loss

        import matplotlib.pyplot as plt
        
        # Initialize colormap using matplotlib
        cmap = plt.get_cmap("viridis", self.config.num_semantic_classes)
        self.colormap = torch.tensor(cmap.colors, dtype=torch.float32)

    def get_outputs(self, ray_bundle: RayBundle):
        """Compute outputs for semantics only."""
        # Sample points along rays using the proposal sampler
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
    
        # Compute field outputs
        field_outputs = self.field(ray_samples)
    
        # Compute density weights
        weights_static = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
    
        # Render semantics
        semantic_weights = weights_static
        semantics = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )
    
        # Ensure semantics has the correct shape
        if semantics.dim() == 4 and semantics.shape[-1] == 1:  # Shape: (B, H, W, 1)
            semantics = semantics.squeeze(-1)  # Convert to (B, H, W)
    
        # Apply colormap for visualization
        semantic_labels = torch.argmax(torch.nn.functional.softmax(semantics, dim=-1), dim=-1)
        semantics_colormap = self.colormap.to(self.device)[semantic_labels]
    
        # Return only semantics-related outputs
        return {
            "semantics": semantics,
            "semantics_colormap": semantics_colormap,
        }


    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}

        # Predictions
        pred_logits = outputs["semantics"]  # [N_rays, num_classes]
        N_rays = pred_logits.shape[0]

        # GT labels
        gt_sem = batch["image"].to(self.device).long().view(-1)  # [H*W] flattened

        # Get ray_indices (if available)
        ray_indices = batch.get("ray_indices", torch.arange(N_rays, device=self.device))

        # Ensure indexing is aligned
        ray_indices = ray_indices.to(self.device)
        gt_labels = gt_sem[ray_indices]  # [N_rays]

        # Cross entropy loss
        loss_dict["semantic_loss"] = self.semantic_loss(pred_logits, gt_labels)

        # Optional losses
        if metrics_dict is not None:
            if "distortion" in metrics_dict:
                loss_dict["distortion_loss"] = (
                    self.config.distortion_loss_mult * metrics_dict["distortion"]
                )
            if "interlevel" in metrics_dict:
                loss_dict["interlevel_loss"] = (
                    self.config.interlevel_loss_mult * metrics_dict["interlevel"]
                )

        return loss_dict


    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}

        ray_indices = batch.get("ray_indices", None)
        gt_sem = batch["image"].to(self.device).long()  # [H, W] or [B, H, W]

        # Flatten GT labels
        gt_flat = gt_sem.view(-1)  # [H*W]

        N_rays = outputs["semantics"].shape[0]

        if ray_indices is None:
            ray_indices = torch.arange(N_rays, device=self.device)

        # Ensure ray_indices has shape [N_rays]
        ray_indices = ray_indices.to(self.device)
        gt_labels = gt_flat[ray_indices]  # [N_rays]

        # Predictions
        pred_logits = outputs["semantics"]  # [N_rays, C]
        pred_labels = pred_logits.argmax(dim=-1)  # [N_rays]

        metrics_dict["semantic_accuracy"] = (pred_labels == gt_labels).float().mean()

        return metrics_dict


    def get_image_metrics_and_images(self, outputs, batch):
        """
        Return:
          - {} for scalar image‐metrics
          - {"semantics": ..., ...} for visuals
        """
        images_dict: Dict[str, torch.Tensor] = {}

        # Colorize the predicted semantics
        pred_logits = outputs["semantics"]                       # [B*H*W? or N_rays]
        # If we can reshape back to (B,H,W,2), do so; else assume H=W=?? for display
        # Here, we expect get_outputs produced [B,H,W,2]:
        if pred_logits.ndim == 2:
            # can't visualize per‐pixel grid if it's per‐ray only
            return {}, {}
        sem_logits = pred_logits                                 # [B,H,W,2]
        pred_labels = sem_logits.argmax(dim=-1).cpu().numpy()    # [B,H,W]

        # Apply a colormap for TensorBoard/viewer
        sem_vis = colormaps.apply_colormap(pred_labels)          # [B,H,W,3] uint8
        images_dict["semantics"] = sem_vis

        return {}, images_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """
        Return a dict mapping group names to lists of Parameters.
        Nerfstudio will merge this with the DataManager param groups.
        """
        param_groups: Dict[str, List[Parameter]] = {}

        # 1) Proposal networks
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())

        # 2) The semantic‐only field
        param_groups["fields"] = list(self.field.parameters())

        # 3) (Optional) any other modules you added, e.g. camera optimizer
        # param_groups["camera_optimizer"] = list(self.camera_optimizer.parameters())

        return param_groups