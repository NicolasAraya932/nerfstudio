from typing import Dict, Optional, Tuple, Literal
import torch
from torch import Tensor, nn
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.field_components.field_heads import SemanticFieldHead, FieldHeadNames
from nerfstudio.fields.base_field import Field


class SemanticIEField(Field):
    """Merged field for density and semantics using hash-based encoding.

    Args:
        aabb: Scene bounding box.
        num_levels: Number of levels in the hash encoding.
        base_res: Minimum resolution of the hash grid.
        max_res: Maximum resolution of the hash grid.
        log2_hashmap_size: Log2 size of the hash map.
        features_per_level: Number of features per level in the hash grid.
        num_layers: Number of layers in the MLP.
        layer_width: Width of each MLP layer.
        geo_feat_dim: Dimension of geometric features.
        num_semantic_classes: Number of semantic classes.
        spatial_distortion: Spatial distortion module.
        average_init_density: Initial density scaling factor.
        implementation: Backend implementation ("tcnn" or "torch").
    """

    def __init__(
        self,
        aabb: Tensor,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        num_layers: int = 9,
        layer_width: int = 256,
        geo_feat_dim: int = 15,
        num_semantic_classes: int = 2,
        average_init_density: float = 1.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.average_init_density = average_init_density
        self.geo_feat_dim = geo_feat_dim
        self.num_semantic_classes = num_semantic_classes

        print("Inside the SemanticIEField constructor")
        print(f"Using {num_levels} levels, base res {base_res}, max res {max_res}, log2_hashmap_size {log2_hashmap_size}")

        # Hash Encoding
        self.encoding = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )

        # Base MLP for density and geometric features
        self.mlp_base = MLP(
            in_dim=self.encoding.get_out_dim(),
            num_layers=num_layers,
            layer_width=layer_width,
            out_dim=1 + geo_feat_dim,  # 1 for density, rest for geometric features
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        # Semantic MLP for semantic logits
        self.mlp_semantic = MLP(
            in_dim=geo_feat_dim,
            num_layers=1,
            layer_width=128,
            out_dim=num_semantic_classes,
            activation=nn.ReLU(),
            out_activation=None,
        )

        # Semantic field head
        self.field_head_semantic = SemanticFieldHead(
            in_dim=self.mlp_semantic.get_out_dim(), num_classes=num_semantic_classes
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes density and geometric features."""
        
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        positions_flat = positions.view(-1, 3)

        # Compute density and geometric features
        h = self.mlp_base(self.encoding(positions_flat)).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        # Rectify density
        density = self.average_init_density * trunc_exp(density_before_activation)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        """Get outputs for density and semantics."""
        assert density_embedding is not None, "density_embedding is None :C"
        outputs = {}

        # Compute density
        density, base_mlp_out = self.get_density(ray_samples)
        outputs[FieldHeadNames.DENSITY] = density

        # Compute semantic logits
        semantics_input = base_mlp_out.view(-1, self.geo_feat_dim)
        semantic_logits = self.mlp_semantic(semantics_input).view(*ray_samples.frustums.directions.shape[:-1], -1)
        outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantic(semantic_logits)

        return outputs