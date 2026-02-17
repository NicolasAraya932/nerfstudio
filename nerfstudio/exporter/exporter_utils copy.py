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
Export utils such as structs, point cloud generation, and rendering code.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pymeshlab
import torch
from jaxtyping import Float
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch import Tensor

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn

if TYPE_CHECKING:
    # Importing open3d can take ~1 second, so only do it below if we actually
    # need it.
    import open3d as o3d


@dataclass
class Mesh:
    """Class for a mesh."""

    vertices: Float[Tensor, "num_verts 3"]
    """Vertices of the mesh."""
    faces: Float[Tensor, "num_faces 3"]
    """Faces of the mesh."""
    normals: Float[Tensor, "num_verts 3"]
    """Normals of the mesh."""
    colors: Optional[Float[Tensor, "num_verts 3"]] = None
    """Colors of the mesh."""


def get_mesh_from_pymeshlab_mesh(mesh: pymeshlab.Mesh) -> Mesh:  # type: ignore
    """Get a Mesh from a pymeshlab mesh.
    See https://pymeshlab.readthedocs.io/en/0.1.5/classes/mesh.html for details.
    """
    return Mesh(
        vertices=torch.from_numpy(mesh.vertex_matrix()).float(),
        faces=torch.from_numpy(mesh.face_matrix()).long(),
        normals=torch.from_numpy(np.copy(mesh.vertex_normal_matrix())).float(),
        colors=torch.from_numpy(mesh.vertex_color_matrix()).float(),
    )


def get_mesh_from_filename(filename: str, target_num_faces: Optional[int] = None) -> Mesh:
    """Get a Mesh from a filename."""
    ms = pymeshlab.MeshSet()  # type: ignore
    ms.load_new_mesh(filename)
    if target_num_faces is not None:
        CONSOLE.print("Running meshing decimation with quadric edge collapse")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_num_faces)
    mesh = ms.current_mesh()
    return get_mesh_from_pymeshlab_mesh(mesh)

def common_resolution_from_largest_aabb(
    aabbs: List[OrientedBox],  # [(min3,max3),...]
    n_points_ref: int,
    factor: float = 2.0,     # UDPC = factor * n_points_ref
    snap: int = 16,          # snap each axis to nearest multiple (8/16)
    max_voxels: int = 200_000_000  # safety cap
) -> Tuple[int,int,int]:
    """
        The volume of the Candidate Region is defined as the multiplication of
        each dimension. V = Lx*Ly*Lz

        The objective is to obtain the shape of each voxel and axes proportional to physical extents.

        With k as the scale of a voxel, we can express the volume of the grid in terms of k:

        k^3*Lx*Ly*Lz approx N_grid

        k = (N_grid/(Lx*Ly*Lz))^(1/3)

        Then the number of voxels along each axis is:
        Dx = ⌈kLx⌉,  Dy = ⌈kLy⌉,  Dz = ⌈kLz⌉
    """

    sizes = []
    volumes = []

    # pick largest volume AABB
    for aabb in aabbs:
        S = aabb.S.detach().cpu().numpy() if isinstance(aabb.S, torch.Tensor) else aabb.S
        Lx, Ly, Lz = float(S[0]), float(S[1]), float(S[2])
        eps = 1e-12
        V = max(Lx*Ly*Lz, eps)
        sizes.append((Lx, Ly, Lz))
        volumes.append(V)
    
    max_idx = int(np.argmax(volumes))
    Lx, Ly, Lz = sizes[max_idx]
    V = volumes[max_idx]

    N_udpc = max(int(factor * n_points_ref), 1)
    k = (N_udpc / V) ** (1.0/3.0)

    Dx = max(1, int(np.ceil(k * Lx)))
    Dy = max(1, int(np.ceil(k * Ly)))
    Dz = max(1, int(np.ceil(k * Lz)))

    # snap to multiples for CNN friendliness
    def snap_to(v, m): return max(m, int(np.ceil(v / m) * m))
    Dx, Dy, Dz = snap_to(Dx, snap), snap_to(Dy, snap), snap_to(Dz, snap)

    # safety cap
    total = Dx * Dy * Dz
    if total > max_voxels:
        scale = (max_voxels / float(total)) ** (1.0/3.0)
        Dx = snap_to(max(1, int(Dx * scale)), snap)
        Dy = snap_to(max(1, int(Dy * scale)), snap)
        Dz = snap_to(max(1, int(Dz * scale)), snap)

    return Dx, Dy, Dz

@torch.no_grad()
def _query_sigma_field(pipeline, P_world: torch.Tensor) -> torch.Tensor:
    """
    Query σ at arbitrary world positions P_world: [N,3] -> [N]
    Tries field.density_fn; fallback to get_density with degenerate RaySamples.
    """
    field = pipeline.model.field
    if hasattr(field, "density_fn") and callable(field.density_fn):
        return field.density_fn(P_world).squeeze(-1)

    # Fallback: build degenerate RaySamples (starts==ends) and call get_density
    from nerfstudio.model_components.ray_samplers import Frustums, RaySamples
    B = P_world.shape[0]
    zeros = torch.zeros(B, 1, device=P_world.device, dtype=P_world.dtype)
    fr = Frustums(
        origins=P_world,
        directions=torch.zeros_like(P_world),
        starts=zeros, ends=zeros,
        pixel_area=torch.ones(B,1, device=P_world.device, dtype=P_world.dtype),
    )
    rs = RaySamples(frustums=fr)
    sigma, _ = field.get_density(rs)   # -> [B,1]
    return sigma.squeeze(-1)

@torch.no_grad()
def export_voxel_alpha_grid(
    pipeline,
    candidate_regions: str,
    factor: float = 2.0,            # ~ target total voxels ≈ factor * num_points_ref
    snap: int = 16,
    max_voxels: int = 200_000_000,
    assume_half_extents: bool = False,
    chunk_points: int = 1_000_000,
    save_npz_pattern: Optional[str] = None,  # e.g. "roi_{i:02d}_alpha.npz"
) -> Dict[str, List]:
    """
    For each OrientedBox in candidate_regions, sample σ at voxel centers on a COMMON resolution
    derived from the largest ROI, convert to α, and return a list of grids and metadata.
    """
    device = pipeline.device
    AABBS: List[OrientedBox] = torch.load(candidate_regions)
    if not all(hasattr(a, "S") or hasattr(a, "size") or hasattr(a, "extents") for a in AABBS):
        CONSOLE.rule("Error", style="red")
        CONSOLE.print("export_voxel_alpha_grid() expects OrientedBox objects with S/size/extents.", justify="center")
        sys.exit(1)

    # You were reading a radiance_field_cloud just to get N. We can just use number of rays * samples
    # or pick a sane reference like 1e6. If you still prefer, pass n_points_ref explicitly.
    n_points_ref = 1_000_000

    # Common resolution from the largest ROI
    Dx, Dy, Dz = common_resolution_from_largest_aabb(
        AABBS, n_points_ref=n_points_ref, factor=factor, snap=snap,
        max_voxels=max_voxels
    )

    grids_alpha: List[torch.Tensor] = []
    metas: List[Dict] = []

    for i, obb in enumerate(AABBS):
        C, R, L = obb._extract_obb_components()          # world
        if assume_half_extents:
            L = 2.0 * L
        Lx, Ly, Lz = L.tolist()

        # Per-ROI spacings (these *will* match the physical extents since Dx,Dy,Dz are fixed)
        dx, dy, dz = Lx/Dx, Ly/Dy, Lz/Dz
        # if camera rays are mostly along world +Z
        Delta = float((dx + dy + dz) / 3.0)             # used in α = 1 - exp(-σΔ)

        # Local grid centers in the OBB frame: [-L/2, L/2] with Dx,Dy,Dz cells
        xs = (-0.5 + (torch.arange(Dx, device=device) + 0.5) / Dx) * Lx  # [Dx]
        ys = (-0.5 + (torch.arange(Dy, device=device) + 0.5) / Dy) * Ly  # [Dy]
        zs = (-0.5 + (torch.arange(Dz, device=device) + 0.5) / Dz) * Lz  # [Dz]

        # We’ll iterate over Z to limit memory
        alpha_zi = torch.zeros((Dz, Dy, Dx), dtype=torch.float32, device=device)

        for iz in range(Dz):
            Z = zs[iz].expand(Dy, Dx)                                 # [Dy,Dx]
            Y, X = torch.meshgrid(ys, xs, indexing="ij")              # [Dy,Dx] each
            P_local = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)   # [Dy*Dx,3]

            # World transform: P_world = C + R @ P_local
            P_world = (P_local @ R.T) + C[None, :]

            # Chunked σ query
            sigmas = []
            for s in range(0, P_world.shape[0], chunk_points):
                sigmas.append(_query_sigma_field(pipeline, P_world[s:s+chunk_points]))
            sigma_slice = torch.cat(sigmas, dim=0).view(Dy, Dx)       # [Dy,Dx]

            # α = 1 - exp(-σΔ) (no transmittance)
            alpha_slice = 1 - torch.exp(-sigma_slice * Delta)
            alpha_zi[iz] = alpha_slice

        grids_alpha.append(alpha_zi.detach().cpu())

        meta = {
            "roi_index": i,
            "center": np.array(C.detach().cpu()),
            "rotation": np.array(R.detach().cpu()),
            "lengths": np.array(L.detach().cpu()),   # side lengths
            "resolution": (Dx, Dy, Dz),
            "spacing": (dx, dy, dz),
            "Delta": Delta,
        }
        metas.append(meta)

        if save_npz_pattern is not None:
            np.savez_compressed(save_npz_pattern.format(i=i),
                                alpha=np.asarray(alpha_zi.cpu()),
                                **meta)

    return {"alpha_grids": grids_alpha, "meta": metas}

def generate_radiance_fields_cloud(
    pipeline: Pipeline,
    num_points: int = 6500000,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    normal_output_name: Optional[str] = None,
    crop_obb: Optional[OrientedBox] = None,
) -> Dict[str, torch.Tensor]:
    """Generate a radiance field dataset from a NeRF model.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        crop_obb: Optional oriented bounding box to crop points.

    Returns:
        A dictionary containing all radiance field data.
    """

    # Initialize progress bar
    progress = Progress(
        TextColumn(":cloud: Computing Radiance Field :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )

    # Initialize lists to store outputs
    points = []
    rgbs = []
    accumulations = []
    depths = []
    origins = []
    directions = []
    pixel_areas = []
    normals = []
    view_directions = []
    density = []
    mask = None

    with progress as progress_bar:
        task = progress_bar.add_task("Generating Radiance Field", total=num_points)
        while not progress_bar.finished:
            normal = None

            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                assert isinstance(ray_bundle, RayBundle)
                outputs = pipeline.model(ray_bundle)

            # Validate outputs
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)

            rgba = pipeline.model.get_rgba_image(outputs, rgb_output_name)
            depth = outputs[depth_output_name]
            sigma = outputs["density"]

            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                normal = outputs[normal_output_name]
                assert (
                    torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
                ), "Normal values from method output must be in [0, 1]"
                normal = (normal * 2.0) - 1.0

            point = ray_bundle.origins + ray_bundle.directions * depth
            view_direction = ray_bundle.directions

            # Filter points with opacity lower than 0.01
            mask = rgba[..., -1] > 0.01
            point = point[mask]
            sigma = sigma[mask]
            view_direction = view_direction[mask]
            rgb = rgba[mask][..., :3]
            if normal is not None:
                normal = normal[mask]

            if crop_obb is not None:
                mask = crop_obb.within(point)
                point = point[mask]
                rgb = rgb[mask]
                view_direction = view_direction[mask]
                if normal is not None:
                    normal = normal[mask]

            # Append data to lists
            points.append(point.cpu())
            rgbs.append(rgb.cpu())
            accumulations.append(outputs["accumulation"][mask].cpu())
            depths.append(outputs["depth"][mask].cpu())
            origins.append(ray_bundle.origins[mask].cpu())
            directions.append(ray_bundle.directions[mask].cpu())
            pixel_areas.append(ray_bundle.pixel_area[mask].cpu())
            view_directions.append(view_direction.cpu())
            density.append(sigma.cpu())

            if normal is not None:
                normals.append(normal.cpu())

            progress.advance(task, point.shape[0])

    # Combine lists into tensors on the CPU
    radiance_field_data = {
        "points": torch.cat(points, dim=0),
        "rgb": torch.cat(rgbs, dim=0),
        "accumulation": torch.cat(accumulations, dim=0),
        "depth": torch.cat(depths, dim=0),
        "origins": origins[0],
        "directions": torch.cat(directions, dim=0),
        "pixel_area": torch.cat(pixel_areas, dim=0),
        "view_directions": torch.cat(view_directions, dim=0),
        "density": torch.cat(density, dim=0),
    }

    if normals:
        radiance_field_data["normals"] = torch.cat(normals, dim=0)

    return radiance_field_data

def generate_point_cloud(
    pipeline: Pipeline,
    num_points: int = 1000000,
    remove_outliers: bool = True,
    estimate_normals: bool = False,
    reorient_normals: bool = False,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    normal_output_name: Optional[str] = None,
    crop_obb: Optional[OrientedBox] = None,
    std_ratio: float = 10.0,
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        reorient_normals: Whether to re-orient the normals based on the view direction.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )
    points = []
    rgbs = []
    normals = []
    view_directions = []
    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            normal = None

            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                assert isinstance(ray_bundle, RayBundle)
                outputs = pipeline.model(ray_bundle)
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)

            rgba = pipeline.model.get_rgba_image(outputs, rgb_output_name)
            depth = outputs[depth_output_name]
            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                normal = outputs[normal_output_name]
                assert torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0, (
                    "Normal values from method output must be in [0, 1]"
                )
                normal = (normal * 2.0) - 1.0
            point = ray_bundle.origins + ray_bundle.directions * depth
            view_direction = ray_bundle.directions

            # Filter points with opacity lower than 0.5
            mask = rgba[..., -1] > 0.5
            point = point[mask]
            view_direction = view_direction[mask]
            rgb = rgba[mask][..., :3]
            if normal is not None:
                normal = normal[mask]

            if crop_obb is not None:
                mask = crop_obb.within(point)
                point = point[mask]
                rgb = rgb[mask]
                view_direction = view_direction[mask]
                if normal is not None:
                    normal = normal[mask]

            points.append(point)
            rgbs.append(rgb)
            view_directions.append(view_direction)
            if normal is not None:
                normals.append(normal)
            progress.advance(task, point.shape[0])
    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)
    view_directions = torch.cat(view_directions, dim=0).cpu()

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())

    ind = None
    if remove_outliers:
        CONSOLE.print("Cleaning Point Cloud")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")
        if ind is not None:
            view_directions = view_directions[ind]

    # either estimate_normals or normal_output_name, not both
    if estimate_normals:
        if normal_output_name is not None:
            CONSOLE.rule("Error", style="red")
            CONSOLE.print("Cannot estimate normals and use normal_output_name at the same time", justify="center")
            sys.exit(1)
        CONSOLE.print("Estimating Point Cloud Normals")
        pcd.estimate_normals()
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")
    elif normal_output_name is not None:
        normals = torch.cat(normals, dim=0)
        if ind is not None:
            # mask out normals for points that were removed with remove_outliers
            normals = normals[ind]
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    # re-orient the normals
    if reorient_normals:
        normals = torch.from_numpy(np.array(pcd.normals)).float()
        mask = torch.sum(view_directions * normals, dim=-1) > 0
        normals[mask] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    return pcd



def extract_fruit_proposal_outputs(

    pipeline: Pipeline,
    num_points: int = 1000000,
    weight_threshold: float = 0.3,
    acc_threshold: float = 0.3,
    sigma_threshold: float = 0.1, # Values from 0 to sigma_max, so percentage
    crop_obb: Optional[OrientedBox] = None,
):
    progress = Progress(
        TextColumn(":cloud: Computing Fruit Proposal Field :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )

    class parameters:

        rgba : Any = None
        sigma : Any = None
        depth : Any = None
        exp_depth: Any = None
        oTransmittance : Any = None
        accumulation : Any = None
        t_mid : Any = None

        def raiseExcept(self):
            if self.rgba is None:
                raise ValueError("rgba is not saved")

            if self.sigma is None:
                raise ValueError("sigma is not saved")
            
            if self.depth is None:
                raise ValueError("depth is not saved")

            if self.exp_depth is None:
                raise ValueError("exp_depth is not saved")

            if self.oTransmittance is None:
                raise ValueError("oTransmittance is not saved")

            if self.accumulation is None:
                raise ValueError("accumulation is not saved")

            if self.t_mid is None:
                raise ValueError("t_mid is not saved")


    exports : List[Dict[str, Any]] = []
    densities = []
    colors = []
    points_by_depth = {}
    points_by_depth["rgba_mask"] = []
    points_by_depth["oTransmittance_mask"] = []
    points_by_depth["acc_mask"] = []
    points_by_depth["sigma_mask"] = []
    points_by_exp_depth = {}
    points_by_exp_depth["rgba_mask"] = []
    points_by_exp_depth["oTransmittance_mask"] = []
    points_by_exp_depth["acc_mask"] = []
    points_by_exp_depth["sigma_mask"] = []
    points_by_oTransmittance = {}
    points_by_oTransmittance["rgba_mask"] = []
    points_by_oTransmittance["oTransmittance_mask"] = []
    points_by_oTransmittance["acc_mask"] = []
    points_by_oTransmittance["sigma_mask"] = []

    with progress as progress_bar:
        task = progress_bar.add_task("Generating Radiance Field", total=num_points)
        while not progress_bar.finished:

            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                assert isinstance(ray_bundle, RayBundle)
                outputs = pipeline.model(ray_bundle)

            exports_labels = outputs.keys()
            export = {}
            for label in exports_labels:
                export[label] = outputs[label]

            view_direction = ray_bundle.directions

            # To obtain
            param = parameters()

            for label in exports_labels:
                if label == "rgb":
                    # To filter by opacity
                    param.rgba = pipeline.model.get_rgba_image(outputs, label)

                if label == "density":
                    # To filter by density
                    param.sigma = outputs[label]

                # Points according to depth
                if label == "depth":
                    param.depth = outputs[label]

                # Points according to expected depth
                if label == "expected_depth":
                    param.exp_depth = outputs[label]

                # Points according to opacity x transmittance
                if label == "opacity_transmittance":
                    param.oTransmittance = outputs[label]

                # Points according to accumulation (SUM(opacity x transmittance))
                if label == "accumulation":
                    param.accumulation = outputs[label]

                if label == "t_mid":
                    param.t_mid = outputs[label].squeeze(-1)
            
            param.raiseExcept()

            # Only density
            density = param.sigma

            # Only color
            rgb = param.rgba[..., :3]

            # POINTS BY:
            # By opacity_transmittance
            idx_oTransmittance = torch.argmax(param.oTransmittance, dim=-1, keepdim=True)             # [N,1]
            t_peak_oTransmittance = torch.gather(param.t_mid, -1, idx_oTransmittance).squeeze(-1)             # [N]
            point_by_oTransmittance = ray_bundle.origins + ray_bundle.directions * t_peak_oTransmittance[..., None]  # [N,3]

            # By accumulation
            idx_acc = torch.argmax(param.accumulation, dim=-1, keepdim=True)             # [N,1]
            t_peak_acc = torch.gather(param.t_mid, -1, idx_acc).squeeze(-1)             # [N]
            point_by_acc = ray_bundle.origins + ray_bundle.directions * t_peak_acc[..., None]  # [N,3]

            # By density
            idx_sigma = torch.argmax(param.sigma, dim=-1, keepdim=True)             # [N,1]
            t_peak_sigma = torch.gather(param.t_mid, -1, idx_sigma).squeeze(-1)             # [N]
            point_by_sigma = ray_bundle.origins + ray_bundle.directions * t_peak_sigma[..., None]  # [N,3]

            # By depth
            depth_point = ray_bundle.origins + view_direction * param.depth

            # By expected depth
            exp_depth_point = ray_bundle.origins + view_direction * param.exp_depth

            # Filter by oTransmittance
            oTransmittance_mask = param.oTransmittance.max(dim=-1).values > weight_threshold

            # Filter by opacity
            rgba_mask = param.rgba[..., -1] > 0.5

            # Filter by acc
            acc_mask = param.accumulation.max(dim=-1).values > acc_threshold

            # Filter by density
            sigma_peak = param.sigma.max(dim=-1).values             # [N] per-ray max density
            sigma_cutoff = sigma_peak.max() * sigma_threshold       # scalar tensor
            sigma_mask = sigma_peak > sigma_cutoff



  

def generate_fruit_proposal_radiance_cloud(
    pipeline: Pipeline,
    num_points: int = 1500000,
    weight_threshold: float = 0.3,
    semantic_threshold: float = 0.5,
    filter_by_semantics: bool = False,
    semantic_output_name: str = "semantic_labels",
    depth_output_name: str = "depth",
    crop_obb: Optional[OrientedBox] = None,
) -> Dict[str, torch.Tensor]:
    """Generate a radiance field dataset from a NeRF model.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        semantic_output_name: Name of the semantic output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        crop_obb: Optional oriented bounding box to crop points.

    Returns:
        A dictionary containing all radiance field data.
    """

    # Initialize progress bar
    progress = Progress(
        TextColumn(":cloud: Computing Semantic Field :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )

    points          = []
    inv_points      = []
    depths          = []
    origins         = []
    directions      = []
    semantics_labels = []
    inv_semantics_labels = []
    semantics_logits = []
    inv_semantics_logits = []

    with progress as progress_bar:
        task = progress_bar.add_task("Generating Radiance Field", total=num_points)
        while not progress_bar.finished:

            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                assert isinstance(ray_bundle, RayBundle)
                outputs = pipeline.model(ray_bundle)

            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)

            if semantic_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {semantic_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --semantic_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)

            if "weights" not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find weights in the model outputs", justify="center")
                CONSOLE.print(f"Keep sure that you are using FruitProposal framework", justify="center")
                sys.exit(1)

            semantic_labels = outputs[semantic_output_name]             # [N]
            inv_semantic_labels = outputs["semantic_labels_inverted"]   # [N]
            weights         = outputs["weights"]                        # [N,S]
            t_mid           = outputs["t_mid"].squeeze(-1)              # TODO delete           # [N,S]

            # per-sample 3D positions: [N,S,3]
            # per-ray weighted depth
            # weights: [N,S], t_mid: [N,S]
            idx = torch.argmax(weights, dim=-1, keepdim=True)             # [N,1]
            inv_idx = torch.argmax(weights, dim=-1, keepdim=True)   # [N,1]
            t_peak = torch.gather(t_mid, -1, idx).squeeze(-1)             # [N]
            inv_t_peak = torch.gather(t_mid, -1, inv_idx).squeeze(-1)       # [N]
            p_peak = ray_bundle.origins + ray_bundle.directions * t_peak[..., None]  # [N,3]
            inv_p_peak = ray_bundle.origins + ray_bundle.directions * inv_t_peak[..., None]  # [N,3]

            # Use density weights to choose points along the ray, but gate inclusion by semantic prediction and confidence
            probs = torch.softmax(outputs["semantics"], dim=-1)
            inv_probs = torch.softmax(outputs["semantics_inverted"], dim=-1)

            weight_mask = weights.max(dim=-1).values > weight_threshold

            if filter_by_semantics:
                # assume class index 1 corresponds to foreground/kept class
                fg_conf = probs[:, 1]
                inv_fg_conf = inv_probs[:, 1]
                mask = weight_mask & (semantic_labels == 1) & (fg_conf > semantic_threshold)
                inv_mask = weight_mask & (inv_semantic_labels == 1) & (inv_fg_conf > semantic_threshold)
            else:
                mask = weight_mask
                inv_mask = weight_mask
            p_peak = p_peak[mask]
            inv_p_peak = inv_p_peak[inv_mask]
            semantic_labels = semantic_labels[mask]
            inv_semantic_labels = inv_semantic_labels[inv_mask]
            probs = probs[mask]
            inv_probs = inv_probs[inv_mask]
            if crop_obb is not None:
                mask = crop_obb.within(p_peak)
                inv_mask = crop_obb.within(inv_p_peak)
                p_peak = p_peak[mask]
                inv_p_peak = inv_p_peak[inv_mask]
                semantic_labels = semantic_labels[mask]
                inv_semantic_labels = inv_semantic_labels[inv_mask]
                probs = probs[mask]
                inv_probs = inv_probs[inv_mask]

            # per-ray mask, then apply to *all* per-ray arrays
            points.append(p_peak.cpu())
            inv_points.append(inv_p_peak.cpu())
            semantics_labels.append(semantic_labels.cpu())
            inv_semantics_labels.append(inv_semantic_labels.cpu())
            semantics_logits.append(probs.cpu())
            inv_semantics_logits.append(inv_probs.cpu())
            depths.append(outputs["depth"].cpu())
            origins.append(ray_bundle.origins.cpu())
            directions.append(ray_bundle.directions.cpu())

            progress.advance(task, p_peak.shape[0])

    # Combine lists into tensors on the CPU
    radiance_field_data = {
        "points": torch.cat(points, dim=0),
        "inv_points": torch.cat(inv_points, dim=0),
        "semantic_labels": torch.cat(semantics_labels, dim=0),
        "inv_semantic_labels": torch.cat(inv_semantics_labels, dim=0),
        "semantic_probs": torch.cat(semantics_logits, dim=0),
        "inv_semantic_probs": torch.cat(inv_semantics_logits, dim=0),
        "depth": torch.cat(depths, dim=0),
        "origins": torch.cat(origins, dim=0),
        "directions": torch.cat(directions, dim=0),
        "weight_threshold": weight_threshold,
        "semantic_threshold": semantic_threshold,
        "filter_by_semantics": filter_by_semantics,
    }

    return radiance_field_data

def generate_semantics_sample_point_cloud(
    pipeline: Pipeline,
    num_points: int = 1000000,
    remove_outliers: bool = True,
    estimate_normals: bool = False,
    reorient_normals: bool = False,
    rgb_output_name: str = "rgb",
    semantic_output_name: str = "semantics",
    semantics_colormap_output_name: str = "semantics_colormap",
    sampled_point_position_output_name: str = "point_location",
    depth_output_name: str = "depth",
    normal_output_name: Optional[str] = None,
    crop_obb: Optional[OrientedBox] = None,
    std_ratio: float = 10.0,
) -> o3d.geometry.PointCloud:

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )

    points = []
    rgbs = []
    normals = []
    view_directions = []

    points_sem = []
    points_only_sem = []
    points_den = []
    points_sem_colormap = []
    color_semantics = []
    color_only_semantics = []
    color_semantics_colormap = []
    densities = []

    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            normal = None

            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                assert isinstance(ray_bundle, RayBundle)
                outputs = pipeline.model(ray_bundle)

            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            
            if semantic_output_name not in outputs:
                return NotImplementedError
            
            if semantics_colormap_output_name not in outputs:
                return NotImplementedError

            if sampled_point_position_output_name not in outputs:
                return NotImplementedError

            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)

            rgba = pipeline.model.get_outputs(outputs, rgb_output_name)
            depth = outputs[depth_output_name]
            

            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                normal = outputs[normal_output_name]
                assert (
                    torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
                ), "Normal values from method output must be in [0, 1]"
                normal = (normal * 2.0) - 1.0
            

            point = ray_bundle.origins + ray_bundle.directions * depth
            view_direction = ray_bundle.directions

            # Filter points with opacity lower than 0.5
            mask = rgba[..., -1] > 0.5
            point = point[mask]
            view_direction = view_direction[mask]
            rgb = rgba[mask][..., :3]
            if normal is not None:
                normal = normal[mask]

            if crop_obb is not None:
                mask = crop_obb.within(point)
                point = point[mask]
                rgb = rgb[mask]
                view_direction = view_direction[mask]
                if normal is not None:
                    normal = normal[mask]

            points.append(point)
            rgbs.append(rgb)
            view_directions.append(view_direction)
            if normal is not None:
                normals.append(normal)
            progress.advance(task, point.shape[0])
    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)
    view_directions = torch.cat(view_directions, dim=0).cpu()

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())

    ind = None
    if remove_outliers:
        CONSOLE.print("Cleaning Point Cloud")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")
        if ind is not None:
            view_directions = view_directions[ind]

    # either estimate_normals or normal_output_name, not both
    if estimate_normals:
        if normal_output_name is not None:
            CONSOLE.rule("Error", style="red")
            CONSOLE.print("Cannot estimate normals and use normal_output_name at the same time", justify="center")
            sys.exit(1)
        CONSOLE.print("Estimating Point Cloud Normals")
        pcd.estimate_normals()
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")
    elif normal_output_name is not None:
        normals = torch.cat(normals, dim=0)
        if ind is not None:
            # mask out normals for points that were removed with remove_outliers
            normals = normals[ind]
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    # re-orient the normals
    if reorient_normals:
        normals = torch.from_numpy(np.array(pcd.normals)).float()
        mask = torch.sum(view_directions * normals, dim=-1) > 0
        normals[mask] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    return pcd


def render_trajectory(
    pipeline: Pipeline,
    cameras: Cameras,
    rgb_output_name: str,
    depth_output_name: str,
    rendered_resolution_scaling_factor: float = 1.0,
    disable_distortion: bool = False,
    return_rgba_images: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Helper function to create a video of a trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        disable_distortion: Whether to disable distortion.
        return_rgba_images: Whether to return RGBA images (default RGB).

    Returns:
        List of rgb images, list of depth images.
    """
    images = []
    depths = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)

    progress = Progress(
        TextColumn(":cloud: Computing rgb and depth images :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(
                camera_indices=camera_idx, disable_distortion=disable_distortion
            ).to(pipeline.device)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if return_rgba_images:
                image = pipeline.model.get_rgba_image(outputs, rgb_output_name)
            else:
                image = outputs[rgb_output_name]
            images.append(image.cpu().numpy())
            depths.append(outputs[depth_output_name].cpu().numpy())
    return images, depths


def collect_camera_poses_for_dataset(
    dataset: Optional[InputDataset], camera_optimizer: Optional[CameraOptimizer] = None
) -> List[Dict[str, Any]]:
    """Collects rescaled, translated and optimised camera poses for a dataset.

    Args:
        dataset: Dataset to collect camera poses for.
        camera_optimizer: Camera optimizer that has been used for adjusting the poses

    Returns:
        List of dicts containing camera poses.
    """

    if dataset is None:
        return []

    cameras = dataset.cameras
    image_filenames = dataset.image_filenames

    frames: List[Dict[str, Any]] = []

    # new cameras are in cameras, whereas image paths are stored in a private member of the dataset
    for idx in range(len(cameras)):
        image_filename = image_filenames[idx]
        if camera_optimizer is None:
            transform = cameras.camera_to_worlds[idx].tolist()
        else:
            # print('exporting optimized camera pose for camera %d' % idx)
            camera = cameras[idx : idx + 1]
            assert camera.metadata is not None
            camera.metadata["cam_idx"] = idx
            transform = camera_optimizer.apply_to_camera(camera).tolist()[0]

        frames.append(
            {
                "file_path": str(image_filename),
                "transform": transform,
            }
        )

    return frames


def collect_camera_poses(pipeline: VanillaPipeline) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collects camera poses for train and eval datasets.

    Args:
        pipeline: Pipeline to evaluate with.

    Returns:
        List of train camera poses, list of eval camera poses.
    """

    train_dataset = pipeline.datamanager.train_dataset
    assert isinstance(train_dataset, InputDataset)

    eval_dataset = pipeline.datamanager.eval_dataset
    assert isinstance(eval_dataset, InputDataset)

    camera_optimizer = None
    if hasattr(pipeline.model, "camera_optimizer"):
        camera_optimizer = pipeline.model.camera_optimizer
        assert isinstance(camera_optimizer, CameraOptimizer)

    train_frames = collect_camera_poses_for_dataset(train_dataset, camera_optimizer)
    # Note: returning original poses, even if --eval-mode=all
    eval_frames = collect_camera_poses_for_dataset(eval_dataset)

    return train_frames, eval_frames
