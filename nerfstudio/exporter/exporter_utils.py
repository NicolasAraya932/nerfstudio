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

import numpy as np
import pathlib
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

    train_frames = collect_camera_poses_for_dataset(train_dataset, camera_optimizer)
    # Note: returning original poses, even if --eval-mode=all
    eval_frames = collect_camera_poses_for_dataset(eval_dataset)

    return train_frames, eval_frames


def sample_volume(
        pipeline: Pipeline,
        num_points: int,
        output_dir: pathlib.Path = None,
        config=None,
        transform_json: dict = None
) -> dict:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points_per_side: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.
        output_dir: save pcds to output dir.

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

    points_sem = []
    points_only_sem = []
    points_den = []
    points_sem_colormap = []
    color_semantics = []
    color_only_semantics = []
    color_semantics_colormap = []
    densities = []

    rgb_flag = True
    # sample_points_along_edge = num_points_per_side # num_points_per_side
    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_sample_volume(0)
                outputs = pipeline.model(ray_bundle)

            # Sampled volume points
            sampled_point_position = outputs['point_location']
            points_3d = sampled_point_position.reshape((-1, 3))

            # Semantic & Density value
            semantic = outputs['semantics'].reshape((-1, 1)).repeat((1, 3))
            semantics_colormap = outputs['semantics_colormap'].reshape((-1, 1)).repeat((1, 3))
            density = outputs['density'].reshape((-1, 1)).repeat((1, 3))
            rgb = outputs['rgb'].reshape((-1, 3))

            # Mask irrelevant semantic masks and density values
            mask_sem = semantic >= 3  # 20
            mask_den = density >= 70  # 10
            mask_sem_colormap = semantics_colormap >= 0.999
            mask_only_sem = semantics_colormap >= 0.99  # 9

            # Semantic colormap
            points_3d_semantic_colormap = points_3d[
                mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            if rgb_flag:
                color_semantic_colormap = rgb[mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            else:
                color_semantic_colormap = semantics_colormap[
                    mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]

            color_semantic_colormap = torch.hstack([color_semantic_colormap, torch.sigmoid(
                semantic[mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)][:, 0]).unsqueeze(-1)])
            points_sem_colormap.append(points_3d_semantic_colormap.cpu())
            color_semantics_colormap.append(color_semantic_colormap.cpu())

            # Semantic
            points_3d_semantic = points_3d[mask_sem.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            if rgb_flag:
                color_semantic = rgb[mask_sem.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            else:
                color_semantic = semantic[mask_sem.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            color_semantic = torch.hstack([color_semantic, torch.sigmoid(
                semantic[mask_sem.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)][:, 0]).unsqueeze(-1)])
            points_sem.append(points_3d_semantic.cpu())  # & mask_den.sum(dim=1).to(bool)
            color_semantics.append(color_semantic.cpu())  # & mask_den.sum(dim=1).to(bool)

            # RGB
            points_3d_density = points_3d[mask_den.sum(dim=1).to(bool)]
            if rgb_flag:
                density_color = rgb[mask_den.sum(dim=1).to(bool)]
            else:
                density_color = density[mask_den.sum(dim=1).to(bool)]
            density_color = torch.hstack(
                [density_color, torch.sigmoid(density[mask_den.sum(dim=1).to(bool)][:, 0]).unsqueeze(-1)])

            # rgb_color = rgb[mask_den.sum(dim=1).to(bool)]
            points_den.append(points_3d_density.cpu())
            # densities.append(rgb_color.cpu())
            densities.append(density_color.cpu())

            if False:
                # Semantic only
                points_3d_only_semantic_colormap = points_3d[mask_only_sem.sum(dim=1).to(bool)]

                if rgb_flag:
                    sem_color_only = rgb[mask_only_sem.sum(dim=1).to(bool)]
                else:
                    sem_color_only = semantic[mask_only_sem.sum(dim=1).to(bool)]

                # sem_color_only = torch.hstack(
                #    [sem_color_only, torch.sigmoid(
                #    semantic[mask_only_sem.sum(dim=1).to(bool)][:, 0]).unsqueeze(-1)])

                points_only_sem.append(points_3d_only_semantic_colormap.cpu())
                color_only_semantics.append(sem_color_only.cpu())

            torch.cuda.empty_cache()
            progress.advance(task, sampled_point_position.shape[0])

    pcd_list = {}

    # Semantic Colormap
    points_sem_colormap = torch.cat(points_sem_colormap, dim=0)
    semantic_colormap_rgbs = torch.cat(color_semantics_colormap, dim=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sem_colormap.detach().double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(semantic_colormap_rgbs.detach().double().cpu().numpy()[:, :3])

    if True:
        T = np.eye(4)
        T[:3, :4] = np.asarray(transform_json['transform'])[:3, :4]
        T[:3, :3] = T[:3, :3]
        T[:3, 3] *= -1

        pcd = pcd.scale(1 / transform_json['scale'], center=np.asarray((0, 0, 0)))
        pcd = pcd.scale(2, center=np.asarray((0, 0, 0)))

    pcd_list.update(
        {'semantic_colormap': {
            'pcd': pcd,
            'path': str(output_dir / config.load_dir.parts[-3] / 'semantic_colormap.ply')
        }})

    # Semantic
    points_sem = torch.cat(points_sem, dim=0)
    semantic_rgbs = torch.cat(color_semantics, dim=0)
    if semantic_rgbs.shape[0] != 0:
        semantic_rgbs /= semantic_rgbs.max()  # Normalize to visualize as point cloud

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sem.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(semantic_rgbs.double().cpu().numpy()[:, :3])

    if True:
        T = np.eye(4)
        T[:3, :4] = np.asarray(transform_json['transform'])[:3, :4]
        T[:3, :3] = T[:3, :3]
        # T = T[np.array([0, 2, 1, 3]), :]
        T[:3, 3] *= -1
        #
        pcd = pcd.scale(1 / transform_json['scale'], center=np.asarray((0, 0, 0)))
        pcd = pcd.scale(2, center=np.asarray((0, 0, 0)))

    pcd_list.update({'semantic': {
        'pcd': pcd,
        'path': str(output_dir / config.load_dir.parts[-3] / 'semantic.ply')
    }})

    # Density
    points_den = torch.cat(points_den, dim=0)
    density_rgb = torch.cat(densities, dim=0)
    if density_rgb.shape[0] != 0:
        density_rgb /= density_rgb.max()  # Normalize to visualize as point cloud

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_den.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(density_rgb.double().cpu().numpy()[:, :3])

    if True:
        T = np.eye(4)
        T[:3, :4] = np.asarray(transform_json['transform'])[:3, :4]
        T[:3, :3] = T[:3, :3]
        # T = T[np.array([0, 2, 1, 3]), :]
        T[:3, 3] *= -1
        #
        pcd = pcd.scale(1 / transform_json['scale'], center=np.asarray((0, 0, 0)))
        pcd = pcd.scale(2, center=np.asarray((0, 0, 0)))
        # pcd = pcd.transform(T)
    #
    ## Cloud compare
    # T = np.asarray([[0.994, -0.007, 0.118, -0.159],
    #                [-0.008, 0.993, 0.127, -0.168],
    #                [-0.118, -0.127, 0.986, 0.007],
    #                [0.000, 0.000, 0.000, 1.000]])
    # pcd = pcd.transform(T)

    # o3d.visualization.draw_geometries([pcd])
    pcd_list.update({'density': {
        'pcd': pcd,
        'path': str(output_dir / config.load_dir.parts[-3] / 'density.ply')
    }})

    return pcd_list
