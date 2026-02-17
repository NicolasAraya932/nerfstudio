from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Literal, Type, Union, Optional
import torch

from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager, ParallelDataManagerConfig
from nerfstudio.data.datasets.fruit_dataset import FruitDataset
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.orthographic_ray_generators import OrthographicRayGenerator

# Utility functions (copied from FruitDataManager)
def get_corners_of_aabb(aabb, device):
    """
    Get the 3D locations of the corners of an Axis-Aligned Bounding Box (AABB).
    """
    min_coords = aabb[0]
    max_coords = aabb[1]
    corners = torch.asarray([
        [min_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],
    ], device=device)
    return corners

def sample_surface_points(aabb, n, device, noise=False):
    """
    Sample points on a single surface of the AABB.
    
    Args:
        aabb: Tensor with shape (2, 3) where first row is min coords and second is max coords.
        n: Number of points to sample along each axis.
        device: Device on which to perform computations.
        noise: Whether to add noise (unused in this simple version).
        
    Returns:
        A tuple (surface_points_tensor, plane_vector)
    """
    # Use first three corners as a starting point.
    corner_1 = aabb[0]
    corner_2 = aabb[1]
    corner_3 = aabb[2]
    dx_y_z = torch.abs(torch.max(aabb, axis=0).values - torch.min(aabb, axis=0).values)
    # Determine the axis that is constant across the selected corners.
    constant_axis_part_pos = int(torch.argmax(torch.logical_and((corner_1 == corner_2), (corner_2 == corner_3)).to(int)))
    
    # Create linspaces for the two varying axes.
    start_x_pos = torch.argmax(torch.abs(corner_1 - corner_2))
    x = torch.linspace(corner_1[start_x_pos], corner_2[start_x_pos],
                       int(dx_y_z[0] / dx_y_z[constant_axis_part_pos] * n),
                       dtype=torch.float32, device=device)
    start_y_pos = torch.argmax(torch.abs(corner_1 - corner_3))
    y = torch.linspace(corner_1[start_y_pos], corner_3[start_y_pos],
                       int(dx_y_z[1] / dx_y_z[constant_axis_part_pos] * n),
                       dtype=torch.float32, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    # Set the constant axis coordinate to the corresponding value from corner_3.
    surface_points = torch.column_stack(
        (xx.flatten(), yy.flatten(), torch.full_like(xx.flatten(), corner_3[constant_axis_part_pos]))
    )
    surface_points_tensor = surface_points.clone()
    # Use the last corner to create a plane vector.
    corner_4 = aabb[-1]
    plane_vector = torch.asarray(
        [[0, 0, torch.sign(corner_4[constant_axis_part_pos]) *
          (torch.abs(corner_1[constant_axis_part_pos]) + torch.abs(corner_4[constant_axis_part_pos]))]],
        dtype=torch.float32,
        device=device,
    )
    return surface_points_tensor, plane_vector

# New config for the ParallelFruitDataManager.
@dataclass
class ParallelFruitDataManagerConfig(ParallelDataManagerConfig):
    _target: Type = field(default_factory=lambda: ParallelFruitDataManager)

class ParallelFruitDataManager(ParallelDataManager):
    """
    A Parallel Data Manager that incorporates FruitDataManager features.
    It uses a FruitDataset for train/eval and adds an inference setup with orthographic ray generation.
    """
    config: ParallelFruitDataManagerConfig
    train_dataset: FruitDataset
    eval_dataset: FruitDataset

    def __init__(
        self,
        config: ParallelFruitDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)
    
    def create_train_dataset(self) -> FruitDataset:
        """Sets up the training dataset using FruitDataset."""
        return FruitDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> FruitDataset:
        """Sets up the evaluation dataset using FruitDataset."""
        return FruitDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def setup_inference(self, aabb: torch.Tensor, num_points: int) -> int:
        """
        Set up the inference mode using orthographic ray generation.
        
        Args:
            aabb: Tensor of shape (2, 3) representing the axis-aligned bounding box.
            num_points: Number of points to sample along each edge.
        
        Returns:
            The total number of rays generated (based on the number of surface points).
        """
        corners = get_corners_of_aabb(aabb=aabb, device=self.device)
        surface_points, plane_vector = sample_surface_points(corners, n=num_points, device=self.device, noise=False)
        self.orthographic_ray_generator = OrthographicRayGenerator(
            surface_points=surface_points,
            plane_normal=plane_vector,
            ray_batch_size=self.config.eval_num_rays_per_batch,
            device=self.device,
            aabb=aabb,
        )
        num_rays = surface_points.shape[0]
        return num_rays

    def next_sample_volume(self, step: int) -> Tuple[RayBundle, Optional[Dict[str, Any]]]:
        """
        Returns the next batch of rays generated using the orthographic ray generator.
        
        Args:
            step: The current step (used here to update a counter for sampling).
        
        Returns:
            A tuple containing a RayBundle and an optional batch (None in this case).
        """
        # Assume self.train_count is maintained elsewhere in the pipeline.
        self.train_count += 1
        ray_bundle = self.orthographic_ray_generator(count=self.train_count)
        return ray_bundle, None
