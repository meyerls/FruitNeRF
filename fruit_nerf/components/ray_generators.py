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
Ray generator.
"""
from torch import Tensor, nn
import torch

from nerfstudio.cameras.rays import RayBundle


class OrthographicRayGenerator(nn.Module):
    """torch.nn Module for generating rays.
    This class is the interface between the scene's cameras/camera optimizer and the ray sampler.

    Args:
        cameras: Camera objects containing camera info.
        pose_optimizer: pose optimization module, for optimizing noisy camera intrinsics/extrinsics.
    """

    image_coords: Tensor

    def __init__(self, surface_points, plane_normal, ray_batch_size, device, aabb) -> None:
        super().__init__()
        self.surface_points = surface_points
        self.surface_normal = torch.nn.functional.normalize(plane_normal).to(device)
        self.surface_vector_norm = torch.linalg.norm(plane_normal).to(device)

        self.ray_batch_size = ray_batch_size
        self.device = device

        self.aabb = aabb

    def forward(self, count) -> RayBundle:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indices for target rays.
        """
        start = self.ray_batch_size * (count - 1)
        end = self.ray_batch_size * count

        if self.ray_batch_size * count >= self.surface_points.shape[0]:
            end = self.surface_points.shape[0]

        num_points = self.surface_points[start:end].shape[0]
        ray_bundle = RayBundle(origins=self.surface_points[start:end],
                               directions=self.surface_normal.repeat(num_points, 1).to(self.device),
                               pixel_area=torch.zeros(num_points, 1).to(self.device),
                               nears=torch.zeros(num_points, 1).to(self.device),
                               fars=torch.ones(num_points, 1).to(self.device) * self.surface_vector_norm
                               )

        return ray_bundle
