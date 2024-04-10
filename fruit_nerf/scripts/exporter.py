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
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import (
    collect_camera_poses,
    generate_point_cloud,
    get_mesh_from_filename,
)
from nerfstudio.exporter.marching_cubes import (
    generate_mesh_with_multires_marching_cubes,
)
from nerfstudio.fields.sdf_field import SDFField
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.scripts.exporter import ExportPointCloud

from fruit_nerf.export.exporter_utils import sample_volume


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


@dataclass
class ExportSemanticPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Maximum of the bounding box, used if use_bounding_box is True."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    num_points_per_side: int = 1000
    """Number of points sampled at the initial side of the volume (cube)"""


    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        config, pipeline, _, _ = eval_setup(self.load_config, test_mode='inference')

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(pipeline.datamanager, VanillaDataManager)
        # assert pipeline.datamanager.train_pixel_sampler is not None
        # pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch
        pipeline.datamanager.config.eval_num_rays_per_batch = self.num_rays_per_batch

        pipeline.model.setup_inference(render_rgb=True, num_inference_samples=self.num_points_per_side)
        num_points = pipeline.datamanager.setup_inference(num_points=self.num_points_per_side,
                                                          aabb=(self.bounding_box_min, self.bounding_box_max))

        # Transform json
        with open(self.load_config.parent / 'dataparser_transforms.json', 'r') as fp:
            transform_json = json.load(fp)

        pcds = sample_volume(
            pipeline=pipeline,
            num_points=num_points,
            output_dir=self.output_dir,
            config=config,
            transform_json=transform_json
        )
        torch.cuda.empty_cache()

        os.makedirs(str(self.output_dir / config.load_dir.parts[-3]), exist_ok=True)

        # o3d.io.write_point_cloud(str(output_dir / config.load_dir.parts[-3] / 'semantic_colormap.ply'), pcd)
        CONSOLE.print("Saving Point Cloud...")

        for pcd_name in pcds.keys():
            pcd = pcds[pcd_name]['pcd']
            pcd_path = pcds[pcd_name]['path']
            o3d.io.write_point_cloud(pcd_path, pcd)

        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportSemanticPointCloud, tyro.conf.subcommand(name="semantic-pointcloud")],
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
