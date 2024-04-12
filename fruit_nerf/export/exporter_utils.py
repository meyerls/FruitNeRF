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

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import pymeshlab
from pathlib import Path
import torch
from jaxtyping import Float
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn


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
