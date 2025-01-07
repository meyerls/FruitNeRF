"""
Copyright (c) 2023 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""
import copy
# Built-in/Generic Imports
from typing import Union, Literal

# Libs
import open3d as o3d
import numpy as np
from pathlib import Path
import alphashape

# Own modules
from clustering_base import load_obj_file, FruitClustering


class AppleClustering(FruitClustering):
    def __init__(self,
                 shape_method: Literal['distance', 'ransac', 'svd'] = 'distance',
                 template_path: Union[str, Path] = './clustering/lemon_template.ply',
                 voxel_size_down_sample: float = 0.00005,
                 remove_outliers_nb_points: int = 800,
                 remove_outliers_radius: float = 0.02,
                 min_samples: int = 60,
                 apple_template_size: float = 0.8,
                 gt_cluster=None):
        super().__init__(voxel_size_down_sample=voxel_size_down_sample,
                         remove_outliers_nb_points=remove_outliers_nb_points,
                         remove_outliers_radius=remove_outliers_radius)
        self.shape_method: Literal['distance', 'ransac'] = shape_method
        self.apple_template_path = './clustering/lemon_template.ply'

        self.min_samples = min_samples

        self.fruit_template = o3d.io.read_point_cloud(self.apple_template_path)
        self.fruit_template = self.fruit_template.scale(apple_template_size, center=(0, 0, 0))
        self.fruit_template = self.fruit_template.translate(-self.fruit_template.get_center())
        self.fruit_alpha_shape_ = alphashape.alphashape(np.asarray(self.fruit_template.points), 10)
        self.fruit_alpha_shape = self.fruit_alpha_shape_.as_open3d.sample_points_uniformly(1000)
        # o3d.visualization.draw_geometries([self.apple_template])

        self.gt_cluster = gt_cluster
        if self.gt_cluster:
            self.gt_mesh, self.gt_cluster_center, self.gt_cluster_pcd = load_obj_file(gt_cluster)

        # self.chamferDist = ChamferDistance()


if __name__ == '__main__':

    Lemon_GT_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/05_lemon/gt/semantic_colormap.ply",
        "remove_outliers_nb_points": 200,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "min_samples": 100,
        'apple_template_size': 1.1
    }

    Lemon_SAM_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/05_lemon/sam/semantic_colormap.ply",
        "remove_outliers_nb_points": 200,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "min_samples": 100,
        'apple_template_size': 1.1
    }

    Baums = [
        Lemon_GT_1024x1024_300,
        Lemon_SAM_1024x1024_300
    ]

    results = {}

    for Baum in Baums:
        apple_clustering = AppleClustering(shape_method="svd",
                                           remove_outliers_nb_points=Baum['remove_outliers_nb_points'],
                                           remove_outliers_radius=Baum['remove_outliers_radius'],
                                           voxel_size_down_sample=Baum['down_sample'],
                                           min_samples=Baum['min_samples'],
                                           apple_template_size=Baum['apple_template_size'],
                                           gt_cluster="/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/05_lemon/fruits.obj"
                                           )
        count = apple_clustering.count(pcd=Baum["path"], eps=Baum['eps'], )

        results.update({Baum['path']: count})

        print(results)
    print(results)
