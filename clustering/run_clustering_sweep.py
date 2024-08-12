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
from clustering_base import (load_obj_file,
                             FruitClustering)
import json


class Clustering(FruitClustering):
    def __init__(self,
                 shape_method: Literal['distance', 'ransac', 'svd'] = 'distance',
                 template_path: Union[str, Path] = './clustering/apple_template.ply',
                 voxel_size_down_sample: float = 0.00005,
                 remove_outliers_nb_points: int = 800,
                 remove_outliers_radius: float = 0.02,
                 min_samples: int = 60,
                 apple_template_size: float = 0.8,
                 cluster_merge_distance: float = 0.04,
                 gt_cluster=None,
                 gt_count: int = None):
        super().__init__(voxel_size_down_sample=voxel_size_down_sample,
                         remove_outliers_nb_points=remove_outliers_nb_points,
                         remove_outliers_radius=remove_outliers_radius,
                         cluster_merge_distance=cluster_merge_distance)
        self.shape_method: Literal['distance', 'ransac'] = shape_method
        self.template_path = template_path

        self.min_samples = min_samples

        self.fruit_template = o3d.io.read_point_cloud(self.template_path)
        self.fruit_template = self.fruit_template.scale(apple_template_size, center=(0, 0, 0))
        self.fruit_template = self.fruit_template.translate(-self.fruit_template.get_center())
        self.fruit_alpha_shape_ = alphashape.alphashape(np.asarray(self.fruit_template.points), 10)
        self.fruit_alpha_shape = self.fruit_alpha_shape_.as_open3d.sample_points_uniformly(1000)
        # o3d.visualization.draw_geometries([self.fruit_template])

        self.gt_cluster = gt_cluster
        if self.gt_cluster:
            if "obj" in self.gt_cluster:
                self.gt_mesh, self.gt_cluster_center, self.gt_cluster_pcd = load_obj_file(gt_cluster)
                # self.gt_position = o3d.io.read_point_cloud(self.gt_cluster)
                # self.gt_position.paint_uniform_color([1,0,1])
                self.gt_position = o3d.geometry.PointCloud()
                self.gt_position.points = o3d.utility.Vector3dVector(np.vstack(self.gt_cluster_center))
                # self.gt_position = self.gt_cluster_center

            else:
                self.gt_position = o3d.io.read_line_set(self.gt_cluster)

        self.gt_count = gt_count


if __name__ == '__main__':

    from clustering.config_synthetic import (Apple_GT_1024x1024_300, Apple_SAM_1024x1024_300,
                                             Pear_GT_1024x1024_300, Pear_SAM_1024x1024_300,
                                             Plum_GT_1024x1024_300, Plum_SAM_1024x1024_300,
                                             Lemon_GT_1024x1024_300, Lemon_SAM_1024x1024_300,
                                             Peach_GT_1024x1024_300, Peach_SAM_1024x1024_300,
                                             Mango_GT_1024x1024_300, Mango_SAM_1024x1024_300)

    from clustering.config_real import (Baum_01_unet, Baum_01_unet_Big, Baum_01_SAM, Baum_01_SAM_Big,
                                        Baum_02_unet, Baum_02_unet_Big, Baum_02_SAM, Baum_02_SAM_Big,
                                        Baum_03_unet, Baum_03_unet_Big, Baum_03_SAM, Baum_03_SAM_Big)

    from clustering.config_real import Fuji_unet, Fuji_unet_big, Fuji_sam, Fuji_sam_big

    Fuji_sam_sweep = {
        "path": "/media/se86kimy/DATA/eval/fuji/sam/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": [50, 60, 75],
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": [0.015, 0.02],
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.2,
        "min_samples": 100,
        'template_path': './clustering/apple_template.ply',
        'apple_template_size': [0.9, 1, 1.1],
        "gt_cluster": "/media/se86kimy/DATA/eval/fuji/data/lineset_aligned.ply",
        "gt_count": 1455
    }

    Fuji_sam_big_sweep = {
        "path": "/media/se86kimy/DATA/eval/fuji/sam_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": [50, 60, 75],
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": [0.015, 0.02],
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.2,
        "min_samples": 100,
        'template_path': './clustering/apple_template.ply',
        'apple_template_size': [0.9, 1, 1.1],
        "gt_cluster": "/media/se86kimy/DATA/eval/fuji/data/lineset_aligned.ply",
        "gt_count": 1455
    }

    Fuji_unet_sweep = {
        "path": "/media/se86kimy/DATA/eval/fuji/unet/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": [50, 60, 75],
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": [0.015, 0.02, 0.025],
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.2,
        "min_samples": 100,
        'template_path': './clustering/apple_template.ply',
        'apple_template_size': [0.9, 1, 1.1],
        "gt_cluster": "/media/se86kimy/DATA/eval/fuji/data/lineset_aligned.ply",
        "gt_count": 1455
    }

    Fuji_unet_big_sweep = {
        "path": "/media/se86kimy/DATA/eval/fuji/unet_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": [50, 60, 75],
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": [0.015, 0.02, 0.025],
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.2,
        "min_samples": 100,
        'template_path': './clustering/apple_template.ply',
        'apple_template_size': [0.9, 1, 1.1],
        "gt_cluster": "/media/se86kimy/DATA/eval/fuji/data/lineset_aligned.ply",
        "gt_count": 1455
    }

    Baums = {
        "Fuji_sam_sweep": Fuji_sam_sweep,
        "Fuji_sam_big_sweep": Fuji_sam_big_sweep,
        "Fuji_unet_sweep": Fuji_unet_sweep,
        "Fuji_unet_big_sweep": Fuji_unet_big_sweep}

    results = {}

    cc = 0

    for Baum_name in Baums:
        Baum = Baums[Baum_name]
        results_intermediate = {}

        for remove_outliers_nb_points in Baum["remove_outliers_nb_points"]:
            for eps in Baum["eps"]:
                for apple_template_size in Baum["apple_template_size"]:

                    clustering = Clustering(remove_outliers_nb_points=remove_outliers_nb_points,
                                            remove_outliers_radius=Baum['remove_outliers_radius'],
                                            voxel_size_down_sample=Baum['down_sample'],
                                            template_path=Baum['template_path'],
                                            min_samples=Baum['min_samples'],
                                            apple_template_size=apple_template_size,
                                            gt_cluster=Baum['gt_cluster'],
                                            cluster_merge_distance=Baum['cluster_merge_distance'],
                                            gt_count=Baum['gt_count']
                                            )
                    count = clustering.count(pcd=Baum["path"], eps=eps, )

                    if Baum['gt_cluster']:
                        results_intermediate.update({cc: {
                            "path": Baum['path'],
                            'count': count,
                            'TP': clustering.true_positive,
                            'gt': clustering.gt_count,
                            'precision': clustering.precision,
                            'recall': clustering.recall,
                            'F1': clustering.F1,
                            'params': {
                                'remove_outliers_nb_points': remove_outliers_nb_points,
                                'eps': eps,
                                'apple_template_size': apple_template_size,
                            }
                        }})
                    else:
                        results_intermediate.update({cc: {
                            "path": Baum['path'],
                            'count': count,
                            'gt': clustering.gt_count,
                        }})
                    cc += 1

                    print(results_intermediate)
            print("\n --------------------------------- \n")

        results.update({Baum_name: results_intermediate})

        print(results)
        with open('./clustering/results_fuji_sweep.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4, default=str)
