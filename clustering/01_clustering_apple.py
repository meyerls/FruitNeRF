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
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
from pathlib import Path
import alphashape
import robust_laplacian
import polyscope as ps
import scipy.sparse.linalg as sla
# from chamferdist import ChamferDistance
from hausdorff import hausdorff_distance
from sklearn.mixture import GaussianMixture
from sklearn.cluster import spectral_clustering, SpectralClustering, AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_kernels
import torch
from tqdm import tqdm

# Own modules
from clustering_base import (create_sphere,
                             check_point_cloud_for_spherical_shape,
                             draw_registration_result,
                             load_obj_file,
                             FruitClustering)


class AppleClustering(FruitClustering):
    def __init__(self,
                 shape_method: Literal['distance', 'ransac', 'svd'] = 'distance',
                 template_path: Union[str, Path] = './clustering/apple_template.ply',
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
            self.gt_mesh, self.gt_cluster_center, self.gt_cluster_pcd = load_obj_file(gt_cluster)


if __name__ == '__main__':

    Baum_01_SAM = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_01/sam/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.015,
        "down_sample": 0.001,
        "eps": 0.02,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        "template_path": './clustering/apple_template.ply',
        'apple_template_size': 1,
        'gt_cluster': None,
        'gt_count': 179,
    }

    Baum_01_SAM_Big = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_01/sam_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.015,
        "down_sample": 0.001,
        "eps": 0.02,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        "template_path": './clustering/apple_template.ply',
        'apple_template_size': 1,
        'gt_cluster': None,
        'gt_count': 179,
    }

    Baum_01_unet = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_01/unet/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.015,
        "down_sample": 0.001,
        "eps": 0.02,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        "template_path": './clustering/apple_template.ply',
        'apple_template_size': 1,
        'gt_cluster': None,
        'gt_count': 179,
    }

    Baum_01_unet_Big = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_01/unet_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.015,
        "down_sample": 0.001,
        "eps": 0.02,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        "template_path": './clustering/apple_template.ply',
        'apple_template_size': 1,
        'gt_cluster': None,
        'gt_count': 179,
    }






    Baum_02_SAM = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_02/sam/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 70,
        "remove_outliers_radius": 0.02,
        "down_sample": 0.001,
        "eps": 0.04,
        "min_samples": 100,  # 70
        'apple_template_size': 1.3
    }

    Baum_02_SAM_Big = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_02/sam_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 70,
        "remove_outliers_radius": 0.02,
        "down_sample": 0.001,
        "eps": 0.04,
        "min_samples": 100,  # 70
        'apple_template_size': 1.3
    }

    Baum_02_unet = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_02/unet/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 70,
        "remove_outliers_radius": 0.02,
        "down_sample": 0.001,
        "eps": 0.04,
        "min_samples": 100,  # 70
        'apple_template_size': 1.3
    }

    Baum_02_unet_Big = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_02/unet_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 70,
        "remove_outliers_radius": 0.02,
        "down_sample": 0.001,
        "eps": 0.04,
        "min_samples": 100,  # 70
        'apple_template_size': 1.3
    }

    Baum_03_SAM = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/sam/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 110,
        "remove_outliers_radius": 0.02,
        "down_sample": 0.001,
        "eps": 0.025,
        "min_samples": 65,
        'apple_template_size': 1.3
    }

    Baum_03_SAM_Big = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/sam_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 110,
        "remove_outliers_radius": 0.02,
        "down_sample": 0.001,
        "eps": 0.025,
        "min_samples": 65,
        'apple_template_size': 1.3
    }

    Baum_03_unet = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/unet/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 110,
        "remove_outliers_radius": 0.02,
        "down_sample": 0.001,
        "eps": 0.025,
        "min_samples": 65,
        'apple_template_size': 1.3
    }

    Baum_03_unet_Big = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/unet_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 110,
        "remove_outliers_radius": 0.02,
        "down_sample": 0.001,
        "eps": 0.025,
        "min_samples": 65,
        'apple_template_size': 1.3
    }

    #Apple_GT_1024x1024_300 = {
    #    "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/01_apple/gt/semantic_colormap.ply",
    #    "remove_outliers_nb_points": 200,
    #    "remove_outliers_radius": 0.01,
    #    "down_sample": 0.001,
    #    "eps": 0.01,
    #    "min_samples": 100,
    #    'apple_template_size': 0.7
    #}

    #Apple_SAM_1024x1024_300 = {
    #    "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/01_apple/sam/semantic_colormap.ply",
    #    "remove_outliers_nb_points": 200,
    #    "remove_outliers_radius": 0.01,
    #    "down_sample": 0.001,
    #    "eps": 0.01,
    #    "min_samples": 100,
    #    'apple_template_size': 0.7
    #}

    Baums = [
        # Baum_01_SAM,
        # Baum_01_SAM_Big,
        # Baum_01_unet,
        # Baum_01_unet_Big,
        # Baum_02_SAM,
        # Baum_02_SAM_Big,
        # Baum_02_unet,
        # Baum_02_unet_Big,
        # Baum_03_SAM,
        # Baum_03_SAM_Big,
        # Baum_03_unet,
        # Baum_03_unet_Big,
        Apple_GT_1024x1024_300,
        Apple_SAM_1024x1024_300
    ]

    results = {}

    for Baum in Baums:
        apple_clustering = AppleClustering(shape_method="svd",
                                           remove_outliers_nb_points=Baum['remove_outliers_nb_points'],
                                           remove_outliers_radius=Baum['remove_outliers_radius'],
                                           voxel_size_down_sample=Baum['down_sample'],
                                           template_path=Baum['template_path'],
                                           min_samples=Baum['min_samples'],
                                           apple_template_size=Baum['apple_template_size'],
                                           gt_cluster="/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/01_apple/fruits.obj"
                                           )
        count = apple_clustering.count(pcd=Baum["path"], eps=Baum['eps'], )

        results.update({Baum['path']: count})

        print(results)
    print(results)

# {'/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_01/sam/semantic_colormap_cropped.ply': 133, '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_01/sam_big/semantic_colormap_cropped.ply': 165, '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_01/unet/semantic_colormap_cropped.ply': 140, '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_01/unet_big/semantic_colormap_cropped.ply': 163}
# {'/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_02/sam/semantic_colormap_cropped.ply': 99, '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_02/sam_big/semantic_colormap_cropped.ply': 126, '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_02/unet/semantic_colormap_cropped.ply': 65, '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_02/unet_big/semantic_colormap_cropped.ply': 87}
# {'/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/sam/semantic_colormap_cropped.ply': 215, '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/sam_big/semantic_colormap_cropped.ply': 284, '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/unet/semantic_colormap_cropped.ply': 299, '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/unet_big/semantic_colormap_cropped.ply': 252}
# {'/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/sam/semantic_colormap_cropped.ply': 216, '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/sam_big/semantic_colormap_cropped.ply': 286, '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/unet/semantic_colormap_cropped.ply': 301, '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/unet_big/semantic_colormap_cropped.ply': 256}
