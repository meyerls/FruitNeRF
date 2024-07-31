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

    Baums = [
        # Fuji_unet, Fuji_unet_big,
        # Fuji_sam, Fuji_sam_big
        # Baum_01_unet, Baum_01_unet_Big, Baum_01_SAM, Baum_01_SAM_Big,
        # Baum_02_unet, Baum_02_unet_Big, Baum_02_SAM, Baum_02_SAM_Big,
        # Baum_03_unet, Baum_03_unet_Big, Baum_03_SAM, Baum_03_SAM_Big,
        Apple_GT_1024x1024_300, Apple_SAM_1024x1024_300,
        # Pear_GT_1024x1024_300, Pear_SAM_1024x1024_300,
        # Plum_GT_1024x1024_300, Plum_SAM_1024x1024_300,
        # Lemon_GT_1024x1024_300, Lemon_SAM_1024x1024_300,
        # Peach_GT_1024x1024_300, Peach_SAM_1024x1024_300,
        # Mango_GT_1024x1024_300, Mango_SAM_1024x1024_300
    ]

    results = {}

    for Baum in Baums:
        clustering = Clustering(shape_method="svd",
                                remove_outliers_nb_points=Baum['remove_outliers_nb_points'],
                                remove_outliers_radius=Baum['remove_outliers_radius'],
                                voxel_size_down_sample=Baum['down_sample'],
                                template_path=Baum['template_path'],
                                min_samples=Baum['min_samples'],
                                apple_template_size=Baum['apple_template_size'],
                                gt_cluster=Baum['gt_cluster'],
                                cluster_merge_distance=Baum['cluster_merge_distance'],
                                gt_count=Baum['gt_count']
                                )
        count = clustering.count(pcd=Baum["path"], eps=Baum['eps'], )

        if Baum['gt_cluster']:
            results.update({Baum['path']: {
                'count': count,
                'TP': clustering.true_positive,
                'gt': clustering.gt_count,
                'precision': clustering.precision,
                'recall': clustering.recall,
                'F1': clustering.F1,
            }})
        else:
            results.update({Baum['path']: {
                'count': count,
                'gt': clustering.gt_count,
            }})

        print(results)
        print("\n --------------------------------- \n")
    print(results)

    import json

    with open('./clustering/results_synthetic.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

"""

{'/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/01_apple/gt/semantic_colormap.ply': {'count': 280, 'gt': 283,
                                                                                                'precision': 1.0,
                                                                                                'recall': 0.9893992932862191,
                                                                                                'F1': 0.9946714031971582},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/01_apple/sam/semantic_colormap.ply': {'count': 282,
                                                                                                 'gt': 283,
                                                                                                 'precision': 0.9929078014184397,
                                                                                                 'recall': 0.9893992932862191,
                                                                                                 'F1': 0.9911504424778761},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/02_pear/gt/semantic_colormap.ply': {'count': 236, 'gt': 250,
                                                                                               'precision': 1.0,
                                                                                               'recall': 0.944,
                                                                                               'F1': 0.9711934156378601},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/02_pear/sam/semantic_colormap.ply': {'count': 229, 'gt': 250,
                                                                                                'precision': 1.0,
                                                                                                'recall': 0.916,
                                                                                                'F1': 0.9561586638830899},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/03_plum/gt/semantic_colormap.ply': {'count': 651, 'gt': 781,
                                                                                               'precision': 0.9738863287250384,
                                                                                               'recall': 0.8117797695262484,
                                                                                               'F1': 0.8854748603351955},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/03_plum/sam/semantic_colormap.ply': {'count': 315, 'gt': 781,
                                                                                                'precision': 1.0,
                                                                                                'recall': 0.4033290653008963,
                                                                                                'F1': 0.5748175182481752},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/05_lemon/gt/semantic_colormap.ply': {'count': 316, 'gt': 326,
                                                                                                'precision': 0.9936708860759493,
                                                                                                'recall': 0.9631901840490797,
                                                                                                'F1': 0.9781931464174455},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/05_lemon/sam/semantic_colormap.ply': {'count': 326,
                                                                                                 'gt': 326,
                                                                                                 'precision': 0.9815950920245399,
                                                                                                 'recall': 0.9815950920245399,
                                                                                                 'F1': 0.98159509202454},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/07_peach/gt/semantic_colormap.ply': {'count': 148, 'gt': 152,
                                                                                                'precision': 1.0,
                                                                                                'recall': 0.9736842105263158,
                                                                                                'F1': 0.9866666666666666},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/07_peach/sam/semantic_colormap.ply': {'count': 148,
                                                                                                 'gt': 152,
                                                                                                 'precision': 1.0,
                                                                                                 'recall': 0.9736842105263158,
                                                                                                 'F1': 0.9866666666666666},
 '/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/08_mango_tree_1024x1024_#300/fruit_nerf/semantic_colormap.ply':
     {'count': 926, 'gt': 1150, 'precision': 0.978401727861771, 'recall': 0.7878260869565218, 'F1': 0.8728323699421966},
 '/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/08_mango_tree_1024x1024_#300_SAM/fruit_nerf/semantic_colormap.ply': {
     'count': 807, 'gt': 1150, 'precision': 0.9888475836431226, 'recall': 0.6939130434782609, 'F1': 0.8155339805825244}}
"""

{'/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_01/unet/semantic_colormap_cropped.ply': {'count': 146,
                                                                                                         'gt': 179},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_01/unet_big/semantic_colormap_cropped.ply': {
     'count': 172, 'gt': 179},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_01/sam/semantic_colormap_cropped.ply': {'count': 147,
                                                                                                        'gt': 179},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_01/sam_big/semantic_colormap_cropped.ply': {
     'count': 173, 'gt': 179}}

{'/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_02/unet/semantic_colormap_cropped.ply': {'count': 87,
                                                                                                         'gt': 113},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_02/unet_big/semantic_colormap_cropped.ply': {
     'count': 113, 'gt': 113},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_02/sam/semantic_colormap_cropped.ply': {'count': 87,
                                                                                                        'gt': 113},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_02/sam_big/semantic_colormap_cropped.ply': {
     'count': 110, 'gt': 113}}

{'/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/unet/semantic_colormap_cropped.ply': {'count': 254,
                                                                                                         'gt': 291},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/unet_big/semantic_colormap_cropped.ply': {
     'count': 223, 'gt': 291},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/sam/semantic_colormap_cropped.ply': {'count': 174,
                                                                                                        'gt': 291},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/tree_03/sam_big/semantic_colormap_cropped.ply': {
     'count': 246, 'gt': 291}}

{'/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/fuji/unet/semantic_colormap_cropped.ply': {'count': 1140,
                                                                                                      'TP': 1067,
                                                                                                      'gt': 1455,
                                                                                                      'precision': 0.9359649122807018,
                                                                                                      'recall': 0.7333333333333333,
                                                                                                      'F1': 0.8223506743737958},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/fuji/unet_big/semantic_colormap_cropped.ply': {'count': 1263,
                                                                                                          'TP': 1165,
                                                                                                          'gt': 1455,
                                                                                                          'precision': 0.9224069675376089,
                                                                                                          'recall': 0.8006872852233677,
                                                                                                          'F1': 0.8572479764532743},
 '/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/eval/fuji/sam/semantic_colormap_cropped.ply': {'count': 799,
                                                                                                     'TP': 775,
                                                                                                     'gt': 1455,
                                                                                                     'precision': 0.9699624530663329,
                                                                                                     'recall': 0.5326460481099656,
                                                                                                     'F1': 0.6876663708961844}}
