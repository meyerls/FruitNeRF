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
from threading import Thread


# Own modules
# ...

def create_sphere(center, radius):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere = sphere.translate(center)

    return sphere


def check_point_cloud_for_spherical_shape(point_cloud):
    points = np.asarray(point_cloud.points)
    center, cov_mat = point_cloud.compute_mean_and_covariance()
    covariances = np.linalg.norm(cov_mat, axis=1)
    # radius = np.mean(covariances)

    # https://math.stackexchange.com/questions/131675/ratio-of-largest-eigenvalue-to-sum-of-eigenvalues-where-to-read-about-it
    proportion_variation_1 = covariances[0] / (covariances.sum())
    proportion_variation_2 = covariances[1] / (covariances.sum())
    proportion_variation_3 = covariances[2] / (covariances.sum())
    spherical_shape = abs(proportion_variation_1 - 0.33333) <= 0.1 and abs(
        proportion_variation_2 - 0.33333) <= 0.1 and abs(proportion_variation_3 - 0.33333) <= 0.1

    dist_from_center_to_points = np.linalg.norm(center - points, axis=1)
    radius = np.max(dist_from_center_to_points)
    radius_vis = dist_from_center_to_points.mean()

    return spherical_shape, radius, radius_vis, center


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def load_obj_file(filename):
    vertices = []
    faces = []

    clusters = []
    init_flag = True

    with open(filename, 'r') as obj_file:
        for line in obj_file:
            # Split the line into components
            parts = line.strip().split()

            if len(parts) > 0:
                # new object data starts with 'o'
                if parts[0] == 'o':
                    if init_flag:
                        init_flag = False
                    else:
                        clusters.append(vertices_cluster)
                    vertices_cluster = []
                # Vertex data starts with 'v'
                elif parts[0] == 'v':
                    vertex = list(map(float, parts[1:]))
                    vertices.append(vertex)
                    vertices_cluster.append(vertex)
                # Face data starts with 'f'
                elif parts[0] == 'f':
                    face = [int(index.split('/')[0]) - 1 for index in
                            parts[1:]]  # Subtract 1 because OBJ indices start from 1
                    faces.append(face)
        clusters.append(vertices_cluster)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    cluster_center = []
    cluster_pcd = []
    for c in clusters:
        cluster_center.append(np.asarray(c).mean(axis=0))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(c)
        cluster_pcd.append(pcd)

    return mesh, cluster_center, cluster_pcd


class FruitClustering(object):
    def __init__(self,
                 voxel_size_down_sample: float = 0.00005,
                 remove_outliers_nb_points: int = 800,
                 remove_outliers_radius: float = 0.02,
                 min_samples: int = 60,
                 cluster_merge_distance: float = 0.04,
                 minimum_size_factor: float = 0.3,
                 template_path: Union[
                     Path, str] = '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
                 apple_template_size: float = 0.8,
                 gt_cluster=None,
                 gt_count=-1):
        super().__init__()

        self.voxel_size_down_sample: float = voxel_size_down_sample
        self.remove_outliers_nb_points: int = remove_outliers_nb_points
        self.remove_outliers_radius: float = remove_outliers_radius

        self.pcd_path: Union[str, None] = None
        self.pcd: Union[o3d.geometry.PointCloud, None] = None
        self.pcd_down_sampled: Union[o3d.geometry.PointCloud, None] = None
        self.pcd_downsampled_cleaned: Union[o3d.geometry.PointCloud, None] = None

        self.debug = False

        self.apple_template_path = template_path

        self.min_samples = min_samples
        self.cluster_merge_distance = cluster_merge_distance
        self.minimum_size_factor = minimum_size_factor
        self.apple_template = o3d.io.read_point_cloud(self.apple_template_path)
        self.apple_template = self.apple_template.scale(apple_template_size, center=(0, 0, 0))
        self.apple_template = self.apple_template.translate(-self.apple_template.get_center())
        self.apple_alpha_shape_ = alphashape.alphashape(np.asarray(self.apple_template.points), 10)
        self.apple_alpha_shape = self.apple_alpha_shape_.as_open3d.sample_points_uniformly(1000)
        # o3d.visualization.draw_geometries([self.apple_template])

        self.gt_cluster = gt_cluster
        self.gt_count = gt_count

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
                # T = np.asarray([[1.09, -0.02, 0.08,  -0.01],
                #                [0.02, 1.09, 0.01,  -5.92],
                #                [-0.08, -0.01, 1.09, 3.04],
                #                [0, 0, 0, 1]])
                # self.gt_position.transform(T)

    def voxel_down_sample(self, pcd):
        return pcd.voxel_down_sample(voxel_size=self.voxel_size_down_sample)

    def remove_outliers(self, pcd):
        return pcd.remove_radius_outlier(nb_points=self.remove_outliers_nb_points, radius=self.remove_outliers_radius)

    def pcd2points_and_color(self, pcd):
        p2d = np.asarray(pcd.points)
        c = np.asarray(pcd.colors)

        return p2d, c

    def visualize_clusters(self, X, labels, visualize=True):
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors_cluster = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors_cluster[labels < 0] = 0

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(X)
        new_pcd.colors = o3d.utility.Vector3dVector(colors_cluster[:, :3])
        if visualize:
            o3d.visualization.draw_geometries([new_pcd])

        return new_pcd

    def visualize_unassigned_clusters(self, unassigned_cluster_list, labels, X, window_name="Unassigned Clusters"):
        max_label = labels.max()

        colors_cluster = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors_cluster[labels < 0] = 0

        pcd_list = []
        for i in unassigned_cluster_list:
            cluster_points_idx = np.argwhere(labels == i)

            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(X[cluster_points_idx.flatten()])
            new_pcd.colors = o3d.utility.Vector3dVector(colors_cluster[:, :3][cluster_points_idx.flatten()])

            pcd_list.append(new_pcd)

        o3d.visualization.draw_geometries(pcd_list, window_name=window_name)

    def cluster(self, pcd, **kwargs):
        self.pcd = pcd
        self.pcd_down_sampled, _ = self.remove_outliers(pcd=self.pcd)
        self.pcd_downsampled_cleaned = self.voxel_down_sample(pcd=self.pcd_down_sampled)

        if np.asarray(self.pcd_downsampled_cleaned.points).shape[0] == 0:
            X = -1
            C = -1
            labels = -1

            return X, C, labels

        o3d.io.write_point_cloud(str(Path(self.pcd_path).parents[0] / 'semantic_cleaned_down_sampled.ply'),
                                 self.pcd_downsampled_cleaned)

        X, C = self.pcd2points_and_color(pcd=self.pcd_downsampled_cleaned)

        clustering = DBSCAN(eps=kwargs['eps'], min_samples=kwargs['min_sampled'], n_jobs=-1).fit(X)  # Apple
        labels = clustering.labels_

        # if self.debug:
        cluster_pcd = self.visualize_clusters(X=X, labels=labels, visualize=False)
        o3d.io.write_point_cloud(str(Path(self.pcd_path).parents[0] / 'semantic_cleaned_down_sampled_cluster.ply'),
                                 cluster_pcd)
        return X, C, labels

    def coarse_clustering(self, X, C, labels):
        unique, counts = np.unique(labels, return_counts=True)

        self.cluster_center = []
        self.assigned_cluster = []
        self.fuse_counter = 0
        self.counter = 0
        for i in unique:
            if i == -1:
                continue

            self.counter += 1

            # Find points with correct labels
            cluster_points_idx = np.argwhere(labels == i)

            cluster_points = X[cluster_points_idx.flatten()]
            current_apple_center = cluster_points.mean(axis=0)

            if self.cluster_center.__len__() > 0:

                # Compute distance between selected clusters and all other clusters
                dist = np.linalg.norm((np.vstack(self.cluster_center) - current_apple_center), axis=1)

                # If cluster distance is smaller than threshold value
                if dist[np.argmin(dist)] < self.cluster_merge_distance:
                    cluster_id = np.argmin(dist)
                    nearest_cluster = self.assigned_cluster[cluster_id]
                    center_1 = nearest_cluster.mean(axis=0)
                    center_2 = current_apple_center
                    fused_center = (center_1 + center_2) / 2
                    # apple = copy.deepcopy(self.apple_template)
                    # apple = apple.translate(c)
                    # o3d.visualization.draw_geometries([self.assigned_cluster[np.argmin(dist)], alpha_shape_pcd, apple], window_name="Near Apples")
                    self.fuse_counter += 1

                    self.cluster_center[cluster_id] = fused_center
                    self.assigned_cluster[cluster_id] = np.vstack([self.assigned_cluster[cluster_id], cluster_points])

                    continue

            self.cluster_center.append(current_apple_center)
            self.assigned_cluster.append(cluster_points)

        print('Coarse Counted apples: {}'.format(self.counter))
        print('Coarse Counted apples and fused clusters: {}'.format(self.counter - self.fuse_counter))

        X = []
        labels = []
        for idx, cluster in enumerate(self.assigned_cluster):
            X.append(cluster)
            labels.append(np.ones(cluster.shape[0], dtype=int) * (idx))

        # self.visualize_clusters(X = np.vstack(X), labels = np.hstack(labels))

        return X, labels

    def fine_clustering(self, X, C, labels):
        def one_apple_cluster(cluster):
            t_init = np.eye(4)
            t_init[:3, 3] = cluster.get_center()

            reg_p2p = o3d.pipelines.registration.registration_icp(
                self.apple_template, cluster, 0.01, t_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
            # draw_registration_result(self.apple_template, cluster, reg_p2p.transformation)

            apple_tmp = copy.deepcopy(self.apple_template)
            apple_tmp.transform(reg_p2p.transformation)

            cluster_apples = [apple_tmp]

            dist = hausdorff_distance(np.asarray(cluster.points), np.asarray(apple_tmp.points), distance="euclidean")

            return dist, cluster_apples

        class ClusterThread(Thread):
            # constructor
            def __init__(self, cluster, n_clusters, template):
                # execute the base constructor
                Thread.__init__(self)
                # set a default value
                self.value = None

                self.template = template
                self.n_clusters = n_clusters
                self.cluster = cluster

                self.dist = None
                self.found_clusters = None

            # function executed in a new thread
            def run(self):
                cluster_apples = []
                apples = o3d.geometry.PointCloud()
                sub_center = []
                clustering = AgglomerativeClustering(n_clusters=self.n_clusters, )
                l = clustering.fit_predict(np.asarray(self.cluster.points))

                for cluster_idx in np.unique(l):
                    new_pcd = o3d.geometry.PointCloud()
                    new_pcd.points = o3d.utility.Vector3dVector(np.asarray(self.cluster.points)[l == cluster_idx])
                    new_pcd.paint_uniform_color(plt.get_cmap("tab20")(cluster_idx)[:3])
                    # cluster_apples.append(new_pcd)
                    t = new_pcd.get_center()
                    sub_center.append(t)
                    apple = copy.deepcopy(self.template)
                    apple = apple.translate(t)
                    cluster_apples.append(apple)
                    apples = apples + apple
                dist = hausdorff_distance(np.asarray(self.cluster.points), np.asarray(apples.points),
                                          distance="euclidean")

                self.dist = dist
                self.found_clusters = cluster_apples

        class AlphaVolumeThread(Thread):
            # constructor
            def __init__(self, cluster):
                # execute the base constructor
                Thread.__init__(self)
                # set a default value
                self.cluster = cluster
                self.alpha_shape_volume = None

            def run(self):
                self.alpha_shape_volume = alphashape.alphashape(self.cluster, 10)

        class AlphaShapeThread(Thread):
            # constructor
            def __init__(self, cluster):
                # execute the base constructor
                Thread.__init__(self)
                # set a default value
                self.cluster = cluster
                self.alpha_shape = None

            def run(self):
                self.alpha_shape = alphashape.alphashape(self.cluster, 100)

        # def agglomerative_clustering(cluster, n_clusters):
        #    cluster_apples = []
        #    apples = o3d.geometry.PointCloud()
        #    sub_center = []
        #    clustering = AgglomerativeClustering(n_clusters=n_clusters, )
        #    l = clustering.fit_predict(np.asarray(cluster.points))
        #
        #    for cluster_idx in np.unique(l):
        #        new_pcd = o3d.geometry.PointCloud()
        #        new_pcd.points = o3d.utility.Vector3dVector(np.asarray(cluster.points)[l == cluster_idx])
        #        new_pcd.paint_uniform_color(plt.get_cmap("tab20")(cluster_idx)[:3])
        #        # cluster_apples.append(new_pcd)
        #        t = new_pcd.get_center()
        #        sub_center.append(t)
        #        apple = copy.deepcopy(self.apple_template)
        #        apple = apple.translate(t)
        #        cluster_apples.append(apple)
        #        apples = apples + apple

        ## cluster_apples.append(apples)
        ## dist = self.chamferDist(torch.from_numpy(np.asarray(cluster.points))[None, ...].to(torch.float32),
        ##                        torch.from_numpy(np.asarray(apples.points))[None, ...].to(torch.float32))

        # dist = hausdorff_distance(np.asarray(cluster.points), np.asarray(apples.points), distance="euclidean")

        # return dist, cluster_apples

        additional_count = 0
        prune_counter = 0

        valid_clusters = []

        for cluster in tqdm(X, desc='Large Fruit Clusters processed'):
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster)

            t1 = AlphaVolumeThread(cluster=cluster)
            t2 = AlphaShapeThread(cluster=cluster)

            t1.start()
            t2.start()

            t1.join()
            t2.join()

            alpha_shape_volume = t1.alpha_shape_volume
            alpha_shape = t2.alpha_shape
            alpha_shape_pcd = alpha_shape.as_open3d.sample_points_uniformly(1000)

            # alpha_shape_pcd = alpha_shape_pcd.translate(-alpha_shape_pcd.get_center())
            alpha_shape_pcd.paint_uniform_color([1, 0, 0])

            centered_cluster = cluster - cluster.mean(axis=0)

            if self.apple_alpha_shape_.volume < 0.9 * alpha_shape_volume.volume:
                if False:
                    o3d.visualization.draw_geometries([alpha_shape_pcd], window_name="Multiple Cluster")

                    alpha_shape_pcd_tmp = copy.deepcopy(alpha_shape_pcd)
                    alpha_shape_pcd_tmp = alpha_shape_pcd_tmp.translate(-alpha_shape_pcd_tmp.get_center())
                    o3d.visualization.draw_geometries([alpha_shape_pcd_tmp, self.apple_template],
                                                      window_name="Multiple Cluster")

                # Compute distances for all clusters
                dist_1, cluster_apples_1 = one_apple_cluster(cluster=alpha_shape_pcd)
                # dist_2, cluster_apples_2 = agglomerative_clustering(cluster=alpha_shape_pcd, n_clusters=2)
                # dist_3, cluster_apples_3 = agglomerative_clustering(cluster=alpha_shape_pcd, n_clusters=3)
                # dist_4, cluster_apples_4 = agglomerative_clustering(cluster=alpha_shape_pcd, n_clusters=4)
                # dist_5, cluster_apples_5 = agglomerative_clustering(cluster=alpha_shape_pcd, n_clusters=5)
                # dist_6, cluster_apples_6 = agglomerative_clustering(cluster=alpha_shape_pcd, n_clusters=6)

                t2 = ClusterThread(cluster=alpha_shape_pcd, n_clusters=2, template=self.apple_template)
                t3 = ClusterThread(cluster=alpha_shape_pcd, n_clusters=3, template=self.apple_template)
                t4 = ClusterThread(cluster=alpha_shape_pcd, n_clusters=4, template=self.apple_template)
                t5 = ClusterThread(cluster=alpha_shape_pcd, n_clusters=5, template=self.apple_template)
                t6 = ClusterThread(cluster=alpha_shape_pcd, n_clusters=6, template=self.apple_template)

                t2.start()
                t3.start()
                t4.start()
                t5.start()
                t6.start()

                t2.join()
                t3.join()
                t4.join()
                t5.join()
                t6.join()

                dist_2, cluster_apples_2 = t2.dist, t2.found_clusters
                dist_3, cluster_apples_3 = t3.dist, t3.found_clusters
                dist_4, cluster_apples_4 = t4.dist, t4.found_clusters
                dist_5, cluster_apples_5 = t5.dist, t5.found_clusters
                dist_6, cluster_apples_6 = t6.dist, t6.found_clusters

                # Min distances
                apple_in_cluster = np.argmin([dist_1, dist_2, dist_3, dist_4, dist_5, dist_6]) + 1

                cluster_apples = [cluster_apples_1, cluster_apples_2,
                                  cluster_apples_3, cluster_apples_4,
                                  cluster_apples_5, cluster_apples_6]

                additional_count += (apple_in_cluster - 1)  # One apple is already counted

                print("Number of apples in cluster: ", apple_in_cluster)

                # valid_clusters.append(cluster_apples[apple_in_cluster - 1])
                valid_clusters = valid_clusters + cluster_apples[apple_in_cluster - 1]

                continue

            elif self.minimum_size_factor * self.apple_alpha_shape_.volume > alpha_shape_volume.volume:
                if False:
                    # o3d.visualization.draw_geometries([alpha_shape_pcd, self.apple_template], window_name="Tiny Cluster")
                    alpha_shape_pcd_tmp = copy.deepcopy(alpha_shape.as_open3d)
                    alpha_shape_pcd_tmp = alpha_shape_pcd_tmp.translate(-alpha_shape_pcd_tmp.get_center())
                    o3d.visualization.draw_geometries([alpha_shape_pcd_tmp, self.apple_template],
                                                      window_name="Tiny Cluster")

                prune_counter += 1  # One apple is already counted
                print("Prune apple")

                continue

            valid_clusters.append(cluster)

        # Get cluster visualization
        cluster_centers = []
        cluster_single = []
        pcd = o3d.geometry.PointCloud()
        for idx, c in enumerate(valid_clusters):
            if isinstance(c, np.ndarray):
                pcd_new = o3d.geometry.PointCloud()
                pcd_new.points = o3d.utility.Vector3dVector(c)
                pcd_new.paint_uniform_color((plt.get_cmap("tab20")(idx % 19))[:3])
                cluster_centers.append(pcd_new.get_center())
                cluster_single.append(pcd_new)
                pcd = pcd + pcd_new
            elif isinstance(c, o3d.geometry.PointCloud):
                c.paint_uniform_color((0, 0, 0))
                c.paint_uniform_color((plt.get_cmap("tab20")((idx + np.random.randint(1, 100, 1)[0]) % 19))[:3])
                cluster_centers.append(c.get_center())
                cluster_single.append(c)

                pcd = pcd + c

            else:
                for p in c:
                    p.paint_uniform_color((0, 0, 0))
                    p.paint_uniform_color((plt.get_cmap("tab20")((idx + np.random.randint(1, 100, 1)[0]) % 19))[:3])
                    cluster_centers.append(p.get_center())
                    cluster_single.append(p)
                    pcd = pcd + p

        if not isinstance(self.gt_cluster, type(None)):
            if "obj" in self.gt_cluster:
                self.gt_cluster_center = np.asarray(self.gt_position.points)
            else:
                line_points = np.asarray(self.gt_position.points)
                line_boxes = line_points.reshape((line_points.shape[0] // 8, 8, 3))
                self.gt_cluster_center = line_boxes.mean(axis=1)

            if self.gt_cluster:
                self.correct_clusters = o3d.geometry.PointCloud()
                real_count = 0
                for cluster, cluster_pcd in zip(cluster_centers, cluster_single):
                    distance = np.linalg.norm(self.gt_cluster_center - cluster, axis=1)
                    minimum = np.argmin(distance)

                    if distance[minimum] < 0.25:
                        real_count += 1
                        self.gt_cluster_center = np.delete(self.gt_cluster_center, minimum, axis=0)
                        self.correct_clusters = self.correct_clusters + cluster_pcd
                    else:
                        # pcd_new = o3d.geometry.PointCloud()
                        # pcd_new.points = o3d.utility.Vector3dVector(self.gt_cluster_center)
                        # pcd_new.paint_uniform_color((0, 0, 0))
                        cluster_pcd.paint_uniform_color((0, 0, 0))
                        self.correct_clusters = self.correct_clusters + cluster_pcd

                        print("Distance: ", distance[minimum])
                        # o3d.visualization.draw_geometries([cluster_pcd, pcd_new, self.gt_mesh])

        o3d.io.write_point_cloud(str(Path(self.pcd_path).parents[0] / 'estimated_clusters.ply'), pcd)

        count = self.counter - self.fuse_counter + additional_count - prune_counter
        print('Counted apples, fused and split: {}'.format(count))
        if not isinstance(self.gt_cluster, type(None)):
            o3d.io.write_point_cloud(str(Path(self.pcd_path).parents[0] / 'clusters_gt_and_estimated.ply'),
                                     self.correct_clusters)

            print('Correclty assigned apples: {}'.format(real_count))
            self.real_count = real_count
            recall = self.real_count / self.gt_count
            precision = self.real_count / count
            print('Recall: {}'.format(recall))
            print('Precision: {}'.format(precision))
            print('F1: {}'.format(2 * recall * precision / (recall + precision)))

        print('Detection Rate: {}'.format(count / self.gt_count))

        if not isinstance(self.gt_cluster, type(None)):
            count = {
                'count': count,
                'detection_rate': count / self.gt_count,
                'Recall': recall,
                'Precision': precision,
                'F1': 2 * recall * precision / (recall + precision)
            }

        return count

    def count(self, pcd: Union[str], eps: float = 0.01) -> int:
        import os
        if isinstance(pcd, str):
            if not os.path.exists(pcd):
                print("Semantic Point Cloud does not exist!!")
                self.real_count = 0
                count = 0
                return count
            self.pcd_path = pcd
            pcd = o3d.io.read_point_cloud(pcd)

        if np.asarray(pcd.points).shape[0] != 0:
            X, C, labels = self.cluster(pcd=pcd, eps=eps, min_sampled=self.min_samples)

            if isinstance(X, int) and X == -1:
                count = 0
                return count

            X, labels = self.coarse_clustering(X, C, labels)
            count = self.fine_clustering(X, C, labels)
        else:
            count = 0
            self.real_count = 0
            print('No points in Point Cloud!!!')

        return count


if __name__ == '__main__':
    Baum_01_SAM = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/Baum_02/ns_SAM/fruit_nerf/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.015,
        "down_sample": 0.001,
        "eps": 0.02,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1,
        'gt_cluster': None,
        'gt_count': 179,
    }

    Baum_01_SAM_Big = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/Baum_02/ns_SAM/fruit_nerf_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.015,
        "down_sample": 0.001,
        "eps": 0.02,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1,
        'gt_cluster': None,
        'gt_count': 179,
    }

    Baum_01_unet = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/Baum_02/ns_unet/fruit_nerf/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.015,
        "down_sample": 0.001,
        "eps": 0.02,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1,
        'gt_cluster': None,
        'gt_count': 179,
    }

    Baum_01_unet_Big = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/Baum_02/ns_unet/fruit_nerf_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.015,
        "down_sample": 0.001,
        "eps": 0.02,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1,
        'gt_cluster': None,
        'gt_count': 179,
    }

    Baum_02_SAM = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/Baum_03/ns_SAM/fruit_nerf/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 65,
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": 0.03,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.1,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1.9,
        'gt_cluster': None,
        'gt_count': 113,
    }

    Baum_02_SAM_Big = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/Baum_03/ns_SAM/fruit_nerf_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 70,
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": 0.03,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.1,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1.9,
        'gt_cluster': None,
        'gt_count': 113,
    }

    Baum_02_unet = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/Baum_03/ns_unet/fruit_nerf/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 50,
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": 0.03,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.1,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1.9,
        'gt_cluster': None,
        'gt_count': 113,
    }

    Baum_02_unet_Big = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/Baum_03/ns_unet/fruit_nerf_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 50,
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": 0.03,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.1,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1.9,
        'gt_cluster': None,
        'gt_count': 113,
    }

    Baum_03_SAM = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/Baum_01/ns_SAM/fruit_nerf/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": 0.025,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1.8,
        'gt_cluster': None,
        'gt_count': 291,
    }

    Baum_03_SAM_Big = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/Baum_01/ns_SAM/fruit_nerf_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": 0.025,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1.8,
        'gt_cluster': None,
        'gt_count': 291,
    }

    Baum_03_unet = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/Baum_01/ns_SAM/fruit_nerf/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": 0.025,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1.8,
        'gt_cluster': None,
        'gt_count': 291,
    }

    Baum_03_unet_Big = {
        "path": "/home/se86kimy/Dropbox/07_data/For5G/Apple_24_08_23/Baum_01/ns_unet/fruit_nerf_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": 0.025,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1.8,
        'gt_cluster': None,
        'gt_count': 291,
    }

    Fuji_sam = {
        "path": "/home/se86kimy/Dropbox/07_data/Fuji-SfM_dataset/2-SfM-set/fused/fruit_nerf/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": 0.025,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.2,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1.1,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/Fuji-SfM_dataset/2-SfM-set/lineset_aligned.ply",
        "gt_count": 1455

    }

    Fuji_sam_big = {
        "path": "/home/se86kimy/Dropbox/07_data/Fuji-SfM_dataset/2-SfM-set/fused/fruit_nerf_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": 0.025,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.2,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1.1,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/Fuji-SfM_dataset/2-SfM-set/lineset_aligned.ply",
        "gt_count": 1455
    }

    Fuji_unet = {
        "path": "/home/se86kimy/Dropbox/07_data/Fuji-SfM_dataset/2-SfM-set/fused_unet/fruit_nerf/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 100,
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": 0.025,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.2,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1.1,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/Fuji-SfM_dataset/2-SfM-set/lineset_aligned.ply",
        "gt_count": 1455
    }

    Fuji_unet_big = {
        "path": "/home/se86kimy/Dropbox/07_data/Fuji-SfM_dataset/2-SfM-set/fused_unet/fruit_nerf_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": 75,
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": 0.025,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.2,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 1.1,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/Fuji-SfM_dataset/2-SfM-set/lineset_aligned.ply",
        "gt_count": 1455
    }

    Apple_SAM_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/01_apple_tree_1024x1024_#300/fruit_nerf/semantic_colormap.ply",
        "remove_outliers_nb_points": 200,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 0.7,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/01_apple_tree_1024x1024_#300/fruits.obj",
        "gt_count": 283
    }

    Apple_GT_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/01_apple_tree_1024x1024_#300/fruit_nerf/semantic_colormap.ply",
        "remove_outliers_nb_points": 200,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 0.7,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/01_apple_tree_1024x1024_#300/fruits.obj",
        "gt_count": 283
    }

    Plum_GT_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/03_plum_tree_1024x1024_#300/fruit_nerf/semantic_colormap.ply",
        "remove_outliers_nb_points": 250,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "cluster_merge_distance": 0.03,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 0.35,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/03_plum_tree_1024x1024_#300/fruits.obj",
        "gt_count": 745
    }

    Plum_SAM_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/03_plum_tree_1024x1024_#300_SAM/fruit_nerf/semantic_colormap.ply",
        "remove_outliers_nb_points": 250,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "cluster_merge_distance": 0.03,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 0.35,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/03_plum_tree_1024x1024_#300/fruits.obj",
        "gt_count": 745
    }

    Lemon_GT_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/05_lemon_tree_1024x1024_#300/fruit_nerf/semantic_colormap.ply",
        "remove_outliers_nb_points": 200,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        "cluster_merge_distance": 0.06,
        "template_path": '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/lemon_template.ply',
        'apple_template_size': 1.1,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/05_lemon_tree_1024x1024_#300/fruits.obj",
        "gt_count": 326
    }

    Lemon_SAM_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/05_lemon_tree_1024x1024_#300_SAM/fruit_nerf/semantic_colormap.ply",
        "remove_outliers_nb_points": 200,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        "cluster_merge_distance": 0.06,
        "template_path": '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/lemon_template.ply',
        'apple_template_size': 1.1,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/05_lemon_tree_1024x1024_#300/fruits.obj",
        "gt_count": 326
    }

    Pear_GT_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/02_pear_tree_1024x1024_#300/fruit_nerf/semantic_colormap.ply",
        "remove_outliers_nb_points": 250,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        "cluster_merge_distance": 0.03,
        "template_path": '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/pear_template.ply',
        'apple_template_size': 1.1,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/02_pear_tree_1024x1024_#300/fruits.obj",
        "gt_count": 250
    }

    Pear_SAM_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/02_pear_tree_1024x1024_#300_SAM/fruit_nerf/semantic_colormap.ply",
        "remove_outliers_nb_points": 250,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "minimum_size_factor": 0.3,
        "min_samples": 100,
        "cluster_merge_distance": 0.03,
        "template_path": '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/pear_template.ply',
        'apple_template_size': 1.1,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/02_pear_tree_1024x1024_#300/fruits.obj",
        "gt_count": 250
    }

    Peach_GT_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/07_peach_tree_1024x1024_#300/fruit_nerf/semantic_colormap.ply",
        "remove_outliers_nb_points": 250,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "minimum_size_factor": 0.3,
        "min_samples": 70,
        "cluster_merge_distance": 0.03,
        "template_path": '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/peach_template2.ply',
        'apple_template_size': 1.2,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/07_peach_tree_1024x1024_#300/fruits.obj",
        "gt_count": 152
    }

    Peach_SAM_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/07_peach_tree_1024x1024_#300_SAM/fruit_nerf/semantic_colormap.ply",
        "remove_outliers_nb_points": 250,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "minimum_size_factor": 0.3,
        "min_samples": 70,
        "cluster_merge_distance": 0.03,
        "template_path": '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/peach_template2.ply',
        'apple_template_size': 1.2,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/07_peach_tree_1024x1024_#300/fruits.obj",
        "gt_count": 152
    }

    Mango_GT_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/08_mango_tree_1024x1024_#300/fruit_nerf/semantic_colormap.ply",
        "remove_outliers_nb_points": 250,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "cluster_merge_distance": 0.01,
        "minimum_size_factor": 0.3,
        "min_samples": 70,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 0.3,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/08_mango_tree_1024x1024_#300/fruits.obj",
        "gt_count": 1150
    }

    Mango_SAM_1024x1024_300 = {
        "path": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/08_mango_tree_1024x1024_#300_SAM/fruit_nerf/semantic_colormap.ply",
        "remove_outliers_nb_points": 250,
        "remove_outliers_radius": 0.01,
        "down_sample": 0.001,
        "eps": 0.01,
        "cluster_merge_distance": 0.01,
        "minimum_size_factor": 0.3,
        "min_samples": 70,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': 0.3,
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/FruitNeRF/data/rendering/08_mango_tree_1024x1024_#300/fruits.obj",
        "gt_count": 1150
    }

    Baum = Fuji_unet_big



    Fuji_unet_big_sweep = {
        "path": "/home/se86kimy/Dropbox/07_data/Fuji-SfM_dataset/2-SfM-set/fused_unet/fruit_nerf_big/semantic_colormap_cropped.ply",
        "remove_outliers_nb_points": [50, 60, 75],
        "remove_outliers_radius": 0.025,
        "down_sample": 0.001,
        "eps": [0.015, 0.02],
        "cluster_merge_distance": 0.04,
        "minimum_size_factor": 0.2,
        "min_samples": 100,
        'template_path': '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/FruitNeRF/clustering/apple_template.ply',
        'apple_template_size': [0.9, 1, 1.1],
        "gt_cluster": "/home/se86kimy/Dropbox/07_data/Fuji-SfM_dataset/2-SfM-set/lineset_aligned.ply",
        "gt_count": 1455
    }

    Baum = Fuji_unet_big_sweep
    counter = {}

    i = 0

    for remove_outliers_nb_points in Baum["remove_outliers_nb_points"]:
        for eps in Baum["eps"]:
            for apple_template_size in Baum["apple_template_size"]:
                apple_clustering = FruitClustering(
                    remove_outliers_nb_points=remove_outliers_nb_points,
                    remove_outliers_radius=Baum['remove_outliers_radius'],
                    voxel_size_down_sample=Baum['down_sample'],
                    min_samples=Baum['min_samples'],
                    cluster_merge_distance=Baum['cluster_merge_distance'],
                    minimum_size_factor=Baum['minimum_size_factor'],
                    template_path=Baum['template_path'],
                    apple_template_size=apple_template_size,
                    gt_cluster=Baum['gt_cluster'],
                    gt_count=Baum['gt_count']
                )
                ret = apple_clustering.count(pcd=Baum["path"], eps=eps)

                counter.update({i: {'count': ret,
                                    'remove_outliers_nb_points': remove_outliers_nb_points,
                                    'apple_template_size': apple_template_size,
                                    'eps': eps,
                                    }})
                i += 1

                print(counter)

    print(counter)

