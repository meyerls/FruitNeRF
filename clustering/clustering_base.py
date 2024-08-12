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
from pathlib import Path
import alphashape
from hausdorff import hausdorff_distance
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from tqdm import tqdm
import warnings
import logging
from threading import Thread

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


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
                 remove_outliers_nb_points: int = 100,
                 remove_outliers_radius: float = 0.01,
                 cluster_merge_distance: float = 0.04):

        self.voxel_size_down_sample: float = voxel_size_down_sample
        self.remove_outliers_nb_points: int = remove_outliers_nb_points
        self.remove_outliers_radius: float = remove_outliers_radius

        self.pcd_path: Union[str, None] = None
        self.pcd: Union[o3d.geometry.PointCloud, None] = None
        self.pcd_down_sampled: Union[o3d.geometry.PointCloud, None] = None
        self.pcd_downsampled_cleaned: Union[o3d.geometry.PointCloud, None] = None

        self.debug = False

        self.fruit_template: Union[o3d.geometry.PointCloud, None] = None
        self.template_path: Union[str, Path, None] = None
        # self.fruit_alpha_shape_ = None

        self.cluster_merge_distance: float = cluster_merge_distance

    def voxel_down_sample(self, pcd):
        return pcd.voxel_down_sample(voxel_size=self.voxel_size_down_sample)

    def remove_outliers(self, pcd):
        return pcd.remove_radius_outlier(nb_points=self.remove_outliers_nb_points,
                                         radius=self.remove_outliers_radius)

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

    def merge_small_clusters(self, X, C, labels):
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
                dist = np.linalg.norm((np.vstack(self.cluster_center) - current_apple_center), axis=1)

                if dist[np.argmin(dist)] < self.cluster_merge_distance:
                    cluster_id = np.argmin(dist)
                    nearest_cluster = self.assigned_cluster[cluster_id]
                    center_1 = nearest_cluster.mean(axis=0)
                    center_2 = current_apple_center
                    fused_center = (center_1 + center_2) / 2

                    self.fuse_counter += 1

                    self.cluster_center[cluster_id] = fused_center
                    self.assigned_cluster[cluster_id] = np.vstack([self.assigned_cluster[cluster_id], cluster_points])

                    continue

            self.cluster_center.append(current_apple_center)
            self.assigned_cluster.append(cluster_points)

        print('First clustering stage count: {}'.format(self.counter))
        print('First clustering stage count after fused (tiny) clusters: {}'.format(self.counter - self.fuse_counter))

        X = []
        labels = []
        for idx, cluster in enumerate(self.assigned_cluster):
            X.append(cluster)
            labels.append(np.ones(cluster.shape[0], dtype=int) * (idx))

        # self.visualize_clusters(X = np.vstack(X), labels = np.hstack(labels))

        return X, labels

    def split_large_cluster(self, X, C, labels):
        def one_apple_cluster(cluster):
            t_init = np.eye(4)
            t_init[:3, 3] = cluster.get_center()

            reg_p2p = o3d.pipelines.registration.registration_icp(
                self.fruit_template, cluster, 0.01, t_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
            # draw_registration_result(self.fruit_template, cluster, reg_p2p.transformation)

            apple_tmp = copy.deepcopy(self.fruit_template)
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

            if self.fruit_alpha_shape_.volume < 0.9 * alpha_shape_volume.volume:
                # o3d.visualization.draw_geometries([alpha_shape_pcd], window_name="Multiple Cluster")

                # alpha_shape_pcd_tmp = copy.deepcopy(alpha_shape_pcd)
                # alpha_shape_pcd_tmp = alpha_shape_pcd_tmp.translate(-alpha_shape_pcd_tmp.get_center())
                # o3d.visualization.draw_geometries([alpha_shape_pcd_tmp, self.fruit_template], window_name="Multiple Cluster")

                # Compute distances for all clusters
                dist_1, cluster_apples_1 = one_apple_cluster(cluster=alpha_shape_pcd)

                t2 = ClusterThread(cluster=alpha_shape_pcd, n_clusters=2, template=self.fruit_template)
                t3 = ClusterThread(cluster=alpha_shape_pcd, n_clusters=3, template=self.fruit_template)
                t4 = ClusterThread(cluster=alpha_shape_pcd, n_clusters=4, template=self.fruit_template)
                t5 = ClusterThread(cluster=alpha_shape_pcd, n_clusters=5, template=self.fruit_template)
                t6 = ClusterThread(cluster=alpha_shape_pcd, n_clusters=6, template=self.fruit_template)

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

                # print("Number of apples in cluster: ", apple_in_cluster)

                # valid_clusters.append(cluster_apples[apple_in_cluster - 1])
                valid_clusters = valid_clusters + cluster_apples[apple_in_cluster - 1]

                continue

            elif 0.3 * self.fruit_alpha_shape_.volume > np.abs(alpha_shape_volume.volume):
                # o3d.visualization.draw_geometries([alpha_shape_pcd, self.fruit_template], window_name="Tiny Cluster")
                prune_counter += 1  # One apple is already counted
                # print("Prune apple")

                continue

            valid_clusters.append(cluster)

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
                cluster_centers.append(c.get_center())
                cluster_single.append(c)

                pcd = pcd + c

            else:
                for p in c:
                    p.paint_uniform_color((0, 0, 0))
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
                gt_center_copy = self.gt_cluster_center.copy()

                self.false_positive = 0
                number_of_correctly_counted_objects = 0
                for cluster, cluster_pcd in zip(cluster_centers, cluster_single):
                    distance = np.linalg.norm(np.asarray(gt_center_copy) - cluster, axis=1)
                    nearest_center_id = np.argmin(distance)

                    if distance[nearest_center_id] < 0.15:  # 0.15
                        number_of_correctly_counted_objects += 1
                        gt_center_copy = np.delete(gt_center_copy, nearest_center_id, axis=0)
                    else:
                        pcd_new = o3d.geometry.PointCloud()
                        pcd_new.points = o3d.utility.Vector3dVector(self.gt_cluster_center)
                        pcd_new.paint_uniform_color((0, 0, 0))
                        print("Distance: ", distance[nearest_center_id])
                        self.false_positive += 1
                        # o3d.visualization.draw_geometries([cluster_pcd, pcd_new, self.gt_mesh])

        # o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(str(Path(self.pcd_path).parents[0] / 'estimated_clusters.ply'),
                                 pcd)
        count = self.counter - self.fuse_counter + additional_count - prune_counter
        print('Second stage clustering count: {}'.format(count))

        if self.gt_cluster:
            print('Correclty assigned fruits: {}'.format(number_of_correctly_counted_objects))
            self.real_count = number_of_correctly_counted_objects

            number_of_gt_objects = self.gt_count  # self.gt_cluster_center.__len__()

            self.false_negative = gt_center_copy.shape[0]
            self.true_positive = number_of_correctly_counted_objects

            self.precision = self.true_positive / (self.true_positive + self.false_positive)
            self.recall = self.true_positive / (self.true_positive + self.false_negative)
            self.F1 = 2 * self.precision * self.recall / (self.precision + self.recall)

            print("Counting result: {}/{}".format(number_of_correctly_counted_objects, number_of_gt_objects))
            print("Precision: {}".format(self.precision))
            print("Recall: {}".format(self.recall))
            print("F1: {}".format(self.F1))

        self.detection_rate = count / self.gt_count
        print("Detection rate: {}".format(self.detection_rate))

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

            X, labels = self.merge_small_clusters(X, C, labels)
            count = self.split_large_cluster(X, C, labels)
        else:
            count = 0
            self.real_count = 0
            print('No points in Point Cloud!!!')

        return count
