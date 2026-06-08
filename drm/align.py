# Open3D registration pipeline
# Based on the Open3D global registration tutorial:
# https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html
# ----------------------------

import copy
import drm
import open3d as o3d
import numpy as np
import time

def draw_registration_result(source, target, transformation = np.eye(4)):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([211/255, 79/255, 60/255])  # Red
    target_temp.paint_uniform_color([0, 64/255, 200/255])  # Blue
    source_temp.transform(transformation)
    return drm.visualise_open3d([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration_ransac(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
    )


def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=distance_threshold
    )
    return o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, option
    )


def refine_registration_icp(source, target, init_transform, voxel_size):
    distance_threshold = voxel_size * 0.4

    source_local = copy.deepcopy(source)
    target_local = copy.deepcopy(target)

    if not source_local.has_normals():
        source_local.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    if not target_local.has_normals():
        target_local.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))

    return o3d.pipelines.registration.registration_icp(
        source_local,
        target_local,
        distance_threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )


def register_pair(source, target, voxel_size=0.05, use_fast_global=False, run_icp_refinement=True):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    t0 = time.time()
    if use_fast_global:
        global_result = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        global_method = "fast_global_registration"
    else:
        global_result = execute_global_registration_ransac(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        global_method = "ransac_fpfh"
    global_time_sec = time.time() - t0

    refined_result = None
    refine_time_sec = 0.0
    final_transform = global_result.transformation
    final_fitness = float(global_result.fitness)
    final_rmse = float(global_result.inlier_rmse)

    if run_icp_refinement:
        t1 = time.time()
        refined_result = refine_registration_icp(source, target, global_result.transformation, voxel_size)
        refine_time_sec = time.time() - t1
        final_transform = refined_result.transformation
        final_fitness = float(refined_result.fitness)
        final_rmse = float(refined_result.inlier_rmse)

    return {
        "global_method": global_method,
        "global_result": global_result,
        "refined_result": refined_result,
        "global_time_sec": global_time_sec,
        "refine_time_sec": refine_time_sec,
        "final_transform": final_transform,
        "final_fitness": final_fitness,
        "final_rmse": final_rmse,
    }