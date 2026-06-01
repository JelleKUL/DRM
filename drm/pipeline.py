import sys
import os
import json
import numpy as np
import trimesh
import drm
from pathlib import Path

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import drm
import drm.detect
import drm.generate
def detect_objects(pointcloud, model):
    return pointcloud



def reconstruct_scene(scene_dir: Path, simple_lama=None) -> trimesh.Scene:
    """
    Run the full scene completion pipeline on a single scan folder and
    save the result as reconstructed_environment.glb.

    Parameters
    ----------
    scene_dir   : Path to the scan folder.
    simple_lama : Optional pre-loaded SimpleLama instance. If None, a new
                  instance is created. Pass from reconstruct_dataset to avoid
                  reloading model weights for every scene.

    Returns
    -------
    The exported trimesh.Scene, or None if processing failed.
    """
    from PIL import Image
    from pathlib import Path

    scene_dir   = Path(scene_dir)
    output_path = scene_dir / "reconstructed_environment.glb"

    SCAN_FILE       = "main.txt"
    PANO_FILE       = "pano.png"
    EMPTY_PANO_FILE = "pano_empty.png"
    BB_FILE         = "main_bb.json"
    VOXEL_SIZE      = 0.05

    print(f"\n{'='*60}")
    print(f"Processing: {scene_dir.name}")
    print(f"{'='*60}")

    # --- Data loading ---
    print("Loading data...")
    pcd, _       = drm.txt_pcd_to_open3d(scene_dir / SCAN_FILE)
    pcd          = pcd.voxel_down_sample(VOXEL_SIZE)
    trans_matrix = drm.read_transform_matrix(scene_dir / SCAN_FILE, apply_unity_conversion=True)

    translate_matrix        = np.eye(4)
    translate_matrix[:3, 3] = trans_matrix[:3, 3]

    boxes          = drm.trimesh_to_open3d(drm.detect.load_gt_json_boxes_as_mesh(scene_dir / BB_FILE))
    expanded_boxes = drm.expand_mesh(boxes, offset=0.1)

    pano_image       = np.array(Image.open(scene_dir / PANO_FILE).convert("RGB"))
    empty_pano_image = np.array(Image.open(scene_dir / EMPTY_PANO_FILE).convert("RGB"))

    # --- Bounding box cutout ---
    print("Cutting out bounding boxes...")
    _, remainder_pcd = drm.detect.split_pointcloud_by_boxes(pcd, expanded_boxes)

    # --- Plane detection ---
    print("Detecting planes...")
    detected_planes, plane_models, _ = drm.detect.detect_planes_iteratively(
        remainder_pcd,
        min_points=100,
        num_iterations=1000,
        distance_threshold=0.1,
    )

    # --- Plane boundary detection ---
    print("Creating plane meshes...")
    plane_meshes = drm.generate.create_plane_meshes(
        plane_models,
        pcd.get_axis_aligned_bounding_box(),
        max_extend_modifier=1,
    )

    filtered_meshes, filtered_pcds = drm.generate.filter_planes_by_points(
        plane_meshes,
        detected_planes,
        distance_threshold=0.05,
        min_points=1000,
    )

    if len(filtered_meshes) == 0:
        print(f"WARNING: No planes detected for {scene_dir.name}, skipping.")
        return None

    # --- Sample unoccupied points and combine ---
    print("Sampling unoccupied plane points...")
    unoccupied    = drm.generate.sample_unoccupied_plane_points(filtered_meshes, filtered_pcds, voxel_size=VOXEL_SIZE)
    combined_pcds = [p + u for p, u in zip(filtered_pcds, unoccupied)]

    # --- Project meshes onto pcd surface ---
    print("Projecting meshes onto surface...")
    projected_meshes = drm.generate.plane_pointclouds_to_meshes(
        combined_pcds,
        filtered_meshes,
        voxel_size=VOXEL_SIZE,
        scanner_center=translate_matrix[:3, 3],
    )

    # --- UV generation ---
    print("Generating UVs...")
    uv_meshes = drm.generate.assign_plane_uvs(
        projected_meshes,
        pcd.get_axis_aligned_bounding_box().get_center(),
    )

    # --- Occlusion mask ---
    print("Generating occlusion mask...")
    masked_pano           = drm.generate.generate_occlusion_mask_image(
        pano_image, empty_pano_image, threshold=3, blur_radius=5.0
    )
    masked_plane_textures = drm.generate.sample_pano_textures(
        uv_meshes, masked_pano, trans_matrix
    )

    # --- Inpainting ---
    print("Inpainting occluded regions...")
    inpainted_textures = drm.generate.inpaint_plane_textures(
        masked_plane_textures, dilation_px=20, simple_lama=simple_lama
    )

    # --- Apply textures and export ---
    print("Applying textures and exporting...")
    textured_meshes = drm.generate.apply_texture_to_planes(uv_meshes, inpainted_textures)
    scene           = drm.visualise_open3d(textured_meshes)
    scene.export(str(output_path))

    print(f"Saved: {output_path}")
    return scene