"""
Batch Occlusion Grid Generator
Finds all main.txt files in subfolders of a dataset directory,
computes the occupation and occlusion grids for each, and saves
both as Open3D VoxelGrid files (.ply) in the same subfolder.

Usage:
    python batch_occlusion_grids.py
    python batch_occlusion_grids.py --dataset ../../../datasets/V-Scan/data
    python batch_occlusion_grids.py --voxel-size 0.1 --threshold 0.15
"""

import argparse
import copy
import sys
from pathlib import Path

import open3d as o3d
import numpy as np

sys.path.insert(0, '../')
import drm
import drm.combine

DATASET_DIR   = Path("../../../datasets/V-Scan/data")
REFERENCE_NAME = "main.txt"
VOXEL_SIZE    = 0.05
GRID_VOXEL_SIZE = 0.2
THRESHOLD_RESOLUTION = 0.1

OCCUPIED_FILENAME = "occupied_grid.ply"
OCCLUDED_FILENAME = "occluded_grid.ply"


def save_voxelgrid(voxel_grid: o3d.geometry.VoxelGrid, path: Path):
    """Save an Open3D VoxelGrid as a PLY file via point cloud (voxel centers)."""
    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        print(f"    Warning: empty grid, skipping save to {path.name}")
        return

    voxel_size = voxel_grid.voxel_size
    origin = np.array(voxel_grid.origin)

    centers = []
    colors = []
    for v in voxels:
        idx = np.array(v.grid_index, dtype=np.float64)
        centers.append(origin + (idx + 0.5) * voxel_size)
        if hasattr(v, 'color') and v.color is not None:
            colors.append(np.array(v.color))
        else:
            colors.append([0.5, 0.5, 0.5])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(centers))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    o3d.io.write_point_cloud(str(path), pcd)


def load_voxelgrid(path: Path, voxel_size: float) -> o3d.geometry.VoxelGrid:
    """Load a VoxelGrid saved by save_voxelgrid (PLY of voxel centers)."""
    pcd = o3d.io.read_point_cloud(str(path))
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)


def process_folder(
    folder: Path,
    voxel_size: float,
    grid_voxel_size: float,
    skip_existing: bool,
) -> bool:
    """
    Process a single scene folder:
    - Loads main.txt
    - Computes occupied + occluded grids
    - Saves both as PLY files

    Returns True on success, False on failure.
    """
    ref_path = folder / REFERENCE_NAME

    occupied_path = folder / OCCUPIED_FILENAME
    occluded_path = folder / OCCLUDED_FILENAME

    if skip_existing and occupied_path.exists() and occluded_path.exists():
        print(f"  [skip] {folder.name} — grids already exist")
        return True

    try:
        refPcd, _ = drm.txt_pcd_to_open3d(ref_path, apply_unity_conversion=True)
        refPcd = refPcd.voxel_down_sample(voxel_size)

        ref_scan_pos = drm.read_transform_matrix(ref_path, apply_unity_conversion=True)[:3, 3]
        movedRefPcd = copy.deepcopy(refPcd).translate(-ref_scan_pos)

        occupied_grid, occluded_grid = drm.combine.build_occlusion_grid(
            movedRefPcd, [0, 0, 0], voxel_size=grid_voxel_size
        )

        save_voxelgrid(occupied_grid, occupied_path)
        save_voxelgrid(occluded_grid, occluded_path)

        n_occ  = len(occupied_grid.get_voxels())
        n_occl = len(occluded_grid.get_voxels())
        print(f"  [ok]   {folder.name} — {n_occ} occupied, {n_occl} occluded voxels")
        return True

    except Exception as e:
        print(f"  [fail] {folder.name} — {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch compute occlusion/occupation grids.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR,
                        help="Root dataset folder containing scene subfolders")
    parser.add_argument("--voxel-size", type=float, default=VOXEL_SIZE,
                        help=f"Downsampling voxel size (default: {VOXEL_SIZE})")
    parser.add_argument("--grid-voxel-size", type=float, default=GRID_VOXEL_SIZE,
                        help=f"Grid voxel size for occlusion (default: {GRID_VOXEL_SIZE})")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip folders that already have both grid files (default: True)")
    parser.add_argument("--no-skip", action="store_false", dest="skip_existing",
                        help="Recompute even if grid files already exist")
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    if not dataset_dir.is_dir():
        print(f"Error: dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Find all subfolders that contain a main.txt
    folders = sorted([f for f in dataset_dir.iterdir()
                      if f.is_dir() and (f / REFERENCE_NAME).exists()])

    if not folders:
        print(f"No subfolders with '{REFERENCE_NAME}' found in {dataset_dir}")
        sys.exit(1)

    print(f"Found {len(folders)} scenes in {dataset_dir}\n")

    success = 0
    failed  = 0
    skipped = 0

    for folder in folders:
        occupied_path = folder / OCCUPIED_FILENAME
        occluded_path = folder / OCCLUDED_FILENAME

        if args.skip_existing and occupied_path.exists() and occluded_path.exists():
            print(f"  [skip] {folder.name}")
            skipped += 1
            continue

        ok = process_folder(folder, args.voxel_size, args.grid_voxel_size, args.skip_existing)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\nDone — {success} processed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
