"""
Batch Occlusion Grid Generator
Finds all main.txt files in subfolders of a dataset directory,
computes the occupation and occlusion grids for each, and saves
both as Open3D VoxelGrid files (.ply) in the same subfolder.

Usage:
    python batch_occlusion_grids.py
    python batch_occlusion_grids.py --dataset ../../../datasets/V-Scan/data
    python batch_occlusion_grids.py --voxel-size 0.1 --grid-voxel-size 0.2
    python batch_occlusion_grids.py --per-object --gt-json gt.json
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
import drm.detect

DATASET_DIR      = Path("../../../datasets/V-Scan/data")
REFERENCE_NAME   = "main.txt"
VOXEL_SIZE       = 0.05
GRID_VOXEL_SIZE  = 0.2
GT_JSON_NAME     = "main_bb.json"            # default GT filename to look for per folder

OCCUPIED_FILENAME = "occupied_grid.ply"
OCCLUDED_FILENAME = "occluded_grid.ply"
PER_OBJECT_SUBDIR = "object_grids"      # subfolder written when --per-object is used


def save_voxelgrid(voxel_grid: o3d.geometry.VoxelGrid, path: Path):
    """Save an Open3D VoxelGrid as a PLY file via point cloud (voxel centers)."""
    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        print(f"    Warning: empty grid, skipping save to {path.name}")
        return

    voxel_size = voxel_grid.voxel_size
    origin     = np.array(voxel_grid.origin)

    centers = []
    colors  = []
    for v in voxels:
        idx = np.array(v.grid_index, dtype=np.float64)
        centers.append(origin + (idx + 0.5) * voxel_size)
        if hasattr(v, "color") and v.color is not None:
            colors.append(np.array(v.color))
        else:
            colors.append([0.5, 0.5, 0.5])

    pcd        = o3d.geometry.PointCloud()
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
    per_object: bool,
    gt_json_name: str,
    no_scene: bool = False,
) -> bool:
    ref_path      = folder / REFERENCE_NAME
    occupied_path = folder / OCCUPIED_FILENAME
    occluded_path = folder / OCCLUDED_FILENAME
    scene_exists  = occupied_path.exists() and occluded_path.exists()

    try:
        refPcd, _    = drm.txt_pcd_to_open3d(ref_path, apply_unity_conversion=True)
        refPcd       = refPcd.voxel_down_sample(voxel_size)
        ref_scan_pos = drm.read_transform_matrix(ref_path, apply_unity_conversion=True)[:3, 3]
        movedRefPcd  = copy.deepcopy(refPcd).translate(-ref_scan_pos)

        if no_scene:
            print(f"  [skip] {folder.name} — full-scene grid skipped (--no-scene)")
        elif skip_existing and scene_exists:
            print(f"  [skip] {folder.name} — scene grids already exist")
        else:
            occupied_grid, occluded_grid = drm.combine.build_occlusion_grid(
                movedRefPcd, [0, 0, 0], voxel_size=grid_voxel_size
            )
            save_voxelgrid(occupied_grid, occupied_path)
            save_voxelgrid(occluded_grid, occluded_path)

            n_occ  = len(occupied_grid.get_voxels())
            n_occl = len(occluded_grid.get_voxels())
            print(f"  [ok]   {folder.name} — {n_occ} occupied, {n_occl} occluded voxels")

    except Exception as e:
        print(f"  [fail] {folder.name} — scene grids: {e}")
        return False

    if not per_object:
        return True

    gt_path = folder / gt_json_name
    if not gt_path.exists():
        print(f"  [warn] {folder.name} — --per-object set but no GT JSON found at {gt_path.name}")
        return True

    obj_dir = folder / PER_OBJECT_SUBDIR
    obj_dir.mkdir(exist_ok=True)

    try:
        boxes = drm.detect.load_gt_boxes_raw(gt_path)
        for box in boxes:
            box["points"] = box["points"] - ref_scan_pos
        results = drm.combine.build_occlusion_grid_per_object(
            reference   = movedRefPcd,
            scanner_pos = [0,0,0],
            bounding_boxes = boxes,
            voxel_size  = grid_voxel_size,
        )

        saved = 0
        for obj_id, (occ_grid, occl_grid) in results.items():
            safe_id  = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(obj_id))
            occ_out  = obj_dir / f"{safe_id}_occupied.ply"
            occl_out = obj_dir / f"{safe_id}_occluded.ply"

            if skip_existing and occ_out.exists() and occl_out.exists():
                print(f"    [skip] object {obj_id}")
                continue

            save_voxelgrid(occ_grid,  occ_out)
            save_voxelgrid(occl_grid, occl_out)

            n_occ  = len(occ_grid.get_voxels())
            n_occl = len(occl_grid.get_voxels())
            print(f"    [obj]  {obj_id} — {n_occ} occupied, {n_occl} occluded voxels")
            saved += 1

        print(f"  [ok]   {folder.name} — {saved} object grids written to {PER_OBJECT_SUBDIR}/")

    except Exception as e:
        print(f"  [fail] {folder.name} — per-object grids: {e}")
        return False

    return True


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
    parser.add_argument("--per-object", action="store_true", default=False,
                        help="Also compute per-object occlusion grids using GT bounding boxes")
    parser.add_argument("--gt-json", type=str, default=GT_JSON_NAME,
                        help=f"GT JSON filename to look for in each scene folder (default: {GT_JSON_NAME})")
    parser.add_argument("--no-scene", action="store_true", default=False,
                        help="Skip the full-scene occlusion grid; useful with --per-object")

    args = parser.parse_args()

    if args.no_scene and not args.per_object:
        parser.error("--no-scene without --per-object would do nothing; add --per-object")

    dataset_dir = args.dataset.resolve()
    if not dataset_dir.is_dir():
        print(f"Error: dataset directory not found: {dataset_dir}")
        sys.exit(1)

    folders = sorted([f for f in dataset_dir.iterdir()
                      if f.is_dir() and (f / REFERENCE_NAME).exists()])

    if not folders:
        print(f"No subfolders with '{REFERENCE_NAME}' found in {dataset_dir}")
        sys.exit(1)

    print(f"Found {len(folders)} scenes in {dataset_dir}")
    if args.per_object:
        print(f"Per-object mode enabled — looking for '{args.gt_json}' in each folder")
    if args.no_scene:
        print("Full-scene grids will be skipped (--no-scene)")
    print()

    success = 0
    failed  = 0
    skipped = 0

    for folder in folders:
        occupied_path = folder / OCCUPIED_FILENAME
        occluded_path = folder / OCCLUDED_FILENAME

        if (args.skip_existing
                and not args.per_object
                and not args.no_scene
                and occupied_path.exists()
                and occluded_path.exists()):
            print(f"  [skip] {folder.name}")
            skipped += 1
            continue

        ok = process_folder(
            folder          = folder,
            voxel_size      = args.voxel_size,
            grid_voxel_size = args.grid_voxel_size,
            skip_existing   = args.skip_existing,
            per_object      = args.per_object,
            gt_json_name    = args.gt_json,
            no_scene        = args.no_scene,
        )
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\nDone — {success} processed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()