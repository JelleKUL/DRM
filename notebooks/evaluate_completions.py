#!/usr/bin/env python3
"""
evaluate_completions.py

Quantitative evaluation of object completions against V-Scan ground truth,
over the `selections` folder structure:

    <selections_root>/<category>/<scene>/<object>/
        gt.ply          - ground truth mesh
        scan.ply        - partial scan (real-world scale/position)
        completion.glb  - generated completion mesh

Metrics computed per object:
    - Chamfer distance (L1 and L2 variants)
    - F-score @ 5cm (also reports precision/recall at that threshold)
    - Volumetric IoU @ 64^3

IMPORTANT - coordinate alignment
---------------------------------
completion.glb is the raw output of a generative pipeline that normalizes
its input via something equivalent to:

    partial_pcd = inference.normalize_mesh(pcd, scale=0.98)

i.e. centered at the partial scan's bounding-box center and rescaled so the
largest bbox dimension maps to length 2*scale. The exported completion mesh
is still in that normalized space - NOT in the real-world scale/position
that gt.ply and scan.ply live in.

This script recovers that transform from scan.ply (which IS in real-world
scale, since it's built directly from the raw scanner points) and inverts
it on completion.glb, then runs a rigid ICP refinement against gt.ply to
clean up any small residual misalignment, before computing any metrics.

ASSUMPTION TO VERIFY: the exact center/scale formula used by your
`normalize_mesh` is reconstructed here as "bbox-center + fit-largest-extent-
to-2*scale", which matches common practice in 3D generation pipelines
(e.g. Hunyuan3D-style) but isn't guaranteed to match your implementation
exactly. Use --save-aligned-dir on a few objects and eyeball them in
Blender against gt.ply to confirm before trusting the full run. If your
normalize_mesh uses a different formula (e.g. centers at the centroid
instead of bbox center, or uses a bounding sphere radius instead of bbox
extent), the rough alignment will be off and ICP will have to do more of
the work - which will show up as a low icp_fitness / high icp_inlier_rmse
in the output CSV for those objects.

PERFORMANCE NOTE: volumetric IoU uses ray-casting (trimesh's `mesh.contains`)
over a resolution^3 grid, which is the slow part of this script - expect
roughly tens of seconds per mesh at the default 64^3 resolution without
extra dependencies. Queries are chunked (see --iou-chunk-size) to keep
memory bounded regardless of grid size; installing `pyembree`
(`pip install pyembree`) lets trimesh use a much faster ray-casting
backend automatically if available.

USAGE
-----
    python evaluate_completions.py \\
        --selections-root /home/jelle-vermandere/Documents/Github/data/selections \\
        --output-dir /home/jelle-vermandere/Documents/Github/data/selections/_eval

    # sanity-check alignment on a handful of objects before a full run:
    python evaluate_completions.py --save-aligned-dir /tmp/aligned_check --limit 5
"""

import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_object_triplets(selections_root):
    """
    Walk <selections_root>/<category>/<scene>/<object>/ and return a list of
    dicts for every folder containing all three required files.
    """
    triplets = []
    if not os.path.isdir(selections_root):
        raise FileNotFoundError(f"selections root not found: {selections_root}")

    for category in sorted(os.listdir(selections_root)):
        cat_dir = os.path.join(selections_root, category)
        if not os.path.isdir(cat_dir):
            continue

        for scene in sorted(os.listdir(cat_dir)):
            scene_dir = os.path.join(cat_dir, scene)
            if not os.path.isdir(scene_dir):
                continue

            for obj in sorted(os.listdir(scene_dir)):
                obj_dir = os.path.join(scene_dir, obj)
                if not os.path.isdir(obj_dir):
                    continue

                gt_path = os.path.join(obj_dir, "gt.ply")
                scan_path = os.path.join(obj_dir, "scan.ply")
                completion_path = os.path.join(obj_dir, "completion.glb")

                missing = [
                    p for p in (gt_path, scan_path, completion_path)
                    if not os.path.isfile(p)
                ]
                if missing:
                    print(f"  [skip] {category}/{scene}/{obj}: missing {[os.path.basename(m) for m in missing]}")
                    continue

                triplets.append({
                    "category": category,
                    "scene": scene,
                    "object": obj,
                    "gt_path": gt_path,
                    "scan_path": scan_path,
                    "completion_path": completion_path,
                })

    return triplets


def load_mesh_any(path):
    """Load a path as a single trimesh.Trimesh, regardless of whether the
    underlying file is a single mesh or a multi-geometry scene (glb)."""
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    return loaded


def load_points_any(path):
    """Load a path's vertex/point positions as an (N, 3) array, whether it's
    stored as a point cloud or a mesh."""
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if hasattr(loaded, "vertices"):
        return np.asarray(loaded.vertices)
    raise ValueError(f"Could not extract points from {path} (type {type(loaded)})")


# ---------------------------------------------------------------------------
# Alignment: denormalize (recover real-world scale/position) + ICP refine
# ---------------------------------------------------------------------------

def estimate_normalize_transform(points, scale=0.98):
    """
    Estimate the (center, scale_factor) used by a bbox-center / fit-to-cube
    style normalize_mesh(points, scale):

        normalized = (points - center) * scale_factor

    Inverse:

        original = normalized / scale_factor + center
    """
    bbmin = points.min(axis=0)
    bbmax = points.max(axis=0)
    center = (bbmin + bbmax) / 2.0
    extent = (bbmax - bbmin).max()
    if extent < 1e-12:
        raise ValueError("degenerate point set, zero extent")
    scale_factor = (2.0 * scale) / extent
    return center, scale_factor


def denormalize_mesh(mesh, center, scale_factor):
    out = mesh.copy()
    out.vertices = out.vertices / scale_factor + center
    return out


def icp_refine(source_mesh, target_mesh, n_points=5000, max_corr_dist=None):
    """Rigid point-to-point ICP refinement of source_mesh onto target_mesh."""
    src_pts, _ = trimesh.sample.sample_surface(source_mesh, n_points)
    tgt_pts, _ = trimesh.sample.sample_surface(target_mesh, n_points)

    if max_corr_dist is None:
        diag = np.linalg.norm(target_mesh.bounds[1] - target_mesh.bounds[0])
        max_corr_dist = 0.1 * diag

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_pts)
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pts)

    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, max_corr_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    aligned = source_mesh.copy()
    aligned.apply_transform(result.transformation)
    return aligned, result.fitness, result.inlier_rmse


def align_completion(completion_mesh, scan_points, gt_mesh, normalize_scale, icp_n_points):
    center, scale_factor = estimate_normalize_transform(scan_points, scale=normalize_scale)
    denormed = denormalize_mesh(completion_mesh, center, scale_factor)
    aligned, fitness, inlier_rmse = icp_refine(denormed, gt_mesh, n_points=icp_n_points)
    return aligned, {"icp_fitness": fitness, "icp_inlier_rmse": inlier_rmse}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def chamfer_and_fscore(pred_mesh, gt_mesh, n_points=10000, f_threshold=0.05):
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, n_points)

    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    dist_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
    dist_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)

    chamfer_l1 = 0.5 * (dist_pred_to_gt.mean() + dist_gt_to_pred.mean())
    chamfer_l2 = 0.5 * (np.square(dist_pred_to_gt).mean() + np.square(dist_gt_to_pred).mean())

    precision = float((dist_pred_to_gt < f_threshold).mean())
    recall = float((dist_gt_to_pred < f_threshold).mean())
    f_score = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "chamfer_l1": chamfer_l1,
        "chamfer_l2": chamfer_l2,
        "precision_5cm": precision,
        "recall_5cm": recall,
        "fscore_5cm": f_score,
    }


def ensure_watertight(mesh):
    m = mesh.copy()
    if not m.is_watertight:
        m.remove_unreferenced_vertices()
        m.fill_holes()
        m.fix_normals()
    return m


def chunked_contains(mesh, points, chunk_size=4096):
    """
    mesh.contains(points) in one shot can blow up memory for large point
    sets (trimesh's ray-triangle intersector isn't memory-bounded by
    default), even for fairly small/simple meshes. Querying in chunks
    keeps peak memory bounded regardless of total grid size.
    """
    results = np.empty(len(points), dtype=bool)
    for i in range(0, len(points), chunk_size):
        results[i:i + chunk_size] = mesh.contains(points[i:i + chunk_size])
    return results


def volumetric_iou(pred_mesh, gt_mesh, resolution=64, chunk_size=4096):
    pred_m = ensure_watertight(pred_mesh)
    gt_m = ensure_watertight(gt_mesh)

    bmin = np.minimum(pred_m.bounds[0], gt_m.bounds[0])
    bmax = np.maximum(pred_m.bounds[1], gt_m.bounds[1])
    margin = 0.02 * (bmax - bmin)
    bmin = bmin - margin
    bmax = bmax + margin

    lin = [np.linspace(bmin[i], bmax[i], resolution) for i in range(3)]
    grid = np.stack(np.meshgrid(*lin, indexing="ij"), axis=-1).reshape(-1, 3)

    pred_inside = chunked_contains(pred_m, grid, chunk_size=chunk_size)
    gt_inside = chunked_contains(gt_m, grid, chunk_size=chunk_size)

    intersection = np.logical_and(pred_inside, gt_inside).sum()
    union = np.logical_or(pred_inside, gt_inside).sum()

    iou = 0.0 if union == 0 else intersection / union
    return iou, pred_m.is_watertight, gt_m.is_watertight


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selections-root", default="/home/jelle-vermandere/Documents/Github/data/selections")
    parser.add_argument("--output-dir", default=None, help="defaults to <selections-root>/_eval")
    parser.add_argument("--normalize-scale", type=float, default=0.98,
                         help="the `scale` argument originally passed to normalize_mesh")
    parser.add_argument("--n-sample-points", type=int, default=10000,
                         help="points sampled per mesh for Chamfer/F-score")
    parser.add_argument("--icp-n-points", type=int, default=5000,
                         help="points sampled per mesh for ICP refinement")
    parser.add_argument("--fscore-threshold", type=float, default=0.05, help="meters (5cm default)")
    parser.add_argument("--iou-resolution", type=int, default=64)
    parser.add_argument("--iou-chunk-size", type=int, default=4096,
                         help="batch size for mesh.contains() queries, bounds peak memory "
                              "during volumetric IoU computation; lower if you hit OOM")
    parser.add_argument("--save-aligned-dir", default=None,
                         help="if set, exports the aligned completion mesh per object here, "
                              "for visually sanity-checking the alignment assumption in Blender")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N objects (for quick checks)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.selections_root, "_eval")
    os.makedirs(output_dir, exist_ok=True)
    if args.save_aligned_dir:
        os.makedirs(args.save_aligned_dir, exist_ok=True)

    print(f"Selections root : {args.selections_root}")
    print(f"Output dir      : {output_dir}")
    print(f"Normalize scale : {args.normalize_scale}  (must match what was used at generation time)")
    print(f"F-score thresh  : {args.fscore_threshold} m")
    print(f"IoU resolution  : {args.iou_resolution}^3")
    print()

    triplets = find_object_triplets(args.selections_root)
    if args.limit:
        triplets = triplets[: args.limit]
    print(f"\nFound {len(triplets)} object(s) with gt/scan/completion present\n")

    rows = []

    for t in triplets:
        tag = f"{t['category']}/{t['scene']}/{t['object']}"
        row = {"category": t["category"], "scene": t["scene"], "object": t["object"], "status": "ok"}

        try:
            gt_mesh = load_mesh_any(t["gt_path"])
            completion_mesh = load_mesh_any(t["completion_path"])
            scan_points = load_points_any(t["scan_path"])

            aligned_mesh, icp_info = align_completion(
                completion_mesh, scan_points, gt_mesh,
                normalize_scale=args.normalize_scale,
                icp_n_points=args.icp_n_points,
            )
            row.update(icp_info)

            if args.save_aligned_dir:
                safe_name = f"{t['category']}_{t['scene']}_{t['object']}".replace("/", "_")
                aligned_mesh.export(os.path.join(args.save_aligned_dir, f"{safe_name}_aligned.glb"))

            cf = chamfer_and_fscore(
                aligned_mesh, gt_mesh,
                n_points=args.n_sample_points, f_threshold=args.fscore_threshold,
            )
            row.update(cf)

            iou, pred_wt, gt_wt = volumetric_iou(
                aligned_mesh, gt_mesh,
                resolution=args.iou_resolution, chunk_size=args.iou_chunk_size,
            )
            row["volumetric_iou_64"] = iou
            row["pred_watertight"] = pred_wt
            row["gt_watertight"] = gt_wt

            print(f"  [ok] {tag}: chamfer_l1={cf['chamfer_l1']:.4f} "
                  f"fscore@5cm={cf['fscore_5cm']:.3f} iou={iou:.3f} "
                  f"(icp_fitness={icp_info['icp_fitness']:.3f})")

        except Exception as e:  # noqa: BLE001
            row["status"] = "error"
            row["error_message"] = f"{e}"
            print(f"  [ERROR] {tag}: {e}")
            traceback.print_exc()

        rows.append(row)

    df = pd.DataFrame(rows)
    per_object_path = os.path.join(output_dir, "metrics_per_object.csv")
    df.to_csv(per_object_path, index=False)
    print(f"\nPer-object results written to {per_object_path}")

    ok_df = df[df["status"] == "ok"]
    metric_cols = [
        "chamfer_l1", "chamfer_l2", "precision_5cm", "recall_5cm",
        "fscore_5cm", "volumetric_iou_64", "icp_fitness", "icp_inlier_rmse",
    ]
    metric_cols = [c for c in metric_cols if c in ok_df.columns]

    if len(ok_df):
        category_avg = ok_df.groupby("category")[metric_cols].mean().reset_index()
        category_avg["n_objects"] = ok_df.groupby("category").size().values
        category_avg_path = os.path.join(output_dir, "metrics_per_category_avg.csv")
        category_avg.to_csv(category_avg_path, index=False)
        print(f"Per-category averages written to {category_avg_path}\n")
        print(category_avg.to_string(index=False))
    else:
        print("No successful results to average.")

    n_ok = (df["status"] == "ok").sum()
    n_err = (df["status"] == "error").sum()
    print(f"\nDone. ok={n_ok} error={n_err} (of {len(df)} total)")


if __name__ == "__main__":
    main()
