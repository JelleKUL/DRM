import sys
import os
import json
import numpy as np
import trimesh
import drm

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def load_gt_boxes_raw(json_path: str) -> list[dict]:
    """
    Parse bounding boxes from JSON, returning raw corner points and IDs.
    Applies the same UNITY2TRIMESH_T transform used in load_gt_json_boxes_as_mesh.

    Returns list of {"id": str, "points": np.ndarray (8,3)}
    """
    with open(json_path) as f:
        data = json.load(f)

    boxes = []
    for box in data["boxes"]:
        pts = np.array([[p["x"], p["y"], p["z"]] for p in box["boundingPoints"]])
        # Apply the same coordinate transform used in the mesh loader
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_transformed = (drm.UNITY2TRIMESH_T @ pts_h.T).T[:, :3]
        boxes.append({"id": box["id"], "points": pts_transformed})

    return boxes

def load_gt_json_boxes_as_mesh(json_path, randomcolors = True):
    """
    Load detection results from a JSON file and convert them to a mesh format.

    Args:
        json_path: Path to the JSON file containing detection results
    """
    with open(json_path) as f:
        data = json.load(f)

    names = []
    points_list = []

    for box in data["boxes"]:
        names.append(box["id"])
        pts = np.array([[p["x"], p["y"], p["z"]] for p in box["boundingPoints"]])
        points_list.append(pts)

    # Colormap for labels
    label_colors = [
        [1, 0, 0, 0.5],  # red, alpha 0.5
        [0, 1, 0, 0.5],  # green
        [0, 0, 1, 0.5],  # blue
        [1, 1, 0, 0.5],  # yellow
        [1, 0, 1, 0.5],  # magenta
        [0, 1, 1, 0.5],  # cyan
    ]

    # Create list of meshes
    gt_bb_meshes = []
    i = 0
    for points in points_list:
        if(randomcolors):
            color = label_colors[i % len(label_colors)]
        else:
            color = [0, 0, 1, 0.5]  # blue, alpha 0.5

        mesh = trimesh.convex.convex_hull(points)
        mesh.visual.face_colors = color
        mesh.apply_transform(drm.UNITY2TRIMESH_T)
        gt_bb_meshes.append(mesh)
        i+=1

    return gt_bb_meshes

def load_detected_boxes_as_mesh(json_path, pred_score_thr=0.3, labelColors = True):
    """
    Load detection results from a JSON file and convert them to a mesh format.

    Args:
        json_path: Path to the JSON file containing detection results
        pred_score_thr: Minimum confidence score for detections to be included
        labelColors: Whether to use colors for different labels or a fixed color
    """
    with open(json_path) as f:
        data = json.load(f)

    # Colormap for labels
    label_colors = [
        [1, 0, 0, 0.5],  # red, alpha 0.5
        [0, 1, 0, 0.5],  # green
        [0, 0, 1, 0.5],  # blue
        [1, 1, 0, 0.5],  # yellow
        [1, 0, 1, 0.5],  # magenta
        [0, 1, 1, 0.5],  # cyan
    ]

    # Create list of meshes
    bb_meshes = []

    for label, score, box in zip(data["labels_3d"], data["scores_3d"], data["bboxes_3d"]):
        if score < pred_score_thr:
            continue

        center = np.array(box[:3])
        size = np.array(box[3:6])
        rotation_z = box[6]
        if(labelColors):
            color = label_colors[label % len(label_colors)]
        else:
            color = [1, 0, 0, 0.5]  # red, alpha 0.5
        
        mesh = drm.create_trimesh_box(center, size, rotation_z, color)
        bb_meshes.append(mesh)

    # Combine meshes for visualization
    return bb_meshes


def split_pointcloud_by_boxes(
    pcd: o3d.geometry.PointCloud,
    boxes: Union[o3d.geometry.TriangleMesh, o3d.geometry.OrientedBoundingBox, list],
) -> tuple[list[o3d.geometry.PointCloud], o3d.geometry.PointCloud]:
    if isinstance(boxes, (o3d.geometry.TriangleMesh, o3d.geometry.OrientedBoundingBox)):
        boxes = [boxes]

    points = np.asarray(pcd.points)
    has_colors  = pcd.has_colors()
    has_normals = pcd.has_normals()
    colors  = np.asarray(pcd.colors)  if has_colors  else None
    normals = np.asarray(pcd.normals) if has_normals else None

    claimed = np.zeros(len(points), dtype=bool)
    cropped_pcds: list[o3d.geometry.PointCloud] = []

    for box in boxes:
        # Accept a raw OBB/AABB or derive the OBB from a TriangleMesh
        if isinstance(box, o3d.geometry.TriangleMesh):
            obb = box.get_minimal_oriented_bounding_box()
        elif isinstance(box, (o3d.geometry.OrientedBoundingBox,
                               o3d.geometry.AxisAlignedBoundingBox)):
            obb = box
        else:
            raise TypeError(f"Unsupported geometry type: {type(box)}")

        cropped = pcd.crop(obb)
        cropped_pcds.append(cropped)

        # Re-use Open3D's own containment test to build the claimed mask
        inside_idx = np.asarray(obb.get_point_indices_within_bounding_box(pcd.points))
        claimed[inside_idx] = True

    remainder_idx = np.where(~claimed)[0]
    remainder_pcd = o3d.geometry.PointCloud()
    remainder_pcd.points = o3d.utility.Vector3dVector(points[remainder_idx])
    if has_colors:
        remainder_pcd.colors  = o3d.utility.Vector3dVector(colors[remainder_idx])
    if has_normals:
        remainder_pcd.normals = o3d.utility.Vector3dVector(normals[remainder_idx])

    return cropped_pcds, remainder_pcd


def detect_planes_iteratively(
    pcd: o3d.geometry.PointCloud,
    min_points: int = 100,
    num_iterations: int = 1000,
    distance_threshold: float = 0.01,
) -> tuple[list[o3d.geometry.PointCloud], list[np.ndarray], Optional[o3d.geometry.PointCloud]]:
    """
    Iteratively detect planes in a PointCloud using RANSAC, removing each
    detected plane before searching for the next.

    Parameters
    ----------
    pcd                : Input PointCloud.
    min_points         : Minimum inlier count to accept a plane.
    num_iterations     : RANSAC iterations per plane fit.
    distance_threshold : Max point-to-plane distance to count as an inlier.

    Returns
    -------
    planes       : List of PointClouds, one per detected plane, in detection order.
    plane_models : List of [a, b, c, d] arrays defining each plane as ax+by+cz+d=0.
    remainder_pc : PointCloud of leftover points, or None if none remain.
    """
    has_colors  = pcd.has_colors()
    has_normals = pcd.has_normals()

    remaining_points  = np.asarray(pcd.points).copy()
    remaining_colors  = np.asarray(pcd.colors).copy()  if has_colors  else None
    remaining_normals = np.asarray(pcd.normals).copy() if has_normals else None

    planes: list[o3d.geometry.PointCloud] = []
    plane_models: list[np.ndarray] = []

    while len(remaining_points) >= min_points:
        tmp = o3d.geometry.PointCloud()
        tmp.points = o3d.utility.Vector3dVector(remaining_points)

        plane_model, inliers = tmp.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=num_iterations,
        )

        if len(inliers) < min_points:
            print("No more large planes detected.")
            break

        plane_models.append(np.asarray(plane_model))  # [a, b, c, d]

        plane_pc = o3d.geometry.PointCloud()
        plane_pc.points = o3d.utility.Vector3dVector(remaining_points[inliers])
        if has_colors  and remaining_colors  is not None:
            plane_pc.colors  = o3d.utility.Vector3dVector(remaining_colors[inliers])
        if has_normals and remaining_normals is not None:
            plane_pc.normals = o3d.utility.Vector3dVector(remaining_normals[inliers])
        planes.append(plane_pc)

        mask = np.ones(len(remaining_points), dtype=bool)
        mask[inliers] = False
        remaining_points = remaining_points[mask]
        if remaining_colors  is not None: remaining_colors  = remaining_colors[mask]
        if remaining_normals is not None: remaining_normals = remaining_normals[mask]

    if len(remaining_points) > 0:
        remainder_pc = o3d.geometry.PointCloud()
        remainder_pc.points = o3d.utility.Vector3dVector(remaining_points)
        if remaining_colors  is not None:
            remainder_pc.colors  = o3d.utility.Vector3dVector(remaining_colors)
        if remaining_normals is not None:
            remainder_pc.normals = o3d.utility.Vector3dVector(remaining_normals)
    else:
        remainder_pc = None

    print(f"Extracted {len(planes)} planes.")
    return planes, plane_models, remainder_pc


def ransac_plane_trimesh(points, num_iterations=1000, distance_threshold=0.01):
    best_plane = None
    best_inliers = []

    for _ in range(num_iterations):
        # Randomly pick 3 points
        sample_indices = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[sample_indices]

        # Compute plane normal
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal /= norm

        # Plane equation: ax + by + cz + d = 0
        d = -np.dot(normal, p1)

        # Distances of all points to the plane
        distances = np.abs(points.dot(normal) + d)

        # Find inliers
        inliers = np.where(distances < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (*normal, d)

    return best_plane, best_inliers




"""
symmetry_detection.py
=====================
Reflective-symmetry axis detection for partial 3D point clouds.

Builds on your existing open3d / trimesh / numpy pipeline:
  - Input  : open3d.geometry.PointCloud  (or raw (N,3) numpy array)
  - Output : SymmetryResult dataclass  +  optional Open3D scene geometries

Algorithm (partial-cloud aware)
--------------------------------
1.  PCA-seeded candidates   – eigenvectors of the cloud covariance give the
    most likely symmetry-plane normals for roughly symmetric objects.
2.  Random-pair candidates  – midpoint / perpendicular bisector of random
    point pairs; robust to missing regions.
3.  KD-tree scoring         – reflect cloud across each candidate plane,
    count how many reflected points land within *threshold* of an original
    point.  Gives a 0-1 score that is partial-cloud safe (only one side
    needs to match the other).
4.  Local refinement        – small angular perturbations around the winner.

Usage
-----
    from symmetry_detection import detect_symmetry_o3d, symmetry_geometries

    pcd, _ = txt_pcd_to_open3d("scan.txt")
    result  = detect_symmetry_o3d(pcd)

    print(result)

    # Get Open3D geometries for your own visualiser
    geoms = symmetry_geometries(result, pcd)
    o3d.visualization.draw_geometries([pcd] + geoms)
"""



# ═══════════════════════════════════════════════════════════════════════════════
#  Result dataclasses
# ═══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class SymmetryResult:
    """Single symmetry plane result."""
 
    plane_normal: np.ndarray   # (3,) unit vector – direction of the symmetry axis
    plane_point:  np.ndarray   # (3,) a point lying on the symmetry plane
    score:        float        # fraction of reflected points within threshold [0, 1]
    threshold:    float        # distance threshold used (original cloud units)
 
    # Full candidate list, sorted best-first: list of (score, point, normal)
    candidates: List[Tuple[float, np.ndarray, np.ndarray]] = field(
        default_factory=list, repr=False
    )
 
    def __str__(self) -> str:
        return (
            f"SymmetryResult\n"
            f"  normal (axis) : {np.round(self.plane_normal, 4)}\n"
            f"  plane point   : {np.round(self.plane_point,  4)}\n"
            f"  score         : {self.score:.4f}  (1.0 = perfect)\n"
            f"  threshold     : {self.threshold:.4f}\n"
            f"  candidates    : {len(self.candidates)}"
        )
 
 
@dataclass
class MultiSymmetryResult:
    """
    Top-K diverse symmetry planes for one point cloud.
 
    Axes are sorted by score (best first) and guaranteed to be at least
    *min_angle_deg* apart from each other (NMS).
    """
    axes:          List[SymmetryResult]   # k entries, best-score first
    min_angle_deg: float                  # NMS angular separation used
    k_requested:   int                    # how many axes were requested
    threshold:     float
 
    @property
    def k(self) -> int:
        return len(self.axes)
 
    @property
    def normals(self) -> np.ndarray:
        """(K, 3) array of unit plane normals."""
        return np.stack([a.plane_normal for a in self.axes])
 
    @property
    def scores(self) -> np.ndarray:
        """(K,) array of symmetry scores."""
        return np.array([a.score for a in self.axes])
 
    def __str__(self) -> str:
        lines = [f"MultiSymmetryResult  k={self.k}/{self.k_requested}  "
                 f"min_angle={self.min_angle_deg}°"]
        for i, ax in enumerate(self.axes):
            lines.append(
                f"  [{i}] normal={np.round(ax.plane_normal, 3)}  score={ax.score:.4f}"
            )
        return "\n".join(lines)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  Core math helpers
# ═══════════════════════════════════════════════════════════════════════════════
 
def _reflect(points: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Reflect *points* across the plane defined by *plane_point* + *plane_normal*."""
    n    = plane_normal / np.linalg.norm(plane_normal)
    d    = points - plane_point
    dist = (d @ n)[:, None]
    return points - 2.0 * dist * n
 
 
def _score(
    points:      np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    threshold:   float,
    tree:        cKDTree,
) -> float:
    """Score a candidate plane: fraction of reflected points within *threshold*."""
    reflected = _reflect(points, plane_point, plane_normal)
    dists, _  = tree.query(reflected, workers=-1)
    return float(np.mean(dists < threshold))
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  Candidate generation
# ═══════════════════════════════════════════════════════════════════════════════
 
def _pca_candidates(pts_n: np.ndarray, centroid: np.ndarray) -> list:
    """Return 3 PCA eigenvector normals passing through the centroid."""
    cov     = np.cov((pts_n - centroid).T)
    _, evec = np.linalg.eigh(cov)      # columns = eigenvectors, ascending eigenvalue
    return [(centroid, evec[:, i]) for i in range(3)]
 
 
def _perturbed_pca_candidates(
    pts_n:    np.ndarray,
    centroid: np.ndarray,
    rng:      np.random.Generator,
    n_perturb: int = 30,
) -> list:
    """PCA normals with small angular noise – bridges between PCA and random."""
    cov     = np.cov((pts_n - centroid).T)
    _, evec = np.linalg.eigh(cov)
    cands   = []
    for i in range(3):
        base = evec[:, i]
        for sigma in [0.05, 0.15, 0.30]:
            for _ in range(n_perturb):
                perturbed  = base + rng.standard_normal(3) * sigma
                perturbed /= np.linalg.norm(perturbed)
                cands.append((centroid, perturbed))
    return cands
 
 
def _random_candidates(
    pts_n: np.ndarray,
    n:     int,
    rng:   np.random.Generator,
) -> list:
    """Perpendicular-bisector planes from random point pairs."""
    idx   = rng.integers(0, len(pts_n), size=(n, 2))
    cands = []
    for i, j in idx:
        diff = pts_n[j] - pts_n[i]
        norm = np.linalg.norm(diff)
        if norm < 1e-8:
            continue
        mid = (pts_n[i] + pts_n[j]) / 2.0
        cands.append((mid, diff / norm))
    return cands
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  Main detection function  (accepts raw numpy array)
# ═══════════════════════════════════════════════════════════════════════════════
 
def detect_symmetry(
    points:               np.ndarray,
    n_random_candidates:  int   = 500,
    threshold:            float = 0.02,
    refine_iterations:    int   = 6,
    refine_samples:       int   = 120,
    seed:                 int   = 42,
    verbose:              bool  = True,
) -> SymmetryResult:
    """
    Detect the best reflective symmetry plane of a (partial) 3-D point cloud.
 
    Parameters
    ----------
    points               : (N, 3) float array
    n_random_candidates  : number of random bisector planes to test
    threshold            : inlier distance as a fraction of the cloud's max extent
                           (e.g. 0.02 = 2 % of the bounding-sphere radius)
    refine_iterations    : gradient-free refinement rounds after the best candidate
    refine_samples       : angular perturbations per refinement round
    seed                 : RNG seed
    verbose              : print progress
 
    Returns
    -------
    SymmetryResult
    """
    pts  = np.asarray(points, dtype=np.float64)
    rng  = np.random.default_rng(seed)
 
    # --- Normalise to unit scale ------------------------------------------------
    centroid = pts.mean(axis=0)
    scale    = np.linalg.norm(pts - centroid, axis=1).max()
    if scale < 1e-10:
        raise ValueError("Degenerate point cloud: all points coincide.")
 
    pts_n   = (pts - centroid) / scale
    thr_n   = threshold                # already expressed as fraction of extent
    tree_n  = cKDTree(pts_n)
 
    # --- Build candidate pool ---------------------------------------------------
    origin  = np.zeros(3)
    cands   = _pca_candidates(pts_n, origin)
    cands  += _perturbed_pca_candidates(pts_n, origin, rng)
    cands  += _random_candidates(pts_n, n_random_candidates, rng)
 
    if verbose:
        print(f"[symmetry] {len(pts)} pts | {len(cands)} candidates | thr={threshold:.3f}")
 
    # --- Score all candidates ---------------------------------------------------
    scored = [
        (_score(pts_n, pp, pn, thr_n, tree_n), pp.copy(), pn.copy())
        for pp, pn in cands
    ]
    scored.sort(key=lambda x: -x[0])
    best_score, best_pt, best_n = scored[0]
 
    if verbose:
        print(f"[symmetry] Initial best score : {best_score:.4f}")
 
    # --- Local refinement -------------------------------------------------------
    for it in range(refine_iterations):
        sigma    = 0.08 / (it + 1)
        improved = False
        for _ in range(refine_samples):
            perturb = rng.standard_normal(3) * sigma
            n_try   = best_n + perturb
            n_try  /= np.linalg.norm(n_try)
            s = _score(pts_n, best_pt, n_try, thr_n, tree_n)
            if s > best_score:
                best_score, best_n = s, n_try
                improved = True
        if verbose:
            print(f"[symmetry]   refine {it+1}/{refine_iterations}: score={best_score:.4f}")
        if not improved:
            break
 
    # --- Map back to original scale --------------------------------------------
    best_pt_orig = best_pt * scale + centroid
    best_n_unit  = best_n / np.linalg.norm(best_n)
 
    candidates_orig = [
        (s, pp * scale + centroid, pn / np.linalg.norm(pn))
        for s, pp, pn in scored
    ]
 
    if verbose:
        print(f"[symmetry] Final score  : {best_score:.4f}")
        print(f"[symmetry] Plane normal : {np.round(best_n_unit, 4)}")
        print(f"[symmetry] Plane point  : {np.round(best_pt_orig, 4)}")
 
    return SymmetryResult(
        plane_normal = best_n_unit,
        plane_point  = best_pt_orig,
        score        = best_score,
        threshold    = threshold * scale,   # store in original cloud units
        candidates   = candidates_orig,
    )
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  NMS + multi-axis detection
# ═══════════════════════════════════════════════════════════════════════════════
 
def _plane_normal_angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """Angle between two plane normals in [0, 90°], accounting for sign flip."""
    cos = abs(float(np.clip(np.dot(n1, n2), -1.0, 1.0)))
    return float(np.degrees(np.arccos(cos)))
 
 
def _nms_top_k(
    scored:        list,           # sorted (score, pt, normal) – best first
    k:             int,
    min_angle_deg: float = 15.0,
) -> list:
    """
    Greedy non-maximum suppression: pick up to *k* axes from the ranked candidate
    list such that every pair is at least *min_angle_deg* apart (plane-normal
    distance, ignoring sign).
    """
    selected = []
    for score, pt, n in scored:
        n_unit = n / np.linalg.norm(n)
        if all(
            _plane_normal_angle_deg(n_unit, sn) >= min_angle_deg
            for _, _, sn in selected
        ):
            selected.append((score, pt, n_unit))
        if len(selected) >= k:
            break
    return selected
 
 
def detect_multi_symmetry(
    points:              np.ndarray,
    k:                   int   = 3,
    min_angle_deg:       float = 15.0,
    n_random_candidates: int   = 500,
    threshold:           float = 0.02,
    refine_iterations:   int   = 4,
    refine_samples:      int   = 80,
    seed:                int   = 42,
    verbose:             bool  = True,
) -> MultiSymmetryResult:
    """
    Detect the top-K diverse reflective symmetry planes of a (partial) point cloud.
 
    Each axis is at least *min_angle_deg* away from every other detected axis
    (non-maximum suppression).  Each candidate is independently refined after
    NMS selection.
 
    Parameters
    ----------
    points              : (N, 3) float array
    k                   : number of symmetry planes to return
    min_angle_deg       : minimum angular separation between returned axes [degrees]
    n_random_candidates : random bisector planes in the initial pool
    threshold           : inlier fraction of cloud extent (e.g. 0.02 = 2%)
    refine_iterations   : refinement rounds per axis
    refine_samples      : perturbation samples per refinement round
    seed                : RNG seed
    verbose             : print progress
 
    Returns
    -------
    MultiSymmetryResult
    """
    pts = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
 
    # --- Normalise ---------------------------------------------------------------
    centroid = pts.mean(axis=0)
    scale    = np.linalg.norm(pts - centroid, axis=1).max()
    if scale < 1e-10:
        raise ValueError("Degenerate point cloud: all points coincide.")
 
    pts_n  = (pts - centroid) / scale
    thr_n  = threshold
    tree_n = cKDTree(pts_n)
    origin = np.zeros(3)
 
    # --- Build & score candidate pool --------------------------------------------
    cands  = _pca_candidates(pts_n, origin)
    cands += _perturbed_pca_candidates(pts_n, origin, rng)
    cands += _random_candidates(pts_n, n_random_candidates, rng)
 
    if verbose:
        print(f"[multi-sym] {len(pts)} pts | {len(cands)} candidates | k={k} | thr={threshold:.3f}")
 
    scored = [
        (_score(pts_n, pp, pn, thr_n, tree_n), pp.copy(), pn.copy())
        for pp, pn in cands
    ]
    scored.sort(key=lambda x: -x[0])
 
    # --- NMS: select k diverse axes ----------------------------------------------
    nms_axes = _nms_top_k(scored, k=k, min_angle_deg=min_angle_deg)
 
    if verbose:
        print(f"[multi-sym] NMS kept {len(nms_axes)}/{k} axes "
              f"(scores: {[round(s,3) for s,_,_ in nms_axes]})")
 
    # --- Refine each axis independently ------------------------------------------
    axes_results: List[SymmetryResult] = []
 
    for ax_idx, (best_score, best_pt, best_n) in enumerate(nms_axes):
        for it in range(refine_iterations):
            sigma    = 0.08 / (it + 1)
            improved = False
            for _ in range(refine_samples):
                perturb = rng.standard_normal(3) * sigma
                n_try   = best_n + perturb
                n_try  /= np.linalg.norm(n_try)
                s = _score(pts_n, best_pt, n_try, thr_n, tree_n)
                if s > best_score:
                    best_score, best_n = s, n_try
                    improved = True
            if not improved:
                break
 
        best_pt_orig = best_pt * scale + centroid
        best_n_unit  = best_n / np.linalg.norm(best_n)
 
        if verbose:
            print(f"[multi-sym]   axis {ax_idx}: normal={np.round(best_n_unit,3)}  "
                  f"score={best_score:.4f}")
 
        axes_results.append(SymmetryResult(
            plane_normal = best_n_unit,
            plane_point  = best_pt_orig,
            score        = best_score,
            threshold    = threshold * scale,
            candidates   = [],   # not repeated per-axis to save memory
        ))
 
    # Attach the full candidate list to the best axis only
    if axes_results:
        axes_results[0].candidates = [
            (s, pp * scale + centroid, pn / np.linalg.norm(pn))
            for s, pp, pn in scored
        ]
 
    return MultiSymmetryResult(
        axes          = axes_results,
        min_angle_deg = min_angle_deg,
        k_requested   = k,
        threshold     = threshold * scale,
    )
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  Open3D convenience wrapper  (matches your pipeline style)
# ═══════════════════════════════════════════════════════════════════════════════
 
def detect_symmetry_o3d(
    pcd: o3d.geometry.PointCloud,
    **kwargs,
) -> SymmetryResult:
    """
    Detect symmetry from an open3d.geometry.PointCloud.
    All **kwargs are forwarded to detect_symmetry().
 
    Example
    -------
        pcd, _ = txt_pcd_to_open3d("scan.txt")
        result  = detect_symmetry_o3d(pcd, threshold=0.02, n_random_candidates=600)
        print(result)
    """
    return detect_symmetry(np.asarray(pcd.points), **kwargs)
 
 
def detect_multi_symmetry_o3d(
    pcd: o3d.geometry.PointCloud,
    **kwargs,
) -> MultiSymmetryResult:
    """
    Detect top-K diverse symmetry planes from an open3d.geometry.PointCloud.
    All **kwargs are forwarded to detect_multi_symmetry().
 
    Example
    -------
        pcd, _ = txt_pcd_to_open3d("scan.txt")
        multi = detect_multi_symmetry_o3d(pcd, k=3, min_angle_deg=20)
        print(multi)
        geoms = multi_symmetry_geometries(multi, pcd)
    """
    return detect_multi_symmetry(np.asarray(pcd.points), **kwargs)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  Open3D geometry builders  (plug directly into your visualiser / scene)
# ═══════════════════════════════════════════════════════════════════════════════
 
def symmetry_geometries(
    result:         SymmetryResult,
    pcd:            o3d.geometry.PointCloud,
    axis_color:     Tuple = (1.0, 0.0, 0.0),
    plane_color:    Tuple = (1.0, 0.5, 0.0),
    reflect_color:  Tuple = (0.0, 0.8, 0.4),
    show_plane:     bool  = True,
    show_reflected: bool  = True,
) -> List[o3d.geometry.Geometry]:
    geoms: List[o3d.geometry.Geometry] = []

    pts    = np.asarray(pcd.points)
    n      = result.plane_normal
    p0     = result.plane_point
    extent = np.linalg.norm(pts - pts.mean(axis=0), axis=1).max()

    # ── Symmetry axis (LineSet) ─────────────────────────────────────────────────
    line_pts = np.array([p0 - n * extent, p0 + n * extent])
    ls = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(line_pts),
        lines  = o3d.utility.Vector2iVector([[0, 1]]),
    )
    ls.colors = o3d.utility.Vector3dVector([list(axis_color)])
    geoms.append(ls)

    # ── Symmetry plane (TriangleMesh quad) ──────────────────────────────────────
    if show_plane:
        ref = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
        u   = ref - (ref @ n) * n;  u /= np.linalg.norm(u)
        v   = np.cross(n, u)

        # Project every point onto the two in-plane axes
        centered  = pts - p0
        coords_u  = centered @ u
        coords_v  = centered @ v

        # Bound the quad tightly to the point cloud's bounding box
        u_min, u_max = coords_u.min(), coords_u.max()
        v_min, v_max = coords_v.min(), coords_v.max()

        corners = np.array([
            p0 + u_max * u + v_max * v,
            p0 + u_min * u + v_max * v,
            p0 + u_min * u + v_min * v,
            p0 + u_max * u + v_min * v,
        ])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(corners)
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        mesh.paint_uniform_color(list(plane_color))
        mesh.compute_vertex_normals()
        geoms.append(mesh)

    # ── Reflected cloud ─────────────────────────────────────────────────────────
    if show_reflected:
        ref_pts = _reflect(pts, p0, n)
        pcd_ref = o3d.geometry.PointCloud()
        pcd_ref.points = o3d.utility.Vector3dVector(ref_pts)
        pcd_ref.paint_uniform_color(list(reflect_color))
        geoms.append(pcd_ref)

    return geoms
 
 
# Palette for up to 6 axes (RGB floats)
_AXIS_PALETTE = [
    (1.0, 0.15, 0.15),   # red
    (0.15, 0.55, 1.0),   # blue
    (0.15, 0.85, 0.35),  # green
    (1.0, 0.75, 0.05),   # yellow
    (0.85, 0.15, 0.85),  # magenta
    (0.05, 0.85, 0.85),  # cyan
]
 
 
def multi_symmetry_geometries(
    multi:          MultiSymmetryResult,
    pcd:            o3d.geometry.PointCloud,
    show_planes:    bool = True,
    show_reflected: bool = False,   # off by default; clutters view with k axes
) -> List[o3d.geometry.Geometry]:
    """
    Build Open3D geometries for all axes in a MultiSymmetryResult.
    Each axis gets its own colour from the built-in palette.
 
    Returns a flat list of geometries ready for draw_geometries() or your scene.
    """
    geoms: List[o3d.geometry.Geometry] = []
    for i, ax in enumerate(multi.axes):
        color = _AXIS_PALETTE[i % len(_AXIS_PALETTE)]
        geoms += symmetry_geometries(
            ax, pcd,
            axis_color    = color,
            plane_color   = tuple(c * 0.6 for c in color),
            reflect_color = tuple(c * 0.8 for c in color),
            show_plane    = show_planes,
            show_reflected= show_reflected,
        )
    return geoms
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  reflect_pointcloud  – preserve colors + normals
# ═══════════════════════════════════════════════════════════════════════════════
 
def reflect_pointcloud(
    pcd:    o3d.geometry.PointCloud,
    result: SymmetryResult,
) -> o3d.geometry.PointCloud:
    """
    Return a new open3d.geometry.PointCloud that is the mirror image of *pcd*
    across the detected symmetry plane.  Colors and normals are preserved.
    """
    pts     = np.asarray(pcd.points)
    ref_pts = _reflect(pts, result.plane_point, result.plane_normal)
 
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(ref_pts)
 
    if pcd.has_colors():
        out.colors = pcd.colors
 
    if pcd.has_normals():
        norms     = np.asarray(pcd.normals)
        n         = result.plane_normal
        ref_norms = norms - 2.0 * (norms @ n)[:, None] * n
        out.normals = o3d.utility.Vector3dVector(ref_norms)
 
    return out