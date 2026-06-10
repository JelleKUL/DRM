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

def create_plane_meshes(
    plane_models: list[np.ndarray],
    bounding_box: o3d.geometry.AxisAlignedBoundingBox,
    max_extend_modifier: float = 2.0,
) -> list[o3d.geometry.TriangleMesh]:
    """
    Create finite plane meshes from RANSAC plane models, clipped to twice the
    bounding box extent, then cut each plane by all other planes to produce
    non-overlapping segments covering the full plane.

    Parameters
    ----------
    plane_models : List of [a, b, c, d] arrays (ax + by + cz + d = 0).
    bounding_box : AABB used to size the initial plane quads.
    max_extend_modifier : Multiplier for bounding box extent to size initial quads.

    Returns
    -------
    List of TriangleMesh fragments — all pieces of all planes after mutual cutting.
    """
    center  = bounding_box.get_center()
    extents = np.asarray(bounding_box.get_extent()) * max_extend_modifier / 2
    size    = np.linalg.norm(extents)

    plane_meshes = [_plane_model_to_mesh(m, center, size) for m in plane_models]

    result = []
    for i, mesh in enumerate(plane_meshes):
        # Start with the full quad and recursively split by every other plane
        fragments = [mesh]
        for j, model in enumerate(plane_models):
            if i == j:
                continue
            next_fragments = []
            for fragment in fragments:
                pos, neg = _split_mesh_by_plane(fragment, model)
                if pos is not None and len(pos.triangles) > 0:
                    orient_normals_toward(pos, center)
                    next_fragments.append(pos)
                if neg is not None and len(neg.triangles) > 0:
                    orient_normals_toward(neg, center)
                    next_fragments.append(neg)
            fragments = next_fragments

        result.extend(fragments)

    return result



# ── helpers ───────────────────────────────────────────────────────────────────

def _plane_model_to_mesh(
    model: np.ndarray,
    center: np.ndarray,
    size: float,
) -> o3d.geometry.TriangleMesh:
    a, b, c, d = model
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    t = -(np.dot(normal, center) + d) / np.dot(normal, normal)
    plane_center = center + t * normal

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, up);  u /= np.linalg.norm(u)
    v = np.cross(normal, u);   v /= np.linalg.norm(v)

    corners = np.array([
        plane_center + size * (-u - v),
        plane_center + size * ( u - v),
        plane_center + size * ( u + v),
        plane_center + size * (-u + v),
    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.compute_vertex_normals()
    return mesh


def _split_mesh_by_plane(
    mesh: o3d.geometry.TriangleMesh,
    model: np.ndarray,
) -> tuple[Optional[o3d.geometry.TriangleMesh], Optional[o3d.geometry.TriangleMesh]]:
    """
    Split a mesh into two halves along a plane.

    Returns
    -------
    pos : fragment where dot(normal, x) + d > 0  (in front of plane)
    neg : fragment where dot(normal, x) + d <= 0 (behind plane)
    """
    a, b, c, d = model
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    dists = np.dot(verts, normal) + d   # signed distance per vertex

    def intersect(p1, p2, d1, d2):
        t = d1 / (d1 - d2)
        return p1 + t * (p2 - p1)

    pos_verts, pos_faces = [], []
    neg_verts, neg_faces = [], []

    def add_vertex(buf, p):
        buf.append(p)
        return len(buf) - 1

    def clip_polygon(poly, poly_dists, keep_positive: bool):
        """Sutherland-Hodgman for one side."""
        clipped = []
        n = len(poly)
        for k in range(n):
            curr, nxt   = poly[k], poly[(k + 1) % n]
            dc, dn      = poly_dists[k], poly_dists[(k + 1) % n]
            inside_curr = (dc > 0) if keep_positive else (dc <= 0)
            inside_next = (dn > 0) if keep_positive else (dn <= 0)
            if inside_curr:
                clipped.append(curr)
            if inside_curr != inside_next:
                clipped.append(intersect(curr, nxt, dc, dn))
        return clipped

    for tri in faces:
        poly       = [verts[i] for i in tri]
        poly_dists = [dists[i] for i in tri]

        for keep_positive, vert_buf, face_buf in (
            (True,  pos_verts, pos_faces),
            (False, neg_verts, neg_faces),
        ):
            clipped = clip_polygon(poly, poly_dists, keep_positive)
            if len(clipped) >= 3:
                idx = [add_vertex(vert_buf, p) for p in clipped]
                for k in range(1, len(idx) - 1):
                    face_buf.append([idx[0], idx[k], idx[k + 1]])

    def build(vbuf, fbuf):
        if not fbuf:
            return None
        m = o3d.geometry.TriangleMesh()
        m.vertices  = o3d.utility.Vector3dVector(np.array(vbuf))
        m.triangles = o3d.utility.Vector3iVector(np.array(fbuf))
        m.merge_close_vertices(1e-8)
        m.remove_duplicated_vertices()
        m.remove_duplicated_triangles()
        m.compute_vertex_normals()
        return m

    return build(pos_verts, pos_faces), build(neg_verts, neg_faces)


def orient_normals_toward(
    mesh: o3d.geometry.TriangleMesh,
    target: np.ndarray,
) -> None:
    """
    Flip any triangle whose normal points away from `target` (in-place).
    """
    mesh.compute_triangle_normals()
    verts   = np.asarray(mesh.vertices)
    faces   = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.triangle_normals)

    # Face centroid → vector toward target
    centroids   = verts[faces].mean(axis=1)          # (F, 3)
    to_target   = target - centroids                 # (F, 3)
    facing_away = np.einsum("fd,fd->f", normals, to_target) < 0  # (F,)

    # Flip winding of offending triangles
    faces[facing_away] = faces[facing_away][:, ::-1]
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

def filter_planes_by_points(
    plane_meshes: list[o3d.geometry.TriangleMesh],
    plane_pcds: list[o3d.geometry.PointCloud],
    distance_threshold: float = 0.01,
    min_points: int = 10,
) -> tuple[list[o3d.geometry.TriangleMesh], list[o3d.geometry.PointCloud]]:
    scored = []

    for mesh in plane_meshes:
        counts = []
        for pcd in plane_pcds:
            count = _count_points_in_mesh_fragment(pcd, mesh, distance_threshold)
            counts.append(count)

        total_count = sum(counts)
        best_plane  = int(np.argmax(counts))

        if total_count >= min_points:
            scored.append((total_count, mesh, plane_pcds[best_plane]))

    scored.sort(key=lambda x: x[0], reverse=True)

    kept_meshes = [m for _, m, _ in scored]
    kept_pcds   = [p for _, _, p in scored]

    print(f"Kept {len(kept_meshes)}/{len(plane_meshes)} fragments.")
    return kept_meshes, kept_pcds

def _count_points_in_mesh_fragment(
    pcd: o3d.geometry.PointCloud,
    mesh: o3d.geometry.TriangleMesh,
    distance_threshold: float,
) -> int:
    if len(pcd.points) == 0 or len(mesh.triangles) == 0:
        return 0

    points = np.asarray(pcd.points)

    # Step 1: distance filter — keep points close to the plane surface
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=2000)
    distances = np.asarray(pcd.compute_point_cloud_distance(mesh_pcd))
    near_mask = distances <= distance_threshold
    if not np.any(near_mask):
        return 0

    near_points = points[near_mask]

    # Step 2: project onto the plane and test inside the fragment's 2D boundary
    inside_mask = _points_in_mesh_2d(near_points, mesh)

    return int(np.sum(inside_mask))


def _points_in_mesh_2d(
    points: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
) -> np.ndarray:
    """
    Test whether points lie inside a flat mesh by projecting everything onto
    the mesh plane and doing a 2D point-in-triangles test.

    Works for any mesh orientation — projects into the mesh's local UV frame.
    """
    mesh.compute_triangle_normals()
    verts   = np.asarray(mesh.vertices)
    faces   = np.asarray(mesh.triangles)
    normal  = np.asarray(mesh.triangle_normals).mean(axis=0)
    normal /= np.linalg.norm(normal)

    # Build a local 2D frame (u, v) in the plane
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, up);  u /= np.linalg.norm(u)
    v = np.cross(normal, u);   v /= np.linalg.norm(v)

    # Project vertices and query points onto the 2D frame
    origin    = verts.mean(axis=0)
    verts_2d  = np.stack([np.dot(verts  - origin, u),
                          np.dot(verts  - origin, v)], axis=1)
    points_2d = np.stack([np.dot(points - origin, u),
                          np.dot(points - origin, v)], axis=1)

    # Test each point against all triangles
    inside = np.zeros(len(points), dtype=bool)
    for tri in faces:
        a, b, c = verts_2d[tri[0]], verts_2d[tri[1]], verts_2d[tri[2]]
        inside |= _points_in_triangle_2d(points_2d, a, b, c)

    return inside


def _points_in_triangle_2d(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """
    Barycentric test: True where each point lies inside triangle (a, b, c).
    """
    v0 = c - a
    v1 = b - a
    v2 = points - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot11 = np.dot(v1, v1)
    dot02 = v2 @ v0
    dot12 = v2 @ v1

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-10:
        return np.zeros(len(points), dtype=bool)

    inv = 1.0 / denom
    u_coord = (dot11 * dot02 - dot01 * dot12) * inv
    v_coord = (dot00 * dot12 - dot01 * dot02) * inv

    return (u_coord >= 0) & (v_coord >= 0) & (u_coord + v_coord <= 1)

def plane_pointclouds_to_meshes(
    plane_pcds:    list[o3d.geometry.PointCloud],
    plane_meshes:  list[o3d.geometry.TriangleMesh],
    scanner_center: np.ndarray,
    voxel_size: Optional[float] = None,
) -> list[o3d.geometry.TriangleMesh]:
    """
    For each plane fragment:
      1. Subdivide the fragment mesh to match the pcd voxel density.
      2. Project each subdivided vertex onto the pcd surface by displacing
         it along the plane normal to the closest pcd point.
      3. Orient normals toward the scanner center.

    Parameters
    ----------
    plane_pcds      : Segmented plane point clouds.
    plane_meshes    : Corresponding filtered plane mesh fragments.
    scanner_center  : 3D position of the scanner (e.g. transform[:3, 3]).
    voxel_size       : Voxel size for the point cloud.
    """
    from scipy.spatial import cKDTree

    result = []

    for pcd, mesh in zip(plane_pcds, plane_meshes):
        points = np.asarray(pcd.points)

        if len(points) < 3 or len(mesh.triangles) == 0:
            result.append(o3d.geometry.TriangleMesh())
            continue

        
        if voxel_size is None:
            voxel_size = float(np.median(
                np.asarray(pcd.compute_nearest_neighbor_distance())
            ))

        centroid  = points.mean(axis=0)
        _, _, vh  = np.linalg.svd(points - centroid)
        normal    = vh[2];  normal /= np.linalg.norm(normal)

        subdiv_mesh = _subdivide_mesh_to_density(mesh, voxel_size)

        verts = np.asarray(subdiv_mesh.vertices).copy()
        tree  = cKDTree(points)

        _, nn_idx   = tree.query(verts, k=1)
        closest_pts = points[nn_idx]
        delta       = closest_pts - verts
        perp_dist   = np.einsum("vd,d->v", delta, normal)
        displaced   = verts + perp_dist[:, None] * normal

        subdiv_mesh.vertices = o3d.utility.Vector3dVector(displaced)
        subdiv_mesh.compute_vertex_normals()
        subdiv_mesh.compute_triangle_normals()
        subdiv_mesh.orient_triangles()

        # --- orient normals toward scanner ---
        orient_normals_toward(subdiv_mesh, scanner_center)

        result.append(subdiv_mesh)
        print(f"Projected mesh: {len(displaced)} verts, "
              f"{len(np.asarray(subdiv_mesh.triangles))} tris, "
              f"voxel_size={voxel_size:.4f}")

    return result

def _get_boundary_edges(
    mesh: o3d.geometry.TriangleMesh,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Extract boundary edges (edges belonging to only one triangle) from a mesh.

    Returns
    -------
    verts  : (V, 3) array of all mesh vertices.
    edges  : List of [i, j] index pairs forming the boundary loop(s),
             indexed into `verts`.
    """
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    edge_count = {}
    for tri in faces:
        for i in range(3):
            edge = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary_edges = [
        [a, b] for (a, b), count in edge_count.items() if count == 1
    ]

    return verts, boundary_edges

def _subdivide_boundary(
    verts: np.ndarray,
    edges: list[list[int]],
    max_edge_length: float,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Subdivide boundary edges so no segment is longer than `max_edge_length`.
    Inserts evenly-spaced intermediate vertices along each edge.

    Parameters
    ----------
    verts           : (V, 3) array of mesh vertices.
    edges           : List of [i, j] boundary edge index pairs.
    max_edge_length : Maximum allowed edge length.

    Returns
    -------
    verts : Extended vertex array including new intermediate vertices.
    edges : New edge list referencing the extended vertex array.
    """
    verts     = list(verts)
    new_edges = []

    for a, b in edges:
        p0      = np.array(verts[a])
        p1      = np.array(verts[b])
        seg_len = np.linalg.norm(p1 - p0)
        n_div   = max(1, int(np.ceil(seg_len / max_edge_length)))

        prev_idx = a
        for k in range(1, n_div):
            t       = k / n_div
            new_pt  = p0 + t * (p1 - p0)
            new_idx = len(verts)
            verts.append(new_pt)
            new_edges.append([prev_idx, new_idx])
            prev_idx = new_idx
        new_edges.append([prev_idx, b])

    return np.array(verts), new_edges

def _subdivide_mesh_to_density(
    mesh: o3d.geometry.TriangleMesh,
    target_edge_length: float,
) -> o3d.geometry.TriangleMesh:
    """
    Remesh a flat plane mesh into a regular quad grid with a given spacing,
    preserving the exact boundary edges of the original mesh.

    Strategy:
    1. Project onto best-fit plane.
    2. Build regular grid interior points.
    3. Subdivide boundary edges to target spacing.
    4. Triangulate interior + boundary together using constrained Delaunay.
    5. Unproject back to 3D.
    """
    import triangle as tr

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles)

    # --- fit plane via SVD ---
    centroid = verts.mean(axis=0)
    _, _, vh = np.linalg.svd(verts - centroid)
    normal   = vh[2];  normal /= np.linalg.norm(normal)

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    u_ax = np.cross(up, normal);  u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(normal, u_ax); v_ax /= np.linalg.norm(v_ax)
    if np.dot(v_ax, up) < 0:
        v_ax = -v_ax;  u_ax = -u_ax

    def to_2d(pts):
        return np.stack([np.dot(pts - centroid, u_ax),
                         np.dot(pts - centroid, v_ax)], axis=1)

    def to_3d(pts_2d):
        return centroid + pts_2d[:, 0:1] * u_ax + pts_2d[:, 1:2] * v_ax

    verts_2d = to_2d(verts)

    # --- extract and subdivide boundary edges ---
    _, boundary_edges = _get_boundary_edges(mesh)
    boundary_verts_3d, boundary_edges = _subdivide_boundary(
        verts, boundary_edges, target_edge_length
    )
    boundary_verts_2d = to_2d(boundary_verts_3d)
    n_boundary = len(boundary_verts_2d)

    # --- build regular interior grid points ---
    u_min, u_max = verts_2d[:, 0].min(), verts_2d[:, 0].max()
    v_min, v_max = verts_2d[:, 1].min(), verts_2d[:, 1].max()

    u_steps = max(1, int(np.ceil((u_max - u_min) / target_edge_length)))
    v_steps = max(1, int(np.ceil((v_max - v_min) / target_edge_length)))

    u_lin = np.linspace(u_min, u_max, u_steps + 1)[1:-1]  # exclude edges
    v_lin = np.linspace(v_min, v_max, v_steps + 1)[1:-1]
    uu, vv = np.meshgrid(u_lin, v_lin)
    interior_pts = np.stack([uu.ravel(), vv.ravel()], axis=1)

    # Keep only interior points that fall inside the original mesh boundary
    if len(interior_pts) > 0:
        in_mesh = np.zeros(len(interior_pts), dtype=bool)
        for tri in faces:
            a = verts_2d[tri[0]]
            b = verts_2d[tri[1]]
            c = verts_2d[tri[2]]
            in_mesh |= _points_in_triangle_2d(interior_pts, a, b, c)
        interior_pts = interior_pts[in_mesh]

    # --- combine boundary + interior vertices ---
    if len(interior_pts) > 0:
        all_verts_2d = np.vstack([boundary_verts_2d, interior_pts])
    else:
        all_verts_2d = boundary_verts_2d

    # --- constrained Delaunay triangulation ---
    tri_input = {
        "vertices": all_verts_2d,
        "segments": np.array(boundary_edges, dtype=np.int32),
    }
    tri_output = tr.triangulate(tri_input, opts="pD")

    if "triangles" not in tri_output or len(tri_output["triangles"]) == 0:
        return mesh

    out_verts_2d = tri_output["vertices"]
    out_faces    = tri_output["triangles"]

    # --- filter triangles outside boundary (holes) ---
    # Test each triangle centroid against original mesh
    centroids_2d = out_verts_2d[out_faces].mean(axis=1)
    keep = np.zeros(len(out_faces), dtype=bool)
    for tri in faces:
        a = verts_2d[tri[0]]
        b = verts_2d[tri[1]]
        c = verts_2d[tri[2]]
        keep |= _points_in_triangle_2d(centroids_2d, a, b, c)
    out_faces = out_faces[keep]

    # --- unproject to 3D ---
    out_verts_3d = to_3d(out_verts_2d)

    result = o3d.geometry.TriangleMesh()
    result.vertices  = o3d.utility.Vector3dVector(out_verts_3d)
    result.triangles = o3d.utility.Vector3iVector(out_faces)
    result.remove_unreferenced_vertices()
    result.compute_vertex_normals()
    result.compute_triangle_normals()
    result.orient_triangles()

    return result

def sample_unoccupied_plane_points(
    plane_meshes: list[o3d.geometry.TriangleMesh],
    plane_pcds: list[o3d.geometry.PointCloud],
    voxel_size: Optional[float] = None,
) -> list[o3d.geometry.PointCloud]:
    """
    For each plane fragment, sample points at the same voxel density as its
    corresponding point cloud, then keep only the sampled points that are
    NOT within one voxel size of any existing point.

    Parameters
    ----------
    plane_meshes : Filtered plane mesh fragments.
    plane_pcds   : Corresponding segmented point clouds, same order.
    voxel_size   : Voxel size for the point cloud.

    Returns
    -------
    List of PointClouds containing only the unoccupied sampled points.
    """
    result = []

    for mesh, pcd in zip(plane_meshes, plane_pcds):
        pcd_points = np.asarray(pcd.points)
        n_pcd      = len(pcd_points)

        if n_pcd < 3:
            result.append(o3d.geometry.PointCloud())
            continue

        # --- estimate voxel size from median nn distance ---
        if(voxel_size is None):
            nn_distances = np.asarray(pcd.compute_nearest_neighbor_distance())
            voxel_size   = float(np.median(nn_distances))

        # --- sample mesh surface densely then voxel downsample ---
        # oversample so voxel grid is well populated
        n_oversample = max(int(mesh.get_surface_area() / (voxel_size ** 2)) * 4, 1000)
        sampled      = mesh.sample_points_uniformly(number_of_points=n_oversample)
        sampled_down = sampled.voxel_down_sample(voxel_size)

        # --- keep only sampled points far from existing pcd points ---
        distances       = np.asarray(sampled_down.compute_point_cloud_distance(pcd))
        unoccupied_mask = distances > voxel_size

        unoccupied_pts = np.asarray(sampled_down.points)[unoccupied_mask]

        unoccupied_pcd        = o3d.geometry.PointCloud()
        unoccupied_pcd.points = o3d.utility.Vector3dVector(unoccupied_pts)

        result.append(unoccupied_pcd)
        print(f"Plane: {n_pcd} existing, {len(np.asarray(sampled_down.points))} voxel-sampled, "
              f"{len(unoccupied_pts)} unoccupied (voxel_size={voxel_size:.4f})")
    return result

def assign_plane_uvs(
    plane_meshes: list[o3d.geometry.TriangleMesh],
    scene_center: np.ndarray,
) -> list[o3d.geometry.TriangleMesh]:
    """
    Assign UV coordinates to each plane mesh fragment such that:
    - U axis is horizontal (right when viewed from the scene center)
    - V axis is up (Z in world space, or Y for horizontal planes)
    - UV [0,0] is bottom-left, [1,1] is top-right when viewed from center

    Parameters
    ----------
    plane_meshes : Filtered plane mesh fragments.
    center : Used to determine the scene center viewpoint.

    Returns
    -------
    List of TriangleMesh with UV coordinates assigned.
    """
    result = []

    for mesh in plane_meshes:
        mesh = o3d.geometry.TriangleMesh(mesh)  # copy

        mesh.compute_triangle_normals()
        verts   = np.asarray(mesh.vertices)
        faces   = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.triangle_normals)

        # Average face normal as plane normal
        normal  = normals.mean(axis=0)
        normal /= np.linalg.norm(normal)

        # Flip normal to point toward scene center
        centroid   = verts.mean(axis=0)
        to_center  = scene_center - centroid
        if np.dot(normal, to_center) < 0:
            normal = -normal

        # --- choose up vector ---
        world_up = np.array([0.0, 0.0, 1.0])
        is_horizontal = abs(np.dot(normal, world_up)) > 0.9
        up = np.array([0.0, 1.0, 0.0]) if is_horizontal else world_up

        # --- build UV frame: right = u, up = v ---
        # u points right when viewed from scene center
        # For vertical planes: u = normal × up  (right-hand: cross of outward normal and Z gives rightward axis when viewed from outside)
        # Flip u if it points away from the "right" direction as seen from center
        u = np.cross(up, normal);  u /= np.linalg.norm(u)
        v = np.cross(normal, u);   v /= np.linalg.norm(v)

        # Ensure v points up (positive Z, or positive Y for horizontal)
        if np.dot(v, up) < 0:
            v = -v
            u = -u

        # --- project vertices onto UV frame ---
        proj_u = np.dot(verts - centroid, u)   # (V,)
        proj_v = np.dot(verts - centroid, v)   # (V,)

        # Normalise to [0, 1]
        u_min, u_max = proj_u.min(), proj_u.max()
        v_min, v_max = proj_v.min(), proj_v.max()

        u_range = u_max - u_min if u_max > u_min else 1.0
        v_range = v_max - v_min if v_max > v_min else 1.0

        uvs_per_vert = np.stack([
            (proj_u - u_min) / u_range,
            (proj_v - v_min) / v_range,
        ], axis=1)                              # (V, 2)

        # Open3D UVs are per triangle-vertex — unroll
        triangle_uvs = uvs_per_vert[faces.flatten()]  # (F*3, 2)

        mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)

        result.append(mesh)

    return result

def apply_texture_to_planes(
    plane_meshes: Union[o3d.geometry.TriangleMesh, list],
    images:       Union[str, np.ndarray, "Image.Image", list],
) -> list[o3d.geometry.TriangleMesh]:
    """
    Apply image texture(s) to a list of UV-unwrapped plane meshes.

    Parameters
    ----------
    plane_meshes : Single mesh or list of UV-unwrapped TriangleMeshes.
    images       : One of:
                   - A single image (path, numpy array, or PIL Image)
                     → applied to all meshes
                   - A list of images of the same length as plane_meshes
                     → applied in order

    Returns
    -------
    List of textured TriangleMesh.
    """
    from PIL import Image as PILImage

    # --- normalise meshes to list ---
    if isinstance(plane_meshes, o3d.geometry.TriangleMesh):
        plane_meshes = [plane_meshes]

    # --- normalise images to list ---
    if not isinstance(images, list):
        images = [images]

    # --- broadcast single image to all meshes ---
    if len(images) == 1:
        images = images * len(plane_meshes)

    # --- validate lengths ---
    if len(images) != len(plane_meshes):
        raise ValueError(
            f"Number of images ({len(images)}) must be 1 or match "
            f"number of meshes ({len(plane_meshes)})."
        )

    def _to_o3d_image(img) -> o3d.geometry.Image:
        if isinstance(img, o3d.geometry.Image):
            return img
        if isinstance(img, str):
            return o3d.io.read_image(img)
        if isinstance(img, PILImage.Image):
            return o3d.geometry.Image(np.ascontiguousarray(np.array(img)))
        if isinstance(img, np.ndarray):
            return o3d.geometry.Image(np.ascontiguousarray(img))
        raise TypeError(f"Unsupported image type: {type(img)}")

    result = []
    for mesh, img in zip(plane_meshes, images):
        out = o3d.geometry.TriangleMesh(mesh)
        out.textures = [_to_o3d_image(img)]
        out.triangle_material_ids = o3d.utility.IntVector(
            np.zeros(len(mesh.triangles), dtype=np.int32)
        )
        result.append(out)

    return result

def draw_plane_corners_on_pano(
    plane_meshes: list[o3d.geometry.TriangleMesh | o3d.geometry.PointCloud],
    pano_image:   np.ndarray,
    transform:    np.ndarray,
    radius:       int = 10,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    if not isinstance(pano_image, np.ndarray):
        pano_image = np.array(pano_image)

    Ry = np.array([
        [ 0, 0, -1, 0],
        [ 0, 1,  0, 0],
        [ 1, 0,  0, 0],
        [ 0, 0,  0, 1]
    ], dtype=np.float64)

    tMat = Ry @ np.linalg.inv(transform)

    pano_h, pano_w = pano_image.shape[:2]
    canvas = Image.fromarray(pano_image.copy())
    draw   = ImageDraw.Draw(canvas)

    colors = [
        (255,   0,   0),
        (  0, 255,   0),
        (  0,   0, 255),
        (255, 255,   0),
        (255,   0, 255),
        (  0, 255, 255),
    ]

    if not isinstance(plane_meshes, list):
        plane_meshes = [plane_meshes]

    for i, geometry in enumerate(plane_meshes):
        if isinstance(geometry, o3d.geometry.PointCloud):
            verts = np.asarray(geometry.points, dtype=np.float64)
        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            verts = np.asarray(geometry.vertices, dtype=np.float64)
        else:
            raise TypeError(f"Unsupported geometry type: {type(geometry)}")

        color = colors[i % len(colors)]

        uv = drm.transform_xyz_to_uv(verts, tMat)     # (V, 2)

        for u, v in uv:
            px = int(round(u * (pano_w - 1)))
            py = int(round(v * (pano_h - 1)))
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                fill=color, outline=(0, 0, 0),
            )

        print(f"Geometry {i} ({type(geometry).__name__}): {len(verts)} points, color={color}")

    return np.array(canvas)


def project_points_to_pano(
    points: np.ndarray,
    pano_wh: tuple[int, int],
    *,
    camera_center: np.ndarray | None = None,
    camera_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """
    Project 3-D world points onto an equirectangular panorama using the
    same coordinate convention as ``pano_to_ply.py``:
 
        yaw   = (1 - u/W) * 2π   →  u=0  ≡  X-axis  (left edge)
        pitch = (0.5 - v/H) * π  →  v=0  ≡  top      (Z-up)
 
    The inverse mapping (world → pixel) is therefore:
 
        yaw   = atan2(Y_cam, X_cam)           ∈ (-π, π]
        pitch = arcsin(Z_cam / r)             ∈ [-π/2, π/2]
 
        u = (1 - yaw  / (2π)) mod 1           normalised ∈ [0, 1)
        v =  0.5 - pitch / π                  normalised ∈ [0, 1]
 
    Parameters
    ----------
    points : (N, 3) float array
        3-D points in **world** coordinates (same frame as the PLY).
 
    pano_wh : (W, H)
        Width and height of the target panorama in pixels.
 
    camera_center : (3,) float array, optional
        World-space camera centre **C** (the ``t`` in R·(p - t)).
        When the point cloud was created *without* ``--apply_transform``
        it is already in camera-local coords, so leave this as ``None``.
 
    camera_rotation : (3, 3) float array, optional
        Camera-to-world rotation matrix **R** from the pose file.
        Pass the *same* matrix returned by ``read_pose()`` when
        ``apply_transform`` was used.  The inverse (world → camera)
        is computed internally.
 
    Returns
    -------
    uv : (N, 2) float array
        Normalised UV coordinates in [0, 1] × [0, 1].
        Multiply by (W-1, H-1) to get pixel coordinates.
        Points behind the camera (r ≈ 0) receive ``nan``.
    """
    points = np.asarray(points, dtype=np.float64)   # (N, 3)
 
    # ------------------------------------------------------------------
    # 1.  Transform world → camera-local  (iff pose is provided)
    # ------------------------------------------------------------------
    if camera_center is not None or camera_rotation is not None:
        if camera_center is None or camera_rotation is None:
            raise ValueError(
                "Provide both camera_center and camera_rotation, or neither."
            )
        C = np.asarray(camera_center,  dtype=np.float64)   # (3,)
        R = np.asarray(camera_rotation, dtype=np.float64)  # (3, 3)  cam→world
        # camera-local = R^T · (p - C)
        pts_cam = (points - C) @ R   # R^T applied as right-multiply
    else:
        pts_cam = points
 
    X = pts_cam[:, 0]
    Y = pts_cam[:, 1]
    Z = pts_cam[:, 2]
 
    # ------------------------------------------------------------------
    # 2.  Cartesian → spherical
    #     Convention matches pano_to_ply.py:
    #       X = cos(pitch)·cos(yaw)
    #       Y = cos(pitch)·sin(yaw)
    #       Z = sin(pitch)
    # ------------------------------------------------------------------
    r = np.sqrt(X**2 + Y**2 + Z**2)
 
    # Mask degenerate points
    valid = r > 1e-9
    yaw   = np.where(valid, np.arctan2(Y, X), np.nan)   # atan2(Y,X): (-π, π]
    pitch = np.where(valid, np.arcsin(np.clip(Z / np.where(valid, r, 1.0),
                                              -1.0, 1.0)), np.nan)
 
    # ------------------------------------------------------------------
    # 3.  Spherical → normalised UV
    #     Inverted from:
    #       yaw   = (1 - u/W) * 2π   →   u/W  =  1 - yaw/(2π)
    #       pitch = (0.5 - v/H) * π  →   v/H  =  0.5 - pitch/π
    # ------------------------------------------------------------------
    u_norm = (1.0 - yaw   / (2.0 * np.pi)) % 1.0   # wrap to [0, 1)
    v_norm =  0.5 - pitch / np.pi                   # [0, 1]
 
    uv = np.stack([u_norm, v_norm], axis=-1)        # (N, 2)
    uv[~valid] = np.nan
 
    return uv.astype(np.float32)
 
 
# ---------------------------------------------------------------------------
# Convenience: draw projected points onto a panorama image
# ---------------------------------------------------------------------------
 
def draw_points_on_pano(
    points:  np.ndarray,
    pano_image: np.ndarray,
    *,
    camera_center:   np.ndarray | None = None,
    camera_rotation: np.ndarray | None = None,
    color:   tuple[int, int, int] = (255, 0, 0),
    radius:  int = 5,
) -> np.ndarray:
    """
    Project ``points`` onto ``pano_image`` and paint filled circles.
 
    Parameters
    ----------
    points : (N, 3) float array  –  world-space 3-D points
    pano_image : (H, W, 3) uint8 numpy array
    camera_center / camera_rotation : see ``project_points_to_pano``
    color  : RGB fill colour for every dot
    radius : circle radius in pixels
 
    Returns
    -------
    Annotated copy of ``pano_image`` as a numpy array.
    """
    from PIL import Image, ImageDraw
 
    pano_h, pano_w = pano_image.shape[:2]
    uv = project_points_to_pano(
        points, (pano_w, pano_h),
        camera_center=camera_center,
        camera_rotation=camera_rotation,
    )
 
    canvas = Image.fromarray(pano_image.copy())
    draw   = ImageDraw.Draw(canvas)
 
    for u_n, v_n in uv:
        if np.isnan(u_n) or np.isnan(v_n):
            continue
        px = int(round(u_n * (pano_w - 1)))
        py = int(round(v_n * (pano_h - 1)))
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            fill=color, outline=(0, 0, 0),
        )
 
    return np.array(canvas)

def sample_pano_textures(
    plane_meshes: list[o3d.geometry.TriangleMesh],
    pano_image:   np.ndarray,
    transform:    np.ndarray,
    base_width:   int = 1024,
) -> list[np.ndarray]:
    """
    Sample a panorama onto each plane mesh and return the raw texture images.
    Preserves alpha channel if present in the input panorama.

    Returns
    -------
    List of (H, W, 3) or (H, W, 4) uint8 numpy arrays, one per mesh.
    """
    from scipy.ndimage import map_coordinates

    if not isinstance(pano_image, np.ndarray):
        pano_image = np.array(pano_image)

    n_channels = pano_image.shape[2]   # 3 for RGB, 4 for RGBA

    Ry = np.array([
        [ 0, 0, -1, 0],
        [ 0, 1,  0, 0],
        [ 1, 0,  0, 0],
        [ 0, 0,  0, 1]
    ], dtype=np.float64)

    tMat = Ry @ np.linalg.inv(transform)

    textures = []

    for mesh in plane_meshes:
        verts = np.asarray(mesh.vertices, dtype=np.float64)

        centroid = verts.mean(axis=0)
        _, _, vh = np.linalg.svd(verts - centroid)
        normal   = vh[2];  normal /= np.linalg.norm(normal)

        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(normal, up)) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
        u_ax = np.cross(up, normal);  u_ax /= np.linalg.norm(u_ax)
        v_ax = np.cross(normal, u_ax); v_ax /= np.linalg.norm(v_ax)
        if np.dot(v_ax, up) < 0:
            v_ax = -v_ax;  u_ax = -u_ax

        proj_u = np.dot(verts - centroid, u_ax)
        proj_v = np.dot(verts - centroid, v_ax)
        u_min, u_max = proj_u.min(), proj_u.max()
        v_min, v_max = proj_v.min(), proj_v.max()
        u_range = u_max - u_min
        v_range = v_max - v_min

        aspect = v_range / u_range if u_range > 0 else 1.0
        tex_w  = base_width
        tex_h  = max(1, int(round(base_width * aspect)))

        px_lin = np.linspace(0.0, 1.0, tex_w)
        py_lin = np.linspace(1.0, 0.0, tex_h)
        uu, vv = np.meshgrid(px_lin, py_lin)

        pts_3d = (
            centroid
            + (uu * u_range + u_min)[..., None] * u_ax
            + (vv * v_range + v_min)[..., None] * v_ax
        ).reshape(-1, 3)

        uv     = drm.transform_xyz_to_uv(pts_3d, tMat)
        pano_h, pano_w = pano_image.shape[:2]
        coords = np.stack([uv[:, 1] * (pano_h - 1),
                           uv[:, 0] * (pano_w - 1)], axis=0)

        texture = np.zeros((tex_h * tex_w, n_channels), dtype=np.uint8)
        for c in range(n_channels):
            sampled = map_coordinates(
                pano_image[..., c].astype(np.float32),
                coords, order=1, mode="wrap",
            )
            texture[:, c] = sampled.clip(0, 255).astype(np.uint8)

        textures.append(texture.reshape(tex_h, tex_w, n_channels))
        print(f"Plane texture: {tex_w}×{tex_h}px  ({n_channels}ch), "
              f"plane size: {u_range:.2f}×{v_range:.2f}m")

    return textures


def project_pano_onto_planes(
    plane_meshes: list[o3d.geometry.TriangleMesh],
    pano_image:   np.ndarray,
    transform:    np.ndarray,
    base_width:   int = 1024,
) -> list[o3d.geometry.TriangleMesh]:
    """
    Sample a panorama onto each plane mesh and return textured TriangleMeshes.
    """
    textures = sample_pano_textures(plane_meshes, pano_image, transform, base_width)

    result = []
    for mesh, texture in zip(plane_meshes, textures):
        out = o3d.geometry.TriangleMesh(mesh)
        out.textures = [o3d.geometry.Image(texture)]
        out.triangle_material_ids = o3d.utility.IntVector(
            np.zeros(len(mesh.triangles), dtype=np.int32)
        )
        result.append(out)

    return result

def generate_occlusion_mask_image(
    pano_image:       np.ndarray,
    empty_pano_image: np.ndarray,
    threshold:        int   = 20,
    blur_radius:      float = 2.0,
) -> np.ndarray:
    """
    Generate an RGBA panorama where pixels that differ significantly between
    the occupied and empty panorama are made transparent (occlusion mask).

    Parameters
    ----------
    pano_image       : (H, W, 3) uint8 RGB occupied panorama.
    empty_pano_image : (H, W, 3) uint8 RGB empty panorama.
    threshold        : Per-channel mean difference above which a pixel is
                       considered occluded (made transparent).
    blur_radius      : Gaussian blur radius applied to the difference mask
                       before thresholding, to smooth jagged edges.

    Returns
    -------
    (H, W, 4) uint8 RGBA image — transparent where occupied != empty.
    """
    from scipy.ndimage import gaussian_filter

    if not isinstance(pano_image, np.ndarray):
        pano_image = np.array(pano_image)
    if not isinstance(empty_pano_image, np.ndarray):
        empty_pano_image = np.array(empty_pano_image)

    # --- per-pixel mean absolute difference across channels ---
    diff = np.abs(pano_image.astype(np.float32) - empty_pano_image.astype(np.float32))
    diff_mean = diff.mean(axis=-1)                     # (H, W)

    # --- optional blur to smooth the mask edges ---
    if blur_radius > 0:
        diff_mean = gaussian_filter(diff_mean, sigma=blur_radius)

    # --- build RGBA output ---
    rgba        = np.zeros((*pano_image.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = pano_image
    rgba[..., 3]  = 255                                # fully opaque by default

    # Pixels that differ → transparent
    occluded          = diff_mean > threshold
    rgba[occluded] = [0,0,0,0]

    print(f"Occluded: {occluded.sum()} px "
          f"({100 * occluded.mean():.1f}% of image)")

    return rgba

def dilate_mask(
    texture: np.ndarray,
    dilation_px: int = 5,
) -> np.ndarray:
    """
    Dilate the transparent (alpha=0) region of an RGBA texture.

    Parameters
    ----------
    texture     : (H, W, 4) uint8 RGBA array where alpha=0 is the mask.
    dilation_px : Number of pixels to expand the mask outward.

    Returns
    -------
    (H, W, 4) RGBA array with expanded transparent region.
    """
    from scipy.ndimage import binary_dilation

    if texture.shape[2] != 4:
        return texture

    out   = texture.copy()
    mask  = texture[..., 3] == 0                        # True where transparent
    dilated = binary_dilation(mask, iterations=dilation_px)
    out[dilated, 3] = 0                                 # expand transparent region

    return out


def dilate_masks(
    textures:    list[np.ndarray],
    dilation_px: int = 5,
) -> list[np.ndarray]:
    return [dilate_mask(t, dilation_px) for t in textures]

####### INPAINTING #######

def inpaint_plane_textures(
    plane_textures: list[np.ndarray],
    dilation_px: int = 5,
    simple_lama=None,
) -> list[np.ndarray]:
    """
    Inpaint transparent regions in RGBA plane textures using SimpleLama.

    Parameters
    ----------
    plane_textures : List of (H, W, 4) uint8 RGBA arrays, alpha=0 = inpaint region.
    simple_lama    : Optional pre-loaded SimpleLama instance. If None, a new
                     instance is created. Pass a pre-loaded instance when
                     processing multiple scenes to avoid reloading model weights.

    Returns
    -------
    List of (H, W, 3) uint8 RGB inpainted arrays.
    """
    from PIL import Image

    if simple_lama is None:
        from simple_lama_inpainting import SimpleLama
        simple_lama = SimpleLama()

    inpainted = []

    for i, texture in enumerate(plane_textures):
        if texture.shape[2] != 4:
            inpainted.append(np.ascontiguousarray(texture[..., :3]))
            continue

        texture = dilate_mask(texture, dilation_px)
        alpha = texture[..., 3]
        mask  = (alpha == 0)

        if not mask.any():
            inpainted.append(np.ascontiguousarray(texture[..., :3]))
            continue

        h, w      = texture.shape[:2]
        image_pil = Image.fromarray(texture[..., :3]).convert("RGB")
        mask_pil  = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")

        result = simple_lama(image_pil, mask_pil)
        inpainted.append(np.ascontiguousarray(np.asarray(result)))

        print(f"Texture {i}: inpainted {mask.sum()} px "
              f"({100 * mask.mean():.1f}% of {w}×{h})")

    return inpainted

