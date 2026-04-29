"""Tools to combine meshes and pointclouds."""
import numpy as np
import open3d as o3d
from typing import Tuple, List
from scipy.spatial import Delaunay
import trimesh


def combine_geometry(ogGeometry: o3d.geometry.PointCloud, newGeometry: o3d.geometry.PointCloud, newScannerPos: np.ndarray,
                     distanceTreshold: float = 0.05, checkVisibility = False, logProcess=False) -> o3d.geometry.PointCloud:
    """Combines 2 aligned geometries assuming ogGeometry is the reference and
    newGeometry will supplement it.

    Steps:
        1) Create a convex hull of newGeometry
        2) Filter relevant points of ogGeometry (inside hull)
        3) Coverage check: points too far from newGeometry are candidates for removal
        4) Visibility check: uncovered points inside the mesh are kept, outside are removed
        5) Filter newGeometry to only keep changed points
        6) Combine everything

    Args:
        ogGeometry       : reference geometry (PointCloud)
        newGeometry      : new geometry to supplement with (PointCloud)
        distanceTreshold : distance threshold for coverage filtering
        logProcess       : print progress steps

    Returns:
        o3d.geometry.PointCloud: the combined geometry
    """
    # Step 1: Convex hull of newGeometry
    newGeoHull, _ = newGeometry.compute_convex_hull()
    if logProcess: print("Convex hull created")

    # Step 2: Filter irrelevant ogGeometry points (outside hull)
    relevantOg, irrelevantOg = get_points_in_hull(ogGeometry, newGeoHull)
    if logProcess: print("Irrelevant points filtered")

    # Step 3: Coverage check
    coveredPoints, unCoveredPoints = filter_pcd_by_distance(relevantOg, newGeometry, distanceTreshold)
    if logProcess: print("Covered points calculated")

    # Step 4: Visibility check on uncovered points
    if(checkVisibility):
        invisibleUncoveredPoints, visibleUncoveredPoints = get_invisible_points_grid(
            unCoveredPoints,
            newGeometry,
            newScannerPos,
            voxel_size=distanceTreshold
        )
        if logProcess: print("Invisible points detected")

    # Step 5: Filter newGeometry to only new/changed points
    _, newNewGeo = filter_pcd_by_distance(newGeometry, relevantOg, distanceTreshold)
    if logProcess: print("New points filtered")

    # Step 6: Combine
    newCombinedGeometry = irrelevantOg + coveredPoints  + newNewGeo + (invisibleUncoveredPoints if checkVisibility else o3d.geometry.PointCloud())
    if logProcess: print("Geometries combined")

    return newCombinedGeometry


def mesh_to_pcd(mesh: o3d.geometry.TriangleMesh, voxelSize: float = 0.1) -> o3d.geometry.PointCloud:
    """Sample a point cloud on a triangle mesh then voxel downsample.

    Args:
        mesh      : source TriangleMesh
        voxelSize : spatial resolution of the output point cloud

    Returns:
        o3d.geometry.PointCloud
    """
    k = round(mesh.get_surface_area() * 1000)
    pcd = mesh.sample_points_uniformly(number_of_points=k, use_triangle_normal=True)
    pcd = pcd.voxel_down_sample(voxelSize)
    return pcd


def get_points_in_hull(geometry: o3d.geometry.PointCloud,
                        hull: o3d.geometry.TriangleMesh) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """Separates a point cloud into points inside and outside a convex hull.

    Args:
        geometry : point cloud to filter
        hull     : convex hull mesh to test against

    Returns:
        Tuple: (points inside hull, points outside hull)
    """
    hull_verts = np.asarray(hull.vertices)
    points     = np.asarray(geometry.points)

    delaunay   = Delaunay(hull_verts)
    inside_mask = delaunay.find_simplex(points) >= 0
    inside_idx  = np.where(inside_mask)[0].tolist()
    outside_idx = np.where(~inside_mask)[0].tolist()

    return geometry.select_by_index(inside_idx), geometry.select_by_index(outside_idx)


def filter_pcd_by_distance(sourcePcd: o3d.geometry.PointCloud,
                            testPcd: o3d.geometry.PointCloud,
                            maxDistance: float) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """Splits sourcePcd into points closer than / farther than maxDistance from testPcd.

    Args:
        sourcePcd   : point cloud to split
        testPcd     : point cloud to measure distance against
        maxDistance : distance threshold

    Returns:
        Tuple: (points within threshold, points beyond threshold)
    """
    dists      = np.asarray(sourcePcd.compute_point_cloud_distance(testPcd))
    inside_idx = np.where(dists < maxDistance)[0]
    return sourcePcd.select_by_index(inside_idx), sourcePcd.select_by_index(inside_idx, invert=True)

def build_occlusion_grid(
    reference: o3d.geometry.PointCloud,
    voxel_size: float = 0.05
) -> tuple:
    # --- Step 1: Translate point cloud so scanner is at origin ---
    pts_world    = np.asarray(reference.points)
    pts_centered = pts_world

    # --- Step 2: OBB local frame of the shifted cloud ---
    shifted_pcd = o3d.geometry.PointCloud()
    shifted_pcd.points = o3d.utility.Vector3dVector(pts_centered)
    obb    = shifted_pcd.get_oriented_bounding_box()
    R      = np.asarray(obb.R)
    center = np.asarray(obb.center)

    pts_local = (pts_centered - center) @ R

    # --- Step 3: Voxel indices ---
    min_bound_local = pts_local.min(axis=0)
    max_bound_local = pts_local.max(axis=0)
    grid_size       = np.floor((max_bound_local - min_bound_local) / voxel_size).astype(int) + 1

    voxel_indices   = np.floor((pts_local - min_bound_local) / voxel_size).astype(int)
    occupied_voxels = set(map(tuple, voxel_indices))

    # Scanner is at origin in shifted space — transform (0,0,0) into local frame
    origin_local = (np.zeros(3) - center) @ R
    origin_voxel = (origin_local - min_bound_local) / voxel_size

    # --- Step 4: Ray march from origin through each occupied voxel ---
    occluded_voxels = set()

    for voxel in occupied_voxels:
        voxel_f    = np.array(voxel, dtype=float) + 0.5
        ray_dir    = voxel_f - origin_voxel
        ray_length = np.linalg.norm(ray_dir)
        if ray_length == 0:
            continue
        ray_dir_n = ray_dir / ray_length

        t     = ray_length + 1.0
        t_max = ray_length + np.linalg.norm(grid_size)

        while t < t_max:
            current = np.floor(origin_voxel + t * ray_dir_n).astype(int)
            if np.any(current < 0) or np.any(current >= grid_size):
                break
            current_tuple = tuple(current)
            if current_tuple not in occupied_voxels:
                occluded_voxels.add(current_tuple)
            t += 1.0

    return occluded_voxels, occupied_voxels, obb, R, min_bound_local


def get_invisible_points_grid(
    points: o3d.geometry.PointCloud,
    reference: o3d.geometry.PointCloud,
    voxel_size: float = 0.05
) -> o3d.geometry.PointCloud:
    occluded_voxels, occupied_voxels, obb, R, min_bound_local = build_occlusion_grid(
        reference, voxel_size
    )

    center = np.asarray(obb.center)
    pts    = np.asarray(points.points)

    # Apply the same shift before transforming into local frame
    pts_local  = (pts - center) @ R
    voxel_idxs = [tuple(np.floor((p - min_bound_local) / voxel_size).astype(int)) for p in pts_local]

    invisible_mask = np.array([
        idx in occluded_voxels or idx in occupied_voxels
        for idx in voxel_idxs
    ])

    return points.select_by_index(np.where(invisible_mask)[0])


def visualise_occlusion_grid(
    occluded_voxels: set,
    occupied_voxels: set,
    obb: o3d.geometry.OrientedBoundingBox,
    R: np.ndarray,
    min_bound_local: np.ndarray,
    voxel_size: float = 0.05,
    show_occupied: bool = True,
    show_occluded: bool = True,
) -> trimesh.Scene:
    center = np.asarray(obb.center)

    unit_cube  = trimesh.creation.box(extents=[1, 1, 1])
    unit_verts = np.asarray(unit_cube.vertices)
    faces      = np.asarray(unit_cube.faces)

    def voxels_to_mesh(voxel_set: set, color: list) -> trimesh.Trimesh:
        if not voxel_set:
            return None
        indices       = np.array(list(voxel_set))
        centres_local = (indices + 0.5) * voxel_size + min_bound_local
        centres_world = centres_local @ R.T + center

        K         = len(indices)
        all_verts = (unit_verts[np.newaxis] * voxel_size + centres_world[:, np.newaxis, :]).reshape(-1, 3)
        offsets   = np.arange(K)[:, np.newaxis, np.newaxis] * 8
        all_faces = (faces[np.newaxis] + offsets).reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
        mesh.visual.face_colors = np.tile(
            np.array([int(c) for c in color], dtype=np.uint8),
            (len(all_faces), 1)
        )
        return mesh

    scene = trimesh.Scene()
    if show_occupied:
        m = voxels_to_mesh(occupied_voxels, color=[220, 60, 60, 180])
        if m is not None:
            scene.add_geometry(m, node_name="occupied")
    if show_occluded:
        m = voxels_to_mesh(occluded_voxels, color=[60, 120, 220, 120])
        if m is not None:
            scene.add_geometry(m, node_name="occluded")

    return scene


def is_occluded(
    point: np.ndarray,
    occluded_voxels: set,
    occupied_voxels: set,
    obb: o3d.geometry.OrientedBoundingBox,
    R: np.ndarray,
    origin_local: np.ndarray,
    min_bound_local: np.ndarray,
    voxel_size: float = 0.05
) -> bool:
    """
    Checks if a single world-space point falls in an occluded voxel.

    Args:
        point           : 1x3 world-space point to test
        occluded_voxels : set of occluded voxel indices from build_occlusion_grid
        occupied_voxels : set of occupied voxel indices from build_occlusion_grid
        obb             : oriented bounding box from build_occlusion_grid
        R               : rotation matrix from build_occlusion_grid
        origin_local    : scanner position in local frame
        min_bound_local : min bound in local frame
        voxel_size      : must match the value used in build_occlusion_grid

    Returns:
        bool: True if occluded (invisible to scanner), False if visible
    """
    center      = np.asarray(obb.center)
    pt_local    = (point - center) @ R
    voxel_idx   = tuple(np.floor((pt_local - min_bound_local) / voxel_size).astype(int))

    return voxel_idx in occluded_voxels or voxel_idx in occupied_voxels


def get_invisible_points(
    points: o3d.geometry.PointCloud,
    reference: o3d.geometry.PointCloud,
    scanner_position: np.ndarray,
    occlusion_radius: float = 0.05
) -> o3d.geometry.PointCloud:
    """Returns only the points that could NOT have been scanned from the scanner position.

    For each point, casts a ray toward the scanner and checks if any point in the
    reference cloud blocks it. If blocked, the point was occluded and is kept.
    If not blocked, the scanner could have seen it → it is out of date → discarded.

    Args:
        points           : uncovered points to test (with normals)
        reference        : the new point cloud acting as the scene (with normals)
        scanner_position : 1x3 position of the new scanner in world space
        occlusion_radius : how close a reference point needs to be to the ray
                           to count as blocking it (in metres, should match voxel size)

    Returns:
        o3d.geometry.PointCloud: only the occluded (invisible) points
    """
    pts           = np.asarray(points.points)       # N x 3
    ref_pts       = np.asarray(reference.points)    # M x 3
    kdtree        = o3d.geometry.KDTreeFlann(reference)
    invisible_mask = np.zeros(len(pts), dtype=bool)

    for i, pt in enumerate(pts):
        ray_dir    = scanner_position - pt           # vector from point to scanner
        ray_length = np.linalg.norm(ray_dir)
        ray_dir_n  = ray_dir / ray_length            # normalised

        # Search for reference points within occlusion_radius of the ray,
        # within the ray segment (not behind the point or beyond the scanner)
        _, idx, _ = kdtree.search_radius_vector_3d(pt, ray_length)

        occluded = False
        for j in idx:
            candidate = ref_pts[j]

            # Project candidate onto ray, skip if behind origin or past scanner
            t = np.dot(candidate - pt, ray_dir_n)
            if t < occlusion_radius or t > ray_length - occlusion_radius:
                continue

            # Check perpendicular distance from candidate to ray
            closest_on_ray  = pt + t * ray_dir_n
            perp_dist       = np.linalg.norm(candidate - closest_on_ray)

            if perp_dist < occlusion_radius:
                occluded = True
                break

        invisible_mask[i] = occluded

    return points.select_by_index(np.where(invisible_mask)[0]), points.select_by_index(np.where(invisible_mask)[0], invert=True)

def filter_geometry_by_distance(geometries: List[o3d.geometry.Geometry],
                                 query_point: np.ndarray,
                                 distance_threshold: float = 500) -> List[o3d.geometry.Geometry]:
    """Filters out parts of geometries that lie too far from a query point.

    Args:
        geometries         : list of PointCloud, TriangleMesh, or LineSet objects
        query_point        : 1×3 center point of the search
        distance_threshold : maximum allowed distance

    Returns:
        List[o3d.geometry.Geometry]: filtered geometries
    """
    result = []

    for g in geometries:
        g = g  # no deepcopy per geometry — caller should copy if needed

        if isinstance(g, o3d.geometry.LineSet):
            points = np.asarray(g.points)
            dist   = np.linalg.norm(points - query_point, axis=1)
            remove_idx = np.where(dist > distance_threshold)[0]

            if remove_idx.size == 0 or remove_idx.size == points.shape[0]:
                if remove_idx.size == 0:
                    result.append(g)
                continue

            g.points = o3d.utility.Vector3dVector(np.delete(points, remove_idx, axis=0))
            line_remove = np.where(np.any(np.isin(np.asarray(g.lines), remove_idx), axis=1))[0]
            if line_remove.size > 0:
                g.lines = o3d.utility.Vector2iVector(np.delete(np.asarray(g.lines), line_remove, axis=0))
            result.append(g)

        elif isinstance(g, o3d.geometry.PointCloud):
            points = np.asarray(g.points)
            dist   = np.linalg.norm(points - query_point, axis=1)
            idx    = np.where(dist <= distance_threshold)[0]

            if idx.size == 0:
                continue
            result.append(g.select_by_index(idx))

        elif isinstance(g, o3d.geometry.TriangleMesh):
            points = np.asarray(g.vertices)
            dist   = np.linalg.norm(points - query_point, axis=1)
            remove_idx = np.where(dist > distance_threshold)[0]

            if remove_idx.size == 0:
                result.append(g)
                continue

            tri_remove = np.where(np.any(np.isin(np.asarray(g.triangles), remove_idx), axis=1))[0]
            if tri_remove.size > 0:
                g.remove_triangles_by_index(tri_remove)
                g.remove_unreferenced_vertices()
            result.append(g)

    return result