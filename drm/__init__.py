import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import trimesh
import random
from scipy.spatial import ConvexHull, Delaunay,cKDTree
import open3d as o3d
import copy
from typing import Union, List

UNITY2TRIMESH_T = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

# Converts Unity coordinate system to open3d (swap Y and Z)
UNITY_TO_OPEN3D = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float64)

def transform_points(pts_3d: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 homogeneous transform to an (N,3) array."""
    ones = np.ones((len(pts_3d), 1), dtype=np.float64)
    pts_h = np.hstack([pts_3d, ones])          # (N, 4)
    return (T @ pts_h.T).T[:, :3]              # (N, 3)



def o3d_mesh_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_normals=np.asarray(mesh.vertex_normals),
        vertex_colors=np.asarray(mesh.vertex_colors)
    )

def o3d_pointcloud_to_trimesh(pcd: o3d.geometry.PointCloud) -> trimesh.Trimesh:
    
    return trimesh.PointCloud(
        vertices=np.asarray(pcd.points),
        vertex_colors=np.asarray(pcd.colors) if pcd.has_colors() else None
    )

def lineset_to_trimesh_path(lineset: o3d.geometry.LineSet) -> trimesh.path.Path3D:
    """
    Convert an open3d LineSet to a trimesh Path3D for rendering.

    Args:
        lineset : open3d.geometry.LineSet

    Returns:
        trimesh.path.Path3D
    """
    points = np.asarray(lineset.points)
    lines  = np.asarray(lineset.lines)   # Nx2 indices into points

    # Build trimesh line entities
    entities = [trimesh.path.entities.Line(line) for line in lines]

    path = trimesh.path.Path3D(entities=entities, vertices=points, colors=[[255, 0, 0, 255]] * len(entities))
    return path

def transform_xyz_to_uv(points, tMat):
    """
    Apply a 4x4 transform matrix to 3D points and convert to equirectangular UV.

    Args:
        points: numpy array of shape (N,3) where each row is [x, y, z]
        T: 4x4 transformation matrix

    Returns:
        uv: numpy array of shape (N,2) with [u,v] coordinates in [0,1]
    """
    N = points.shape[0]

    # Convert to homogeneous coordinates (N,4)
    points_h = np.hstack([points, np.ones((N, 1))])

    # Apply transformation (matrix multiplication)
    points_cam_h = (tMat @ points_h.T).T#(np.linalg.inv(tMat) @ points_h.T).T  # shape (N,4)

    # Convert back to 3D (ignore w, assuming affine transform)
    points_cam = points_cam_h[:, :3] #- np.repeat([[-tMat[2,3],tMat[1,3],tMat[0,3]]],N,axis = 0) # np.repeat([[tMat[0,3],tMat[1,3],tMat[2,3]]],N,axis = 0)

    # --- Convert transformed points to equirectangular UV ---
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    r = np.linalg.norm(points_cam, axis=1)
    r[r == 0] = 1  # avoid division by zero

    theta = np.arccos(y / r)    # polar angle
    phi = np.arctan2(z, x)      # azimuthal angle

    u = (phi + np.pi) / (2 * np.pi)
    v = theta / np.pi

    # Handle origin points safely
    zero_mask = r == 0
    u[zero_mask] = 0.5
    v[zero_mask] = 0.5

    return np.stack([1-u, v], axis=1)

def detect_objects(binPath, model="votenet",scoreThr = 0.3, outputDir="../_output"):
    demoFile = "../../mmdetection3d/demo/pcd_demo.py"
    if(model == "votenet"):
        configFile = "/home/jvermandere/projects/mmdetection3d/configs/votenet/votenet_8xb8_scannet-3d.py"
        weightsFile = "/home/jvermandere/projects/DRM/_weights/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth"
    elif(model == "votenet-sunrgb"):
        configFile = "../../mmdetection3d/configs/votenet/votenet_8xb16_sunrgbd-3d.py"
        weightsFile = "/home/jvermandere/projects/DRM/_weights/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth"
    elif (model == "tr3d"):
        configFile = "/home/jvermandere/projects/mmdetection3d/projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py"
        weightsFile = "/home/jvermandere/projects/DRM/_weights/tr3d_1xb16_scannet-3d-18class.pth"



    command = f'python {demoFile} "{binPath}" {configFile} {weightsFile} --pred-score-thr {scoreThr} --out-dir {outputDir}'
    print("running command: " + command)

    os.system(command)

    jsonFile = Path(Path(outputDir).absolute() / "preds" / Path(binPath).stem).with_suffix(".json")
    return jsonFile

def set_axes_equal(ax, pad=0.0):
    """
    Make a 3D plot have equal scale on all axes so cubes appear as cubes.
    pad : fraction of half-range to add as padding (can be negative to zoom in)
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_center = sum(x_limits) / 2
    y_center = sum(y_limits) / 2
    z_center = sum(z_limits) / 2

    max_range = max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0],
    ) / 2

    max_range *= (1 + pad)

    ax.set_xlim3d(x_center - max_range, x_center + max_range)
    ax.set_ylim3d(y_center - max_range, y_center + max_range)
    ax.set_zlim3d(z_center - max_range, z_center + max_range)


def plot_points_3d(points, colors=None, up_axis='z', size=20, cmap='viridis', zoom=0.15):
    points = np.asarray(points)

    axis_order = {
        'x': (1, 2, 0),
        'y': (0, 2, 1),
        'z': (0, 1, 2),
    }[up_axis]

    pts = points[:, axis_order]

    # Default: color by normalized XYZ
    if colors is None:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        denom = np.where(maxs > mins, maxs - mins, 1.0)
        colors = (pts - mins) / denom
        use_cmap = False
    else:
        use_cmap = colors.ndim == 1

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    sc = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=colors,
        s=size,
        cmap=cmap if use_cmap else None
    )

    # Remove all axis visuals
    ax.set_axis_off()

    # Zoom in + preserve cube geometry
    set_axes_equal(ax, pad=-zoom)
    ax.set_box_aspect((1, 1, 1))

    if use_cmap:
        plt.colorbar(sc, ax=ax, shrink=0.6)

    plt.show()

def create_trimesh_box(bottom_center, size, rotation_z=0.0, color=[1,0,0,0.5]):
    """
    Create a trimesh Box mesh with given center, size, rotation, and color.
    """
    # Box is created centered at origin
    box = trimesh.creation.box(extents=size, transform=None)
    
    # Rotation matrix around z-axis
    c, s = np.cos(rotation_z), np.sin(rotation_z)
    R = np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])
    
    # Translation to center
    T = np.eye(4)
    T[:3, 3] = [
    bottom_center[0],
    bottom_center[1],
    bottom_center[2] + size[2] / 2.0
    ]

    # Apply transform
    box.apply_transform(T @ R)
    
    # Set color (RGBA)
    box.visual.face_colors = color
    
    return box

def load_bin_pointcloud(file_path):
    """Load KITTI-style .bin point cloud"""
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 6)  # x, y, z, rgb
    cloud = trimesh.points.PointCloud(points[:, :3], colors=points[:, 3:6]/255)
    return cloud

def txt_pcd_to_ply(txt_path, ply_path):
    with open(txt_path, "r") as f:
        rows = [line.strip().split() for line in f if line.strip()]

    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if len(rows[0]) == 9:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for r in rows:
            f.write(" ".join(r) + "\n")


def txt_pcd_to_bin(txtPath, binPath = "", rotateX = False, containsNormals = False, mirrorX = False,mirrorY = False,mirrorZ = False):
    #if binPath is empty, save at the same location
    if(binPath == ""):
        binPath = (str)(Path(txtPath).with_suffix(".bin"))
    # Load txt file
    # Assuming format: x y z nx ny nz r g b
    data = np.loadtxt(txtPath, comments="#")

    # Keep only x, y, z, r, g, b for VoteNet/ScanNet format
    if(containsNormals):
        points = data[:, [0, 1, 2, 6, 7, 8]]
    else:
        points = data[:, [0, 1, 2, 3, 4, 5]]
    if(rotateX):
        xyz = points[:, :3]
        # Rotation matrix for 90 deg around X
        R = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        xyz_rot = xyz @ R.T
        points[:, :3] = xyz_rot
    if(mirrorX):
        points[:, 0] *= -1
    if(mirrorY):
        points[:, 1] *= -1
    if(mirrorZ):
        points[:, 2] *= -1
    # Convert RGB from 0-255 to 0-1
    #points[:, 3:6] /= 255.0

    # Ensure float32 type
    points = points.astype(np.float32)

    # Save to .bin
    points.tofile(binPath)

    print(f"Saved {points.shape[0]} points to {binPath}")
    return binPath

def bin_to_txt(bin_path, txt_path = ""):
    """
    Convert VoteNet .bin (x y z r g b in 0-1) back to txt format (x y z r g b),
    restoring RGB to 0-255.
    """
    if(txt_path == ""):
        txt_path = (str)(Path(bin_path).with_suffix(".txt"))
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 6)
    #points[:, 3:6] *= 255.0                # Restore RGB to 0-255
    points = points.astype(np.float32)
    # Save as txt (x y z r g b)
    np.savetxt(txt_path, points, fmt='%.6f')
    print(f"Saved {points.shape[0]} points to {txt_path} (RGB restored to 0-255)")
    return txt_path

def txt_pcd_to_trimesh(txtPath, rotateX=False, mirrorX=False, mirrorY=False, mirrorZ=False):
    """
    Reads a point cloud .txt file and converts it to a trimesh.PointCloud.
    
    Expected format (with header comments):
        # x y z nx ny nz r g b
        x y z nx ny nz r g b
    
    Returns:
        cloud  (trimesh.PointCloud) — with vertex colors set
        normals (np.ndarray, shape N×3) — per-point normals (trimesh doesn't
                 store these on PointCloud natively, so returned separately)
    """
    data = np.loadtxt(txtPath, comments="#")

    # Parse columns: x y z nx ny nz r g b
    xyz     = data[:, 0:3]
    normals = data[:, 3:6]
    rgb     = data[:, 6:9]          # uint8 0-255

    # --- Optional transforms ---
    if rotateX:
        R = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ])
        xyz     = xyz     @ R.T
        normals = normals @ R.T     # rotate normals the same way

    if mirrorX:
        xyz[:, 0]     *= -1
        normals[:, 0] *= -1
    if mirrorY:
        xyz[:, 1]     *= -1
        normals[:, 1] *= -1
    if mirrorZ:
        xyz[:, 2]     *= -1
        normals[:, 2] *= -1

    # Build RGBA colors (trimesh wants uint8 with alpha)
    alpha  = np.full((len(rgb), 1), 255, dtype=np.uint8)
    colors = np.hstack([rgb.astype(np.uint8), alpha])   # N×4

    # Create trimesh PointCloud
    cloud = trimesh.PointCloud(vertices=xyz.astype(np.float32), colors=colors)

    # Normalise normals and attach as metadata
    # (trimesh.PointCloud has no dedicated normals field, so we attach them
    #  as metadata and also return them for convenience)
    norms_length = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_length[norms_length == 0] = 1          # avoid div-by-zero
    normals_unit = (normals / norms_length).astype(np.float32)

    cloud.metadata["normals"] = normals_unit     # retrievable later

    print(f"Loaded {len(xyz)} points from {txtPath}")
    return cloud, normals_unit


def read_transform_matrix(txtPath, apply_unity_conversion=False):
    """
    Reads the 4x4 transform matrix from a point cloud .txt header.

    Args:
        txtPath               : path to the .txt point cloud file
        apply_unity_conversion: if True, premultiplies with UNITY_TO_OPEN3D
        save_matrix           : if True, saves the matrix as a .npy next to the input

    Returns:
        matrix (np.ndarray, shape 4×4, float64)
    """
    with open(txtPath, "r") as f:
        lines = f.readlines()

    matrix_start = None
    for i, line in enumerate(lines):
        if line.startswith("# transform_matrix"):
            matrix_start = i + 1
            break
    if matrix_start is None:
        raise ValueError(f"Transform matrix not found in file: {txtPath}")

    matrix = np.fromstring(lines[matrix_start][2:], dtype=np.float64, sep=' ').reshape((4, 4))

    if apply_unity_conversion:
        matrix = UNITY_TO_OPEN3D @ matrix

    return matrix


def txt_pcd_to_open3d(txtPath, apply_transform=False, inverse_transform=True,
                       apply_unity_conversion=True):
    """
    Reads a point cloud .txt file and converts it to an open3d.geometry.PointCloud.

    Args:
        txtPath               : path to the .txt point cloud file
        rotateX               : rotate 90° around X axis
        mirrorX/Y/Z           : flip along respective axis
        apply_transform       : if True, reads and applies the header transform matrix
        inverse_transform     : if True, applies the inverse of the transform instead
        apply_unity_conversion: if True, premultiplies transform with UNITY_TO_OPEN3D
        save_matrix           : if True, saves the transform matrix as a .npy file

    Returns:
        pcd    (open3d.geometry.PointCloud)
        matrix (np.ndarray 4×4) or None if apply_transform is False
    """
    data = np.loadtxt(txtPath, comments="#")

    xyz     = data[:, 0:3]
    normals = data[:, 3:6]
    rgb     = data[:, 6:9] / 255.0


    # Normalise normals
    norms_length = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_length[norms_length == 0] = 1
    normals = normals / norms_length

    # --- Build open3d PointCloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    pcd.colors  = o3d.utility.Vector3dVector(rgb.astype(np.float64))

    # --- Optionally apply header transform ---
    matrix = None
    if apply_transform:
        matrix = read_transform_matrix(txtPath,
                                       apply_unity_conversion=False)
        T = np.linalg.inv(matrix) if inverse_transform else matrix
        pcd.transform(T)
    if apply_unity_conversion:
        pcd.transform(UNITY_TO_OPEN3D)

    print(f"Loaded {len(pcd.points)} points from {txtPath}")
    return pcd, matrix


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

def visualize_pointclouds_random_colors(planes):
    """
    Visualize extracted planes with unique colors, plus leftover points in gray.
    
    Parameters:
        planes (list of trimesh.points.PointCloud): Detected planes.
    """
    scene = trimesh.Scene()
    
    # Assign a random color to each plane and add to the scene
    for plane in planes:
        new_plane = plane.copy()
        color = np.array([random.random(), random.random(), random.random(),1]) * 255
        if new_plane.colors is None:
            new_plane.colors = np.tile(color, (len(new_plane.vertices), 1))
        else:
            # Optionally blend existing colors with random tint
            new_plane.colors = (new_plane.colors.astype(float) * 0.5 + color * 0.5).astype(np.uint8)
        scene.add_geometry(new_plane)
    
    # return the scene
    return scene

# ── Type alias for anything this function accepts ─────────────────────────────
O3DGeometry = Union[
    o3d.geometry.PointCloud,
    o3d.geometry.TriangleMesh,
    o3d.geometry.LineSet,
    o3d.geometry.VoxelGrid,
    o3d.geometry.AxisAlignedBoundingBox,
    o3d.geometry.OrientedBoundingBox,
]
 
 
def _random_rgba() -> np.ndarray:
    """Return a random opaque RGBA color as uint8 (1, 4)."""
    return (np.array([random.random(), random.random(), random.random(), 1.0]) * 255).astype(np.uint8)
 
 
# ── Per-type converters ───────────────────────────────────────────────────────
 
def _convert_pointcloud(pcd: o3d.geometry.PointCloud, random_color: bool):
    xyz = np.asarray(pcd.points).astype(np.float32)
    if random_color or not pcd.has_colors():
        rgba = np.tile(_random_rgba(), (len(xyz), 1))
    else:
        rgb  = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        rgba = np.hstack([rgb, np.full((len(rgb), 1), 255, dtype=np.uint8)])
    return trimesh.PointCloud(vertices=xyz, colors=rgba)
 
 
def _convert_trianglemesh(mesh: o3d.geometry.TriangleMesh, random_color: bool):
    verts = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles)
 
    if random_color:
        color = _random_rgba()
        vertex_colors = np.tile(color, (len(verts), 1))
    elif mesh.has_vertex_colors():
        rgb   = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
        vertex_colors = np.hstack([rgb, np.full((len(rgb), 1), 255, dtype=np.uint8)])
    else:
        vertex_colors = None
 
    kwargs = dict(vertices=verts, faces=faces)
    if vertex_colors is not None:
        kwargs["vertex_colors"] = vertex_colors
 
    tm = trimesh.Trimesh(**kwargs)
 
    if mesh.has_vertex_normals():
        tm.vertex_normals = np.asarray(mesh.vertex_normals).astype(np.float32)
 
    return tm
 
 
def _convert_lineset(ls: o3d.geometry.LineSet, random_color: bool):
    pts   = np.asarray(ls.points)
    lines = np.asarray(ls.lines)          # (M, 2) int indices
 
    if len(pts) == 0 or len(lines) == 0:
        return None
 
    if random_color or not ls.has_colors():
        fallback = _random_rgba()
        entities = [
            trimesh.path.entities.Line(seg, color=fallback)
            for seg in lines
        ]
    else:
        line_rgb  = (np.asarray(ls.colors) * 255).astype(np.uint8)
        line_rgba = np.hstack([line_rgb, np.full((len(line_rgb), 1), 255, dtype=np.uint8)])
        entities  = [
            trimesh.path.entities.Line(seg, color=line_rgba[i])
            for i, seg in enumerate(lines)
        ]
 
    return trimesh.path.Path3D(entities=entities, vertices=pts)
 
 
def _convert_voxelgrid(vg: o3d.geometry.VoxelGrid, random_color: bool):
    """
    Represent each voxel as a small trimesh Box.
    All boxes are merged into a single Trimesh for efficiency.
    """
    voxels = vg.get_voxels()
    if not voxels:
        return None
 
    s    = vg.voxel_size
    ext  = np.array([s, s, s])
    meshes = []
 
    fixed_color = _random_rgba() if random_color else None
 
    for v in voxels:
        center = vg.get_voxel_center_coordinate(v.grid_index)
        box    = trimesh.creation.box(extents=ext)
        box.apply_translation(center)
 
        if fixed_color is not None:
            color = fixed_color
        else:
            rgb   = (np.asarray(v.color) * 255).astype(np.uint8)
            color = np.append(rgb, 255).astype(np.uint8)
 
        box.visual.face_colors = np.tile(color, (len(box.faces), 1))
        meshes.append(box)
 
    return trimesh.util.concatenate(meshes)
 
 
def _bbox_to_path3d(pts_8: np.ndarray, color_rgba: np.ndarray) -> trimesh.path.Path3D:
    """
    Build a trimesh Path3D wireframe from 8 corner points of a bounding box.
    open3d get_box_points() returns corners in a fixed order — the 12 edges
    are hard-coded to match that order.
    """
    # Edge pairs for an OBB / AABB returned by o3d.get_box_points()
    edges = [
        [0, 1], [0, 2], [0, 3],
        [1, 6], [1, 7],
        [2, 5], [2, 7],
        [3, 5], [3, 6],
        [4, 5], [4, 6], [4, 7],
    ]
    entities = [trimesh.path.entities.Line(e, color=color_rgba) for e in edges]
    return trimesh.path.Path3D(entities=entities, vertices=pts_8)
 
 
def _convert_aabb(aabb: o3d.geometry.AxisAlignedBoundingBox, random_color: bool):
    pts   = np.asarray(aabb.get_box_points())
    color = _random_rgba() if random_color else np.array([*((np.asarray(aabb.color) * 255).astype(np.uint8)), 255], dtype=np.uint8)
    return _bbox_to_path3d(pts, color)
 
 
def _convert_obb(obb: o3d.geometry.OrientedBoundingBox, random_color: bool):
    pts   = np.asarray(obb.get_box_points())
    color = _random_rgba() if random_color else np.array([*((np.asarray(obb.color) * 255).astype(np.uint8)), 255], dtype=np.uint8)
    return _bbox_to_path3d(pts, color)
 
 
# ── Dispatch table ─────────────────────────────────────────────────────────────
 
_CONVERTERS = {
    o3d.geometry.PointCloud:              _convert_pointcloud,
    o3d.geometry.TriangleMesh:            _convert_trianglemesh,
    o3d.geometry.LineSet:                 _convert_lineset,
    o3d.geometry.VoxelGrid:              _convert_voxelgrid,
    o3d.geometry.AxisAlignedBoundingBox: _convert_aabb,
    o3d.geometry.OrientedBoundingBox:    _convert_obb,
}
 
 
# ── Public API ─────────────────────────────────────────────────────────────────
 
def visualise_open3d(
    geometries: Union[O3DGeometry, List[O3DGeometry]],
    random_color: bool = False,
) -> trimesh.Scene:
    """
    Convert one or more Open3D geometry objects into a trimesh Scene.
 
    Supported types
    ---------------
    - PointCloud           → trimesh.PointCloud
    - TriangleMesh         → trimesh.Trimesh  (vertex colors + normals preserved)
    - LineSet              → trimesh.path.Path3D  (per-segment colors preserved)
    - VoxelGrid            → trimesh.Trimesh  (one box per voxel, merged)
    - AxisAlignedBoundingBox → trimesh.path.Path3D  (wireframe)
    - OrientedBoundingBox  → trimesh.path.Path3D  (wireframe)
 
    Parameters
    ----------
    geometries   : single geometry or list/tuple of geometries (may be mixed types)
    random_color : if True, override all colors with a random color per object
 
    Returns
    -------
    trimesh.Scene
    """
    if not isinstance(geometries, (list, tuple)):
        geometries = [geometries]
 
    scene   = trimesh.Scene()
    counts  = {}   # track per-type index for unique node names
 
    for geom in geometries:
        geom_type = type(geom)
        converter = _CONVERTERS.get(geom_type)
 
        if converter is None:
            print(f"[visualise_open3d] Unsupported type {geom_type.__name__}, skipping.")
            continue
 
        result = converter(geom, random_color)
        if result is None:
            continue
 
        # Build a unique node name: e.g. "PointCloud_0", "LineSet_2"
        label = geom_type.__name__
        idx   = counts.get(label, 0)
        counts[label] = idx + 1
 
        scene.add_geometry(result, node_name=f"{label}_{idx}")
 
    return scene
 

def randomly_transform_pointcloud(
    pcd: o3d.geometry.PointCloud,
    translation_bounds,
    rotate=True,
    up_axis="z",
    rotation_bounds_deg=(-180.0, 180.0),
    seed=None,
):
    """
    Randomly translate an Open3D pointcloud within bounds and optionally rotate
    around a defined up axis.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Input point cloud.
        
    translation_bounds : dict or tuple
        Either:
        {
            "x": (min_x, max_x),
            "y": (min_y, max_y),
            "z": (min_z, max_z)
        }
        OR
        ((min_x,max_x), (min_y,max_y), (min_z,max_z))
        
    rotate : bool
        If True, applies random rotation around up axis.
        
    up_axis : str
        One of: "x", "y", "z"
        
    rotation_bounds_deg : tuple
        (min_deg, max_deg) rotation range.
        
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    transformed_pcd : open3d.geometry.PointCloud
        New transformed point cloud.
        
    transform : np.ndarray shape (4,4)
        Applied homogeneous transform matrix.
        
    info : dict
        Random translation and rotation values used.
    """

    rng = np.random.default_rng(seed)

    # Parse translation bounds
    if isinstance(translation_bounds, dict):
        bx = translation_bounds["x"]
        by = translation_bounds["y"]
        bz = translation_bounds["z"]
    else:
        bx, by, bz = translation_bounds

    tx = rng.uniform(*bx)
    ty = rng.uniform(*by)
    tz = rng.uniform(*bz)

    angle_deg = 0.0
    angle_rad = 0.0

    if rotate:
        angle_deg = rng.uniform(*rotation_bounds_deg)
        angle_rad = np.deg2rad(angle_deg)

    # Rotation matrix
    if up_axis.lower() == "x":
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad),  np.cos(angle_rad)]
        ])
    elif up_axis.lower() == "y":
        R = np.array([
            [ np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif up_axis.lower() == "z":
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("up_axis must be 'x', 'y', or 'z'")

    transformed = copy.deepcopy(pcd)

    # Full transform matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]

    transformed.transform(T)  # apply full transform (rotation + translation)

    return transformed, T

def radial_fov_crop_pointcloud(
    pcd: o3d.geometry.PointCloud,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    horizontal_center_deg=None,
    vertical_center_deg: float = 0.0,
    origin=(0.0, 0.0, 0.0),
    seed=None,
    keep_normals=True,
    keep_colors=True,
):
    """
    Crop an Open3D point cloud by angular field-of-view from a fixed origin,
    simulating a partial scan with a smaller sensor FOV.

    The crop is defined in spherical angles around the origin:
    - horizontal angle: azimuth around Z, in degrees
    - vertical angle: elevation above horizon, in degrees

    The horizontal center is random if not provided.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input point cloud.

    horizontal_fov_deg : float
        Width of the horizontal field of view in degrees.

    vertical_fov_deg : float
        Height of the vertical field of view in degrees.

    horizontal_center_deg : float or None
        Center azimuth in degrees. If None, sampled uniformly from [-180, 180).

    vertical_center_deg : float
        Center elevation in degrees.

    origin : tuple[float, float, float]
        Sensor origin from which angles are measured.

    seed : int or None
        Random seed for reproducibility.

    keep_normals : bool
        Preserve normals if present.

    keep_colors : bool
        Preserve colors if present.

    Returns
    -------
    cropped_pcd : o3d.geometry.PointCloud
        Cropped partial scan.

    mask : np.ndarray
        Boolean mask over original points indicating which were kept.

    info : dict
        Metadata about the chosen crop window.
    """
    if horizontal_fov_deg <= 0 or horizontal_fov_deg > 360:
        raise ValueError("horizontal_fov_deg must be in (0, 360].")
    if vertical_fov_deg <= 0 or vertical_fov_deg > 180:
        raise ValueError("vertical_fov_deg must be in (0, 180].")

    rng = np.random.default_rng(seed)

    if horizontal_center_deg is None:
        horizontal_center_deg = rng.uniform(-180.0, 180.0)

    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return o3d.geometry.PointCloud(), np.array([], dtype=bool), {
            "horizontal_center_deg": horizontal_center_deg,
            "vertical_center_deg": vertical_center_deg,
            "horizontal_fov_deg": horizontal_fov_deg,
            "vertical_fov_deg": vertical_fov_deg,
            "num_kept": 0,
            "num_total": 0,
        }

    origin = np.asarray(origin, dtype=float)
    rel = pts - origin[None, :]

    x = rel[:, 0]
    y = rel[:, 1]
    z = rel[:, 2]

    r_xy = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)

    # Avoid divide-by-zero for points exactly at the origin
    valid = r > 1e-12

    # Horizontal angle (azimuth): [-180, 180]
    azimuth_deg = np.degrees(np.arctan2(y, x))

    # Vertical angle (elevation): [-90, 90]
    elevation_deg = np.degrees(np.arctan2(z, r_xy))

    def wrapped_angle_diff_deg(a, b):
        """Smallest signed angular difference a-b in degrees, wrapped to [-180, 180)."""
        return (a - b + 180.0) % 360.0 - 180.0

    half_h = horizontal_fov_deg / 2.0
    half_v = vertical_fov_deg / 2.0

    az_diff = wrapped_angle_diff_deg(azimuth_deg, horizontal_center_deg)
    el_diff = elevation_deg - vertical_center_deg

    mask = (
        valid
        & (np.abs(az_diff) <= half_h)
        & (np.abs(el_diff) <= half_v)
    )

    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])

    if keep_colors and pcd.has_colors():
        colors = np.asarray(pcd.colors)
        cropped.colors = o3d.utility.Vector3dVector(colors[mask])

    if keep_normals and pcd.has_normals():
        normals = np.asarray(pcd.normals)
        cropped.normals = o3d.utility.Vector3dVector(normals[mask])

    info = {
        "horizontal_center_deg": float(horizontal_center_deg),
        "vertical_center_deg": float(vertical_center_deg),
        "horizontal_fov_deg": float(horizontal_fov_deg),
        "vertical_fov_deg": float(vertical_fov_deg),
        "num_kept": int(mask.sum()),
        "num_total": int(len(mask)),
        "fraction_kept": float(mask.mean()) if len(mask) > 0 else 0.0,
    }

    return cropped, mask, info

def fill_plane_holes(plane_pc, target_density=0.01):
    """
    Fill holes in a planar point cloud within its convex hull.
    
    Parameters:
        plane_pc (trimesh.points.PointCloud): Original plane point cloud.
        target_density (float): Approximate distance between points in the filled grid.
    
    Returns:
        filled_pc (trimesh.points.PointCloud): Plane with added points to fill holes.
    """
    points = plane_pc.vertices
    colors = plane_pc.colors if plane_pc.colors is not None else None

    # 1. Compute the plane normal via PCA
    centroid = points.mean(axis=0)
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]  # smallest eigenvalue => normal direction

    # 2. Create 2D coordinates on plane
    # Choose two axes orthogonal to the normal
    u = eigvecs[:, 2]
    v = eigvecs[:, 1]

    # Project points onto 2D plane coordinates
    points_2d = np.dot(points - centroid, np.column_stack((u, v)))

    # 3. Compute 2D convex hull
    hull = ConvexHull(points_2d)
    delaunay = Delaunay(points_2d[hull.vertices])

    # 4. Create grid inside hull bounding box
    min_xy = points_2d.min(axis=0)
    max_xy = points_2d.max(axis=0)

    x_vals = np.arange(min_xy[0], max_xy[0], target_density)
    y_vals = np.arange(min_xy[1], max_xy[1], target_density)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # 5. Keep only points inside convex hull
    mask = delaunay.find_simplex(grid_points) >= 0
    valid_points_2d = grid_points[mask]

    # 6. Map 2D points back to 3D
    filled_points_3d = centroid + np.outer(valid_points_2d[:,0], u) + np.outer(valid_points_2d[:,1], v)

    # 7. Interpolate colors using nearest neighbor
    if colors is not None:
        kdtree = cKDTree(points)
        _, idx = kdtree.query(filled_points_3d)
        filled_colors = colors[idx]
    else:
        filled_colors = None

    filled_pc = trimesh.points.PointCloud(vertices=filled_points_3d, colors=filled_colors)
    return filled_pc


def project_planes_with_infill_mask(original_pc, filled_pc, resolution=512, point_radius=2):
    """
    Generate a 2D orthographic image of a plane using the filled point cloud colors,
    enlarged points for visual density, and a mask highlighting filled points.

    Parameters:
        original_pc (trimesh.points.PointCloud): Original plane points.
        filled_pc (trimesh.points.PointCloud): Plane after filling holes (with high-res colors).
        resolution (int): Output image resolution (HxW).
        point_radius (int): Radius in pixels to draw each point.

    Returns:
        img (np.ndarray): HxWx4 image with filled colors.
        mask (np.ndarray): HxW binary mask of filled points.
        filled_coords (np.ndarray): Nx2 pixel coordinates of filled points.
    """

    points = original_pc.vertices
    filled_points = filled_pc.vertices
    colors = filled_pc.colors.astype(np.uint8) if filled_pc.colors is not None else np.full((len(filled_points),3),255,dtype=np.uint8)

    # --------------------------
    # PCA for plane alignment
    # --------------------------
    centroid = points.mean(axis=0)
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    u = eigvecs[:,2]
    v = eigvecs[:,1]

    # Project points to 2D
    points_2d = np.dot(points - centroid, np.column_stack((u,v)))
    filled_2d = np.dot(filled_points - centroid, np.column_stack((u,v)))

    # --------------------------
    # Compute scaling
    # --------------------------
    hull = ConvexHull(filled_2d)
    min_xy = filled_2d.min(axis=0)
    max_xy = filled_2d.max(axis=0)
    scale = resolution / (max_xy - min_xy)
    scale_factor = np.min(scale)

    def to_img_coords(p):
        return np.clip(((p - min_xy) * scale_factor).astype(int), 0, resolution-1)

    filled_coords = to_img_coords(filled_2d)
    orig_coords = to_img_coords(points_2d)

    # --------------------------
    # Initialize image and mask
    # --------------------------
    img = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    mask = np.zeros((resolution, resolution), dtype=np.uint8)

    def draw_point(x, y, img_array, color):
        x_min = max(x - point_radius, 0)
        x_max = min(x + point_radius + 1, resolution)
        y_min = max(y - point_radius, 0)
        y_max = min(y + point_radius + 1, resolution)
        img_array[y_min:y_max, x_min:x_max] = color

    # --------------------------
    # Draw filled points with colors
    # --------------------------
    for idx, (x, y) in enumerate(filled_coords):
        draw_point(x, y, img, colors[idx])

    # --------------------------
    # Determine holes
    # --------------------------
    orig_kdtree = cKDTree(points)
    dists, _ = orig_kdtree.query(filled_points)
    hole_mask = dists > 0.05

    for idx, is_hole in enumerate(hole_mask):
        if is_hole:
            x, y = filled_coords[idx]
            draw_point(x, y, mask, 1)

    return img, mask, filled_coords

def rotation_matrix(axis, degrees=90):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    if axis.lower() == 'x':
        R = np.array([[1, 0, 0, 0],
                      [0, c,-s, 0],
                      [0, s, c, 0],
                      [0, 0, 0, 1]])
    elif axis.lower() == 'y':
        R = np.array([[ c, 0, s, 0],
                      [ 0, 1, 0, 0],
                      [-s, 0, c, 0],
                      [ 0, 0, 0, 1]])
    elif axis.lower() == 'z':
        R = np.array([[c,-s, 0, 0],
                      [s, c, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    return R