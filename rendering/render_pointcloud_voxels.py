"""
Isometric Point Cloud + Voxel Grid Renderer
Renders a point cloud .txt file alongside its occlusion/occupation .ply voxel grids.
Points are subsampled to the voxel size and rendered as spheres.
Voxels are rendered as cubes slightly smaller than the voxel size.

Usage:
  Single file:
    python render_pointcloud_voxels.py --input scan.txt --output render.png

  Folder (processes each subfolder containing main.txt):
    python render_pointcloud_voxels.py --input ./data --output ./renders

Options:
  --input           Input .txt file or data folder
  --output          Output .png file or folder
  --size            Image size in pixels (default: 1024)
  --bg              Background color #RRGGBBAA (default: transparent)
  --margin          Margin factor (default: 1.1)
  --voxel-size      Voxel size matching the saved grids (default: 0.2)
  --cube-fill       Cube size as fraction of voxel size (default: 0.85)
  --sphere-radius   Sphere radius as fraction of voxel size (default: 0.4)
  --occupied-file   Filename of occupied grid ply (default: occupied_grid.ply)
  --occluded-file   Filename of occluded grid ply (default: occluded_grid.ply)
  --occupied-color  Color of occupied voxels #RRGGBBAA (default: #E8844AFF)
  --occluded-color  Color of occluded voxels #RRGGBBAA (default: #4A90E8AA)
  --no-clip         Disable wall/ceiling clipping
  --clip-ceiling    % of Y range to clip ceiling (default: 90)
  --clip-x          % of X range to clip near wall (default: 15)
  --clip-z          % of Z range to clip near wall (default: 15)
  --pattern         Glob pattern for txt files (default: main*)
  --no-bbox         Disable bounding box
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
import pyrender
from PIL import Image


# ---------------------------------------------------------------------------
# Point cloud parsing
# ---------------------------------------------------------------------------

UNITY2TRIMESH_T = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def parse_pointcloud(
    filepath,
    apply_unity_conversion: bool = False,
    apply_file_translation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a point cloud .txt file.

    Args:
        filepath:                Path to the .txt file
        apply_unity_conversion:  Swap Y and Z axes to match Unity/trimesh space
        apply_file_translation:  Subtract the scan position (from the header matrix,
                                 converted to Unity space) to center at origin
    """
    points, colors = [], []
    transform = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                content = line.lstrip("#").strip()
                if content.startswith("transform_matrix"):
                    continue
                parts = content.split()
                if len(parts) == 16:
                    try:
                        transform = np.array([float(v) for v in parts]).reshape(4, 4)
                    except ValueError:
                        pass
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                r, g, b = int(parts[6]), int(parts[7]), int(parts[8])
                points.append([x, y, z])
                colors.append([r, g, b, 255])
            except (ValueError, IndexError):
                continue

    pts = np.array(points, dtype=np.float64)
    clr = np.array(colors, dtype=np.uint8)

    if len(pts) == 0:
        return pts, clr

    if apply_unity_conversion:
        pts = pts[:, [0, 2, 1]]

    if apply_file_translation and transform is not None:
        # Transform the matrix into Unity space, extract translation, subtract
        translation = (UNITY2TRIMESH_T @ transform)[:3, 3]
        pts = pts - translation

    return pts, clr


# ---------------------------------------------------------------------------
# Voxel grid loading
# ---------------------------------------------------------------------------

def load_voxelgrid_centers(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load voxel grid PLY as-is — no coordinate transformation applied."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    centers = np.asarray(pcd.points, dtype=np.float64)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    if len(colors) == 0 or colors.shape[1] < 3:
        colors = np.full((len(centers), 3), 128, dtype=np.uint8)

    return centers, colors


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def clip_pointcloud(points, colors, clip_ceiling_pct, clip_x_pct, clip_y_pct,
                    abs_ceiling, abs_x, abs_y):
    if len(points) == 0:
        return points, colors
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_range = bbox_max - bbox_min

    # Z-up, camera looks toward +X/+Y:
    # - ceiling = high Z → clip above z_max
    # - near X wall = low X (negative direction) → clip below x_min
    # - near Y wall = low Y (negative direction) → clip below y_min
    z_max = abs_ceiling if abs_ceiling is not None else bbox_max[2] - bbox_range[2] * (clip_ceiling_pct / 100.0)
    x_min = abs_x       if abs_x       is not None else bbox_min[0] + bbox_range[0] * (clip_x_pct      / 100.0)
    y_min = abs_y       if abs_y       is not None else bbox_min[1] + bbox_range[1] * (clip_y_pct      / 100.0)

    mask = (points[:, 2] <= z_max) & (points[:, 0] >= x_min) & (points[:, 1] >= y_min)
    return points[mask], colors[mask]


def clip_centers(centers, clip_ceiling_pct, clip_x_pct, clip_y_pct,
                 abs_ceiling, abs_x, abs_y, ref_bbox_min, ref_bbox_range):
    """Clip voxel centers using the same thresholds as the point cloud."""
    if len(centers) == 0:
        return centers

    ref_bbox_max = ref_bbox_min + ref_bbox_range

    z_max = abs_ceiling if abs_ceiling is not None else ref_bbox_max[2] - ref_bbox_range[2] * (clip_ceiling_pct / 100.0)
    x_min = abs_x       if abs_x       is not None else ref_bbox_min[0] + ref_bbox_range[0] * (clip_x_pct      / 100.0)
    y_min = abs_y       if abs_y       is not None else ref_bbox_min[1] + ref_bbox_range[1] * (clip_y_pct      / 100.0)

    mask = (centers[:, 2] <= z_max) & (centers[:, 0] >= x_min) & (centers[:, 1] >= y_min)
    return centers[mask]


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def points_to_spheres(
    points: np.ndarray,
    colors: np.ndarray,
    radius: float,
    subdivisions: int = 2,
) -> trimesh.Trimesh:
    """Convert point cloud to a mesh of spheres."""
    if len(points) == 0:
        return trimesh.Trimesh()

    template = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    n_v = len(template.vertices)
    n_f = len(template.faces)
    n   = len(points)

    all_verts  = np.empty((n * n_v, 3), dtype=np.float64)
    all_faces  = np.empty((n * n_f, 3), dtype=np.int64)
    all_colors = np.empty((n * n_v, 4), dtype=np.uint8)

    for i, (pt, c) in enumerate(zip(points, colors)):
        all_verts[i*n_v:(i+1)*n_v] = template.vertices + pt
        all_faces[i*n_f:(i+1)*n_f] = template.faces + i * n_v
        rgba = c if len(c) == 4 else np.append(c, 255)
        all_colors[i*n_v:(i+1)*n_v] = rgba

    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    mesh.visual.vertex_colors = all_colors
    return mesh


def points_to_cubes(
    points: np.ndarray,
    colors: np.ndarray,
    cube_size: float,
    face_shading: tuple = (0.55, 0.75, 1.0),
) -> trimesh.Trimesh:
    """
    Convert point cloud to a mesh of shaded cubes, one per point.
    Each cube uses the point's own color, shaded per axis like voxel cubes.
    """
    if len(points) == 0:
        return trimesh.Trimesh()

    template = trimesh.creation.box(extents=[cube_size, cube_size, cube_size])
    verts = template.vertices
    faces = template.faces
    face_normals = template.face_normals

    shading = np.array([face_shading[np.argmax(np.abs(n))] for n in face_normals])

    n_v = len(verts)
    n_f = len(faces)
    n   = len(points)

    all_verts  = np.empty((n * n_v, 3), dtype=np.float64)
    all_faces  = np.empty((n * n_f, 3), dtype=np.int64)
    all_colors = np.empty((n * n_f, 4), dtype=np.uint8)

    for i, (pt, c) in enumerate(zip(points, colors)):
        all_verts[i*n_v:(i+1)*n_v] = verts + pt
        all_faces[i*n_f:(i+1)*n_f] = faces + i * n_v

        base_rgb = c[:3].astype(np.float64)
        base_a   = c[3] if len(c) == 4 else 255
        shaded   = np.clip(base_rgb[None, :] * shading[:, None], 0, 255).astype(np.uint8)
        alpha    = np.full((n_f, 1), base_a, dtype=np.uint8)
        all_colors[i*n_f:(i+1)*n_f] = np.hstack([shaded, alpha])

    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=all_colors)
    return mesh


def centers_to_cubes(
    centers: np.ndarray,
    color_rgba: np.ndarray,
    cube_size: float,
    face_shading: tuple = (0.55, 0.75, 1.0),
) -> trimesh.Trimesh:
    """
    Convert voxel centers to a mesh of cubes with face shading.
    The three axis-aligned face pairs get different brightness multipliers
    so the cube reads as a 3D object even without a light source.

    face_shading: (top/bottom Z, front/back Y, left/right X) brightness multipliers.
    """
    if len(centers) == 0:
        return trimesh.Trimesh()

    # Box vertices and faces from trimesh
    template = trimesh.creation.box(extents=[cube_size, cube_size, cube_size])
    verts = template.vertices  # 8 vertices
    faces = template.faces     # 12 triangles (2 per face = 6 faces)

    # Determine which axis each face points along by its normal
    # Compute per-face normal from the template
    face_normals = template.face_normals  # [12, 3]

    # Map each face to a shading multiplier based on dominant axis
    # abs of normal tells us the axis; sign doesn't matter for shading
    shading = np.ones(len(faces))
    for i, n in enumerate(face_normals):
        dominant = np.argmax(np.abs(n))
        shading[i] = face_shading[dominant]  # X=face_shading[0], Y=[1], Z=[2]

    n_v = len(verts)
    n_f = len(faces)
    n   = len(centers)

    all_verts  = np.empty((n * n_v, 3), dtype=np.float64)
    all_faces  = np.empty((n * n_f, 3), dtype=np.int64)
    all_colors = np.empty((n * n_f, 4), dtype=np.uint8)  # per-face colors

    base_rgb = color_rgba[:3].astype(np.float64)
    base_a   = color_rgba[3]

    for i, pt in enumerate(centers):
        all_verts[i*n_v:(i+1)*n_v] = verts + pt
        all_faces[i*n_f:(i+1)*n_f] = faces + i * n_v

        # Apply shading per face
        shaded = np.clip(base_rgb[None, :] * shading[:, None], 0, 255).astype(np.uint8)
        alpha_col = np.full((n_f, 1), base_a, dtype=np.uint8)
        all_colors[i*n_f:(i+1)*n_f] = np.hstack([shaded, alpha_col])

    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    # Use face colors instead of vertex colors for crisp face shading
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=all_colors)
    return mesh


def create_bbox_lines(bbox_min, bbox_max, color, radius_fraction=0.002):
    lo, hi = bbox_min, bbox_max
    corners = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    extent = np.linalg.norm(bbox_max - bbox_min)
    radius = extent * radius_fraction
    primitives = []
    for i, j in edges:
        cyl = trimesh.creation.cylinder(radius=radius, segment=[corners[i], corners[j]])
        cyl.visual.vertex_colors = np.tile(color, (len(cyl.vertices), 1)).astype(np.uint8)
        primitives.extend(pyrender.Mesh.from_trimesh(cyl).primitives)
    return pyrender.Mesh(primitives=primitives)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def build_camera_pose(center, extent):
    # Z is up. Look from (-X, -Y, +Z) toward (+X, +Y, -Z) — isometric, positive X/Y quadrant.
    cam_dir = np.array([-1.0, -1.0, 1.0])
    cam_dir /= np.linalg.norm(cam_dir)
    distance = extent * 2.0
    eye = center + cam_dir * distance

    forward = center - eye
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose, distance


def render_scene(
    pr_meshes: list,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    size: int,
    bg_color: tuple,
    margin: float,
    draw_bbox: bool,
    bbox_color: np.ndarray,
) -> np.ndarray:
    center = (bbox_min + bbox_max) / 2
    extent = np.linalg.norm(bbox_max - bbox_min)
    cam_pose, distance = build_camera_pose(center, extent)

    ortho_half = (extent * margin) / 2.0
    camera = pyrender.OrthographicCamera(
        xmag=ortho_half, ymag=ortho_half,
        znear=0.01, zfar=distance * 3,
    )

    scene = pyrender.Scene(
        bg_color=np.array(bg_color, dtype=np.float32),
        ambient_light=np.array([0.6, 0.6, 0.6, 1.0]),
    )

    # Add a directional light for shading on the mesh geometry
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_pose = cam_pose.copy()
    scene.add(light, pose=light_pose)

    for mesh in pr_meshes:
        scene.add(mesh)

    if draw_bbox:
        scene.add(create_bbox_lines(bbox_min, bbox_max, bbox_color))

    scene.add(camera, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size, point_size=1.0)
    color, _ = renderer.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
    renderer.delete()
    return color


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    grid_origin: np.ndarray | None = None,
):
    """
    Downsample points to voxel grid centers.
    If grid_origin is provided, the grid is aligned to that origin
    so voxel centers match an externally defined grid (e.g. the occlusion grid).
    Otherwise snaps to the global (0,0,0) origin.
    Colors are averaged per voxel.
    """
    if len(points) == 0:
        return points, colors

    if grid_origin is None:
        grid_origin = np.zeros(3)

    # Shift points relative to the grid origin before snapping
    shifted = points - grid_origin
    indices = np.floor(shifted / voxel_size).astype(np.int64)
    # Voxel centers in world space
    centers = grid_origin + (indices + 0.5) * voxel_size

    keys = indices[:, 0] * 1_000_003 + indices[:, 1] * 1_000_033 + indices[:, 2]
    _, unique_idx, inverse = np.unique(keys, return_index=True, return_inverse=True)

    voxel_centers = centers[unique_idx]

    avg_colors = np.zeros((len(unique_idx), colors.shape[1]), dtype=np.float64)
    np.add.at(avg_colors, inverse, colors.astype(np.float64))
    counts = np.bincount(inverse)
    avg_colors = (avg_colors / counts[:, None]).astype(np.uint8)

    return voxel_centers, avg_colors


# ---------------------------------------------------------------------------
# Process single file
# ---------------------------------------------------------------------------

def process_file(
    txt_path: Path,
    output_path: Path,
    occupied_path: Path | None,
    occluded_path: Path | None,
    size: int,
    bg_color: tuple,
    margin: float,
    voxel_size: float,
    cube_fill: float,
    sphere_radius_fraction: float,
    occupied_color: np.ndarray,
    occluded_color: np.ndarray,
    clip_ceiling_pct: float,
    clip_x_pct: float,
    clip_y_pct: float,
    abs_ceiling, abs_x, abs_y,
    draw_bbox: bool,
    bbox_color: np.ndarray,
    point_style: str = "spheres",  # "spheres" or "cubes"
):
    print(f"Processing: {txt_path.name}")

    # --- Load and clip point cloud ---
    points, colors = parse_pointcloud(
        str(txt_path),
        apply_unity_conversion=True,
        apply_file_translation=True,
    )
    print(f"  Loaded {len(points)} points")

    if len(points) == 0:
        print("  Skipping: no valid points.")
        return

    bbox_min_full = points.min(axis=0)
    bbox_range_full = points.max(axis=0) - bbox_min_full

    points, colors = clip_pointcloud(
        points, colors,
        clip_ceiling_pct, clip_x_pct, clip_y_pct,
        abs_ceiling, abs_x, abs_y,
    )
    print(f"  After clipping: {len(points)} points")

    if len(points) == 0:
        print("  Skipping: no points after clipping.")
        return

    # --- Derive grid origin from occlusion or occupied grid ---
    # Use the minimum corner of the voxel grid as the shared grid origin
    # so point cloud voxels snap to the same grid as the occlusion grid.
    grid_origin = None
    for ply_path in [occluded_path, occupied_path]:
        if ply_path is not None and ply_path.exists():
            ref_centers, _ = load_voxelgrid_centers(ply_path)
            if len(ref_centers) > 0:
                # Grid origin = min corner of the first voxel center minus half voxel
                grid_origin = ref_centers.min(axis=0) - voxel_size * 0.5
                break

    # --- Subsample point cloud to voxel size ---
    ds_points, ds_colors = voxel_downsample(points, colors, voxel_size, grid_origin=grid_origin)
    print(f"  After downsampling: {len(ds_points)} points")

    sphere_radius = voxel_size * sphere_radius_fraction
    cube_size     = voxel_size * cube_fill

    # --- Build point cloud mesh ---
    if point_style == "cubes":
        point_mesh = points_to_cubes(ds_points, ds_colors, cube_size=cube_size)
    else:
        point_mesh = points_to_spheres(ds_points, ds_colors, radius=sphere_radius)

    pr_meshes = []
    if len(point_mesh.vertices) > 0:
        pr_meshes.append(pyrender.Mesh.from_trimesh(point_mesh, smooth=False))

    # --- Load and add voxel grids ---
    all_centers = [ds_points]  # for bbox computation

    for ply_path, color_rgba in [(occupied_path, occupied_color), (occluded_path, occluded_color)]:
        if ply_path is None or not ply_path.exists():
            continue
        centers, _ = load_voxelgrid_centers(ply_path)
        print(f"  Loaded {len(centers)} voxels from {ply_path.name}")

        centers = clip_centers(
            centers,
            clip_ceiling_pct, clip_x_pct, clip_y_pct,
            abs_ceiling, abs_x, abs_y,
            bbox_min_full, bbox_range_full,
        )
        print(f"  After clipping: {len(centers)} voxels")

        if len(centers) == 0:
            continue

        cube_mesh = centers_to_cubes(centers, color_rgba, cube_size)
        if len(cube_mesh.vertices) > 0:
            pr_meshes.append(pyrender.Mesh.from_trimesh(cube_mesh, smooth=False))
            all_centers.append(centers)

    if not pr_meshes:
        print("  Skipping: nothing to render.")
        return

    # Compute scene bbox from all geometry
    all_pts = np.vstack(all_centers)
    bbox_min = all_pts.min(axis=0)
    bbox_max = all_pts.max(axis=0)

    img_arr = render_scene(pr_meshes, bbox_min, bbox_max, size, bg_color, margin, draw_bbox, bbox_color)
    Image.fromarray(img_arr, "RGBA").save(output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def parse_color_float(s: str) -> tuple:
    s = s.lstrip("#")
    if len(s) == 6:
        return tuple(int(s[i:i+2], 16) / 255.0 for i in (0, 2, 4)) + (1.0,)
    elif len(s) == 8:
        return tuple(int(s[i:i+2], 16) / 255.0 for i in (0, 2, 4, 6))
    raise ValueError(f"Invalid color: {s}")


def parse_color_rgba(s: str) -> np.ndarray:
    s = s.lstrip("#")
    if len(s) == 6:
        vals = [int(s[i:i+2], 16) for i in (0, 2, 4)] + [255]
    elif len(s) == 8:
        vals = [int(s[i:i+2], 16) for i in (0, 2, 4, 6)]
    else:
        raise ValueError(f"Invalid color: {s}")
    return np.array(vals, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render point clouds with voxel grid overlays.")
    parser.add_argument("--input",          required=True)
    parser.add_argument("--output",         default=None)
    parser.add_argument("--size",           type=int,   default=1024)
    parser.add_argument("--bg",             default="#00000000")
    parser.add_argument("--margin",         type=float, default=1.1)
    parser.add_argument("--voxel-size",     type=float, default=0.2)
    parser.add_argument("--cube-fill",      type=float, default=0.85,
                        help="Cube size as fraction of voxel size (default: 0.85)")
    parser.add_argument("--sphere-radius",  type=float, default=0.3,
                        help="Sphere radius as fraction of voxel size (default: 0.3)")
    parser.add_argument("--occupied-file",  default="occupied_grid.ply")
    parser.add_argument("--occluded-file",  default="occluded_grid.ply")
    parser.add_argument("--no-occupied",    action="store_true", help="Skip occupied grid")
    parser.add_argument("--no-occluded",    action="store_true", help="Skip occluded grid")
    parser.add_argument("--occupied-color", default="#5A4AE8FF")
    parser.add_argument("--occluded-color", default="#DD2F18FF")
    parser.add_argument("--pattern",        default="main.txt")
    parser.add_argument("--clip-ceiling",   type=float, default=10)
    parser.add_argument("--clip-x",         type=float, default=10)
    parser.add_argument("--clip-y",         type=float, default=10)
    parser.add_argument("--abs-ceiling",    type=float, default=None)
    parser.add_argument("--abs-x",         type=float, default=None)
    parser.add_argument("--abs-y",         type=float, default=None)
    parser.add_argument("--no-clip",        action="store_true")
    parser.add_argument("--no-bbox",        action="store_true")
    parser.add_argument("--bbox-color",     default="#4D4D4DFF")
    parser.add_argument("--point-style",    default="spheres", choices=["spheres", "cubes"],
                        help="Render point cloud as spheres or shaded cubes (default: spheres)")

    args = parser.parse_args()

    bg_color       = parse_color_float(args.bg)
    occupied_color = parse_color_rgba(args.occupied_color)
    occluded_color = parse_color_rgba(args.occluded_color)
    bbox_color     = parse_color_rgba(args.bbox_color)
    draw_bbox      = not args.no_bbox

    is_scene = args.pattern.startswith("main")
    no_clip  = args.no_clip or not is_scene
    clip_ceiling = 100.0 if no_clip else args.clip_ceiling
    clip_x       = 0.0   if no_clip else args.clip_x
    clip_y       = 0.0   if no_clip else args.clip_y
    abs_ceiling  = None  if no_clip else args.abs_ceiling
    abs_x        = None  if no_clip else args.abs_x
    abs_y        = None  if no_clip else args.abs_y

    glob_pattern = args.pattern if args.pattern.endswith(".txt") else args.pattern + ".txt"
    input_path   = Path(args.input)

    def _process(txt_file: Path, out_file: Path):
        folder = txt_file.parent
        process_file(
            txt_path       = txt_file,
            output_path    = out_file,
            occupied_path  = None if args.no_occupied else folder / args.occupied_file,
            occluded_path  = None if args.no_occluded else folder / args.occluded_file,
            size           = args.size,
            bg_color       = bg_color,
            margin         = args.margin,
            voxel_size     = args.voxel_size,
            cube_fill      = args.cube_fill,
            sphere_radius_fraction = args.sphere_radius,
            occupied_color = occupied_color,
            occluded_color = occluded_color,
            clip_ceiling_pct = clip_ceiling,
            clip_x_pct    = clip_x,
            clip_y_pct    = clip_y,
            abs_ceiling   = abs_ceiling,
            abs_x         = abs_x,
            abs_y         = abs_y,
            draw_bbox     = draw_bbox,
            bbox_color    = bbox_color,
            point_style   = args.point_style,
        )

    if input_path.is_file():
        output = Path(args.output or input_path.stem + "_render.png")
        _process(input_path, output)

    elif input_path.is_dir():
        output_dir = Path(args.output or "renders")
        output_dir.mkdir(parents=True, exist_ok=True)
        txt_files = sorted(input_path.rglob(glob_pattern))
        if not txt_files:
            print(f"No files matching '{glob_pattern}' found.")
            sys.exit(1)
        print(f"Found {len(txt_files)} files\n")
        for txt_file in txt_files:
            out_name = f"{txt_file.parent.name}_{txt_file.stem}.png"
            _process(txt_file, output_dir / out_name)
        print(f"\nDone. {len(txt_files)} renders saved to '{output_dir}'")
    else:
        print(f"Error: '{input_path}' is not a valid path.")
        sys.exit(1)


if __name__ == "__main__":
    main()