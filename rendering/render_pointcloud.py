"""
Isometric Point Cloud Renderer (trimesh + pyrender)
Renders point cloud .txt files as isometric images using trimesh's
orthographic camera and offscreen rendering via pyrender (OSMesa).

Usage:
  Single file:
    python render_pointcloud.py --input scan.txt --output render.png

  Folder of .txt files:
    python render_pointcloud.py --input ./scans --output ./renders

Options:
  --input       Input .txt file or folder of .txt files
  --output      Output .png file or folder
  --size        Image size in pixels, square (default: 1024)
  --point-size  Point size (default: 2.0)
  --bg          Background color as #RRGGBB or #RRGGBBAA (default: #FFFFFFFF)
  --margin      Extra margin factor around the scene (default: 1.1)

Requirements:
  pip install trimesh pyrender pyglet numpy Pillow
  For headless (no display): pip install PyOpenGL==3.1.7 osmesa
  Or set: export PYOPENGL_PLATFORM=osmesa
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
import pyrender
from PIL import Image


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_pointcloud(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a point cloud .txt file.
    Returns (points, colors_rgba) as numpy arrays.
    Applies transform_matrix from header if present.
    """
    points = []
    colors = []
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
                        vals = [float(v) for v in parts]
                        transform = np.array(vals).reshape(4, 4)
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

    if transform is not None and len(pts) > 0:
        ones = np.ones((len(pts), 1))
        pts_h = np.hstack([pts, ones])
        pts = (transform @ pts_h.T).T[:, :3]

    return pts, clr


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def clip_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    clip_ceiling_pct: float,
    clip_x_pct: float,
    clip_z_pct: float,
    abs_ceiling: float | None,
    abs_x: float | None,
    abs_z: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clip the point cloud to remove the ceiling and the two walls
    closest to the camera. Camera looks from (-X, +Y, -Z), so we remove:
      - Ceiling: points above a Y threshold
      - Near X wall: points below an X threshold (low X = close to camera)
      - Near Z wall: points below a Z threshold (low Z = close to camera)
    Thresholds are either absolute values or percentages of the bounding box range.
    """
    if len(points) == 0:
        return points, colors

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_range = bbox_max - bbox_min

    y_max = abs_ceiling if abs_ceiling is not None else bbox_min[1] + bbox_range[1] * (clip_ceiling_pct / 100.0)
    x_min = abs_x if abs_x is not None else bbox_min[0] + bbox_range[0] * (clip_x_pct / 100.0)
    z_min = abs_z if abs_z is not None else bbox_min[2] + bbox_range[2] * (clip_z_pct / 100.0)

    mask = (points[:, 1] <= y_max) & (points[:, 0] >= x_min) & (points[:, 2] >= z_min)
    return points[mask], colors[mask]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_isometric(
    points: np.ndarray,
    colors: np.ndarray,
    size: int,
    point_size: float,
    bg_color: tuple[float, float, float, float],
    margin: float,
    draw_bbox: bool = True,
    bbox_color: tuple[float, float, float] = (0.3, 0.3, 0.3),
) -> np.ndarray:
    """
    Render a point cloud isometrically using trimesh + pyrender
    with an orthographic camera. Returns an RGBA numpy array.
    """
    # Scene center and extent
    center = points.mean(axis=0)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = np.linalg.norm(bbox_max - bbox_min)

    # Isometric camera: looking from (-X, +Y, -Z) toward center
    # 45° azimuth, 35.264° elevation
    cam_dir = np.array([-1.0, 1.0, -1.0])
    cam_dir = cam_dir / np.linalg.norm(cam_dir)

    distance = extent * 2.0
    eye = center + cam_dir * distance

    # Build camera-to-world matrix (look-at)
    forward = center - eye
    forward = forward / np.linalg.norm(forward)

    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    true_up = np.cross(right, forward)

    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = true_up
    cam_pose[:3, 2] = -forward  # camera looks along -Z in its own frame
    cam_pose[:3, 3] = eye

    # Orthographic camera: xmag/ymag = half the visible world-space extent
    ortho_half = (extent * margin) / 2.0
    camera = pyrender.OrthographicCamera(
        xmag=ortho_half,
        ymag=ortho_half,
        znear=0.01,
        zfar=distance * 3,
    )

    # Build pyrender scene
    scene = pyrender.Scene(
        bg_color=np.array([bg_color[0], bg_color[1], bg_color[2], bg_color[3]]),
        ambient_light=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    # Add point cloud
    mesh = pyrender.Mesh.from_points(points, colors=colors / 255.0)
    scene.add(mesh)

    # Add bounding box wireframe
    if draw_bbox:
        bbox_mesh = _create_bbox_lines(bbox_min, bbox_max, bbox_color)
        scene.add(bbox_mesh)

    scene.add(camera, pose=cam_pose)

    # Offscreen render — use RGBA flags for transparent background
    renderer = pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size, point_size=point_size)
    color, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
    renderer.delete()

    return color


def _create_bbox_lines(bbox_min: np.ndarray, bbox_max: np.ndarray, color: tuple[float, float, float]) -> pyrender.Mesh:
    """
    Create a wireframe bounding box as thin cylinders between the 8 corners.
    Returns a pyrender.Mesh.
    """
    # 8 corners of the AABB
    lo, hi = bbox_min, bbox_max
    corners = np.array([
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]],
        [lo[0], hi[1], hi[2]],
    ])

    # 12 edges of a box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]

    extent = np.linalg.norm(bbox_max - bbox_min)
    radius = extent * 0.002  # thin lines relative to scene size

    primitives = []
    for i, j in edges:
        cyl = trimesh.creation.cylinder(
            radius=radius,
            segment=[corners[i], corners[j]],
        )
        cyl.visual.vertex_colors = np.array([*[int(c * 255) for c in color], 255], dtype=np.uint8)
        primitives.append(pyrender.Mesh.from_trimesh(cyl))

    # Combine all primitives into one mesh
    all_primitives = []
    for m in primitives:
        all_primitives.extend(m.primitives)

    return pyrender.Mesh(primitives=all_primitives)


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_file(
    filepath: str,
    output_path: str,
    size: int,
    point_size: float,
    bg_color: tuple[float, float, float, float],
    margin: float,
    clip_ceiling_pct: float,
    clip_x_pct: float,
    clip_z_pct: float,
    abs_ceiling: float | None,
    abs_x: float | None,
    abs_z: float | None,
    draw_bbox: bool = True,
    bbox_color: tuple[float, float, float] = (0.3, 0.3, 0.3),
):
    print(f"Processing: {filepath}")

    points, colors = parse_pointcloud(filepath)
    print(f"  Loaded {len(points)} points")

    if len(points) == 0:
        print("  Skipping: no valid points found.")
        return

    points, colors = clip_pointcloud(
        points, colors,
        clip_ceiling_pct, clip_x_pct, clip_z_pct,
        abs_ceiling, abs_x, abs_z,
    )
    print(f"  After clipping: {len(points)} points")

    if len(points) == 0:
        print("  Skipping: no points remaining after clipping.")
        return

    img_arr = render_isometric(points, colors, size, point_size, bg_color, margin, draw_bbox, bbox_color)

    img = Image.fromarray(img_arr, "RGBA")
    img.save(output_path)
    print(f"  Saved: {output_path}")


def parse_color(color_str: str) -> tuple[float, float, float, float]:
    """Parse #RRGGBB or #RRGGBBAA to (r, g, b, a) floats 0-1."""
    color_str = color_str.lstrip("#")
    if len(color_str) == 6:
        r, g, b = int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16)
        return (r / 255.0, g / 255.0, b / 255.0, 1.0)
    elif len(color_str) == 8:
        r, g, b, a = int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16), int(color_str[6:8], 16)
        return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
    else:
        print(f"Error: Invalid color '{color_str}'. Use #RRGGBB or #RRGGBBAA.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Render isometric point cloud images using trimesh + pyrender orthographic camera."
    )
    parser.add_argument("--input", required=True, help="Input .txt file or folder of .txt files")
    parser.add_argument("--output", default=None, help="Output .png file or folder")
    parser.add_argument("--size", type=int, default=1024, help="Square image size in pixels (default: 1024)")
    parser.add_argument("--point-size", type=float, default=2.0, help="Point size (default: 2.0)")
    parser.add_argument("--bg", default="#00000000", help="Background color as #RRGGBB or #RRGGBBAA (default: transparent)")
    parser.add_argument("--margin", type=float, default=1.1, help="Margin factor around scene (default: 1.1)")
    parser.add_argument("--clip-ceiling", type=float, default=90, help="Remove points above this %% of Y range (default: 90)")
    parser.add_argument("--clip-x", type=float, default=15, help="Remove points below this %% of X range (default: 15)")
    parser.add_argument("--clip-z", type=float, default=15, help="Remove points below this %% of Z range (default: 15)")
    parser.add_argument("--abs-ceiling", type=float, default=None, help="Absolute Y max cutoff (overrides --clip-ceiling)")
    parser.add_argument("--abs-x", type=float, default=None, help="Absolute X min cutoff (overrides --clip-x)")
    parser.add_argument("--abs-z", type=float, default=None, help="Absolute Z min cutoff (overrides --clip-z)")
    parser.add_argument("--no-clip", action="store_true", help="Disable all clipping, render full cloud")
    parser.add_argument("--pattern", default="main*", help="Filename glob pattern to match (default: 'main*'). Use '*Clone*' for objects.")
    parser.add_argument("--no-bbox", action="store_true", help="Disable bounding box wireframe")
    parser.add_argument("--bbox-color", default="#4D4D4D", help="Bounding box color as #RRGGBB (default: #4D4D4D)")

    args = parser.parse_args()

    bg_color = parse_color(args.bg)
    bbox_color_hex = args.bbox_color.lstrip("#")
    bbox_color = (
        int(bbox_color_hex[0:2], 16) / 255.0,
        int(bbox_color_hex[2:4], 16) / 255.0,
        int(bbox_color_hex[4:6], 16) / 255.0,
    )
    draw_bbox = not args.no_bbox
    input_path = Path(args.input)

    # If no-clip or pattern isn't main*, disable clipping
    is_scene = args.pattern.startswith("main")
    no_clip = args.no_clip or not is_scene
    clip_ceiling = 100.0 if no_clip else args.clip_ceiling
    clip_x = 0.0 if no_clip else args.clip_x
    clip_z = 0.0 if no_clip else args.clip_z
    abs_ceiling = None if no_clip else args.abs_ceiling
    abs_x = None if no_clip else args.abs_x
    abs_z = None if no_clip else args.abs_z

    # Build the glob pattern with .txt extension
    glob_pattern = args.pattern + ".txt" if not args.pattern.endswith(".txt") else args.pattern

    if input_path.is_file():
        output = args.output or input_path.stem + "_render.png"
        process_file(
            str(input_path), output, args.size, args.point_size, bg_color, args.margin,
            clip_ceiling, clip_x, clip_z, abs_ceiling, abs_x, abs_z,
            draw_bbox, bbox_color,
        )

    elif input_path.is_dir():
        output_dir = Path(args.output or "renders")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Recursively find matching .txt files in subfolders
        txt_files = sorted(input_path.rglob(glob_pattern))
        if not txt_files:
            print(f"No files matching '{glob_pattern}' found in '{input_path}' or its subfolders.")
            sys.exit(1)

        print(f"Found {len(txt_files)} point cloud files matching '{glob_pattern}'")

        for txt_file in txt_files:
            # Name: parentfolder_filename.png
            folder_name = txt_file.parent.name
            out_name = f"{folder_name}_{txt_file.stem}.png"
            out_file = output_dir / out_name

            process_file(
                str(txt_file), str(out_file), args.size, args.point_size, bg_color, args.margin,
                clip_ceiling, clip_x, clip_z, abs_ceiling, abs_x, abs_z,
                draw_bbox, bbox_color,
            )

        print(f"\nDone. {len(txt_files)} renders saved to '{output_dir}'")

    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()