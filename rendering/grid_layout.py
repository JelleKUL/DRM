"""
PNG Grid Layout Tool
Arranges PNG images in a rectangular, hexagonal, or isometric grid pattern.

Usage:
  Rectangular:
    python grid_layout.py --input ./thumbnails --mode rect --cols 5 --rows 3 --output grid.png

  Hexagonal:
    python grid_layout.py --input ./thumbnails --mode hex --radius 3 --output grid.png

  Isometric:
    python grid_layout.py --input ./thumbnails --mode iso --cols 5 --rows 3 --output iso_grid.png

Options:
  --input       Folder containing PNG files
  --output      Output image path (default: grid.png)
  --mode        Layout mode: 'rect', 'hex', or 'iso'
  --cols        Number of columns (rect/iso mode)
  --rows        Number of rows (rect/iso mode)
  --radius      Hex grid radius, 1 = single tile, 2 = 7 tiles, 3 = 19, etc. (hex mode)
  --padding     Pixels between tiles (default: 4)
  --bg          Background color as hex, e.g. '#00000000' for transparent (default: transparent)
  --sort        Sort images alphabetically by name (default: true)
  --pointy      Hex mode: pointy-top orientation
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_images(folder: str, sort: bool = True) -> list[tuple[str, Image.Image]]:
    """Load all PNG files from a folder."""
    folder = Path(folder)
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    files = sorted(folder.glob("*.png")) if sort else list(folder.glob("*.png"))

    if not files:
        print(f"Error: No PNG files found in '{folder}'.")
        sys.exit(1)

    images = []
    for f in files:
        img = Image.open(f).convert("RGBA")
        images.append((f.stem, img))

    return images


def remove_white_bg(images: list[tuple[str, Image.Image]], threshold: int) -> list[tuple[str, Image.Image]]:
    """
    Make white/near-white pixels transparent.
    Any pixel where R, G, and B are all >= threshold gets alpha set to 0.
    """
    result = []
    for name, img in images:
        arr = np.array(img)
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
        is_white = (r >= threshold) & (g >= threshold) & (b >= threshold)
        arr[:, :, 3] = np.where(is_white, 0, a)
        result.append((name, Image.fromarray(arr, "RGBA")))
    return result


def parse_color(color_str: str) -> tuple[int, int, int, int]:
    """Parse a hex color string like '#FF0000' or '#FF000080' into an RGBA tuple."""
    color_str = color_str.lstrip("#")
    if len(color_str) == 6:
        r, g, b = int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16)
        return (r, g, b, 255)
    elif len(color_str) == 8:
        r, g, b, a = (
            int(color_str[0:2], 16),
            int(color_str[2:4], 16),
            int(color_str[4:6], 16),
            int(color_str[6:8], 16),
        )
        return (r, g, b, a)
    else:
        print(f"Error: Invalid color '{color_str}'. Use #RRGGBB or #RRGGBBAA.")
        sys.exit(1)


def make_uniform(images: list[tuple[str, Image.Image]], tile_size: int) -> list[tuple[str, Image.Image]]:
    """Resize all images to the same tile size, preserving aspect ratio and centering."""
    result = []
    for name, img in images:
        # Fit within tile_size x tile_size
        img.thumbnail((tile_size, tile_size), Image.LANCZOS)

        # Center on a transparent tile
        tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        offset_x = (tile_size - img.width) // 2
        offset_y = (tile_size - img.height) // 2
        tile.paste(img, (offset_x, offset_y), img)
        result.append((name, tile))

    return result


def layout_rect(
    images: list[tuple[str, Image.Image]],
    cols: int,
    rows: int,
    padding: int,
    bg_color: tuple[int, int, int, int],
) -> Image.Image:
    """Lay out images in a rectangular grid, discarding extras."""
    max_count = cols * rows
    images = images[:max_count]

    if not images:
        print("Error: No images to place.")
        sys.exit(1)

    tile_w = images[0][1].width
    tile_h = images[0][1].height

    canvas_w = cols * tile_w + (cols + 1) * padding
    canvas_h = rows * tile_h + (rows + 1) * padding

    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_color)

    for idx, (name, img) in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = padding + c * (tile_w + padding)
        y = padding + r * (tile_h + padding)
        canvas.paste(img, (x, y), img)

    return canvas


def layout_iso(
    images: list[tuple[str, Image.Image]],
    cols: int,
    rows: int,
    padding: int,
    bg_color: tuple[int, int, int, int],
) -> Image.Image:
    """
    Lay out images in an isometric diamond grid using true isometric projection.
    The two axes run at ±30° from horizontal, giving the standard 2:1 pixel ratio
    (for every 2px horizontal, 1px vertical). Each column steps right+down along
    one axis, each row steps left+down along the other.
    Extras beyond cols*rows are discarded.
    """
    max_count = cols * rows
    images = images[:max_count]

    if not images:
        print("Error: No images to place.")
        sys.exit(1)

    tile_size = images[0][1].width
    step = tile_size + padding

    # True isometric axes at ±30°:
    #   axis_col = (cos(30°), sin(30°)) = (√3/2, 1/2)
    #   axis_row = (-cos(30°), sin(30°)) = (-√3/2, 1/2)
    ax = step * math.sqrt(3) / 2
    ay = step * 0.5

    # Compute pixel positions for each cell
    pixel_positions = []
    for r in range(rows):
        for c in range(cols):
            px = c * ax - r * ax
            py = c * ay + r * ay
            pixel_positions.append((px, py))

    # Find bounding box
    min_px = min(p[0] for p in pixel_positions)
    max_px = max(p[0] for p in pixel_positions)
    min_py = min(p[1] for p in pixel_positions)
    max_py = max(p[1] for p in pixel_positions)

    margin = padding
    canvas_w = int(max_px - min_px) + tile_size + margin * 2
    canvas_h = int(max_py - min_py) + tile_size + margin * 2

    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_color)

    for idx, (name, img) in enumerate(images):
        if idx >= len(pixel_positions):
            break
        px, py = pixel_positions[idx]
        x = int(px - min_px) + margin
        y = int(py - min_py) + margin
        canvas.paste(img, (x, y), img)

    return canvas



    """
    Generate axial coordinates for a regular hexagonal grid.
    radius=1 → 1 tile (center only)
    radius=2 → 7 tiles
    radius=3 → 19 tiles
    The number of tiles is 3*r*(r-1)+1 where r = radius.

    pointy_top=False → flat-top hex (side at top)
    pointy_top=True  → pointy-top hex (vertex at top)
    """
    positions = []
    r = radius - 1  # rings around center
    for q in range(-r, r + 1):
        for s in range(-r, r + 1):
            cube_r = -q - s
            if abs(q) <= r and abs(s) <= r and abs(cube_r) <= r:
                positions.append((q, s))

    # Sort for consistent fill order: top-to-bottom, left-to-right
    def sort_key(axial):
        q, s = axial
        if pointy_top:
            px = q * (math.sqrt(3) / 2)
            py = q * 0.5 + s
        else:
            px = q + s * 0.5
            py = s * (math.sqrt(3) / 2)
        return (round(py, 4), round(px, 4))

    positions.sort(key=sort_key)
    return positions


def layout_hex(
    images: list[tuple[str, Image.Image]],
    radius: int,
    padding: int,
    bg_color: tuple[int, int, int, int],
    pointy_top: bool = False,
) -> Image.Image:
    """Lay out images in a regular hexagonal grid pattern."""
    positions = hex_grid_positions(radius, pointy_top)
    max_count = len(positions)
    images = images[:max_count]

    if not images:
        print("Error: No images to place.")
        sys.exit(1)

    tile_size = images[0][1].width
    step = tile_size + padding

    # Convert axial coords to pixel positions
    pixel_positions = []
    for q, s in positions:
        if pointy_top:
            # Pointy-top: columns aligned vertically, rows staggered horizontally
            px = q * (math.sqrt(3) / 2) * step
            py = (q * 0.5 + s) * step
        else:
            # Flat-top: rows aligned horizontally, columns staggered vertically
            px = (q + s * 0.5) * step
            py = s * (math.sqrt(3) / 2) * step
        pixel_positions.append((px, py))

    # Find bounding box
    min_px = min(p[0] for p in pixel_positions)
    max_px = max(p[0] for p in pixel_positions)
    min_py = min(p[1] for p in pixel_positions)
    max_py = max(p[1] for p in pixel_positions)

    margin = padding
    canvas_w = int(max_px - min_px) + tile_size + margin * 2
    canvas_h = int(max_py - min_py) + tile_size + margin * 2

    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_color)

    for idx, (name, img) in enumerate(images):
        if idx >= len(pixel_positions):
            break
        px, py = pixel_positions[idx]
        x = int(px - min_px) + margin
        y = int(py - min_py) + margin
        canvas.paste(img, (x, y), img)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Arrange PNGs in a rectangular or hexagonal grid.")
    parser.add_argument("--input", required=True, help="Folder containing PNG files")
    parser.add_argument("--output", default="grid.png", help="Output image path")
    parser.add_argument("--mode", choices=["rect", "hex", "iso"], required=True, help="Layout mode: rect, hex, or iso")
    parser.add_argument("--cols", type=int, default=5, help="Columns (rect/iso mode)")
    parser.add_argument("--rows", type=int, default=3, help="Rows (rect/iso mode)")
    parser.add_argument("--radius", type=int, default=3, help="Hex radius (hex mode): 1=1, 2=7, 3=19 tiles")
    parser.add_argument("--padding", type=int, default=4, help="Pixels between tiles")
    parser.add_argument("--tile-size", type=int, default=0, help="Force tile size in px (0 = use image size)")
    parser.add_argument("--bg", default="#00000000", help="Background color as #RRGGBB or #RRGGBBAA")
    parser.add_argument("--no-sort", action="store_true", help="Don't sort images alphabetically")
    parser.add_argument("--pointy", action="store_true", help="Hex mode: pointy-top orientation (default is flat-top/side at top)")
    parser.add_argument("--shuffle", action="store_true", help="Randomize the order of images")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --shuffle (for reproducible results)")
    parser.add_argument("--remove-bg", action="store_true", help="Make white background pixels transparent")
    parser.add_argument("--bg-threshold", type=int, default=240, help="Brightness threshold for --remove-bg (0-255, default: 240)")

    args = parser.parse_args()

    bg_color = parse_color(args.bg)
    images = load_images(args.input, sort=not args.no_sort)

    if args.remove_bg:
        images = remove_white_bg(images, args.bg_threshold)

    if args.shuffle:
        if args.seed is not None:
            random.seed(args.seed)
        random.shuffle(images)

    print(f"Loaded {len(images)} images from '{args.input}'")

    # Determine tile size
    tile_size = args.tile_size if args.tile_size > 0 else images[0][1].width
    images = make_uniform(images, tile_size)

    if args.mode == "rect":
        capacity = args.cols * args.rows
        if len(images) > capacity:
            print(f"Grid fits {capacity} tiles, discarding {len(images) - capacity} extra images.")
        result = layout_rect(images, args.cols, args.rows, args.padding, bg_color)
    elif args.mode == "iso":
        capacity = args.cols * args.rows
        if len(images) > capacity:
            print(f"Iso grid fits {capacity} tiles, discarding {len(images) - capacity} extra images.")
        result = layout_iso(images, args.cols, args.rows, args.padding, bg_color)
    else:
        capacity = 3 * (args.radius - 1) * args.radius + 1 if args.radius > 1 else 1
        if len(images) > capacity:
            print(f"Hex grid (radius {args.radius}) fits {capacity} tiles, discarding {len(images) - capacity} extra images.")
        result = layout_hex(images, args.radius, args.padding, bg_color, args.pointy)

    result.save(args.output)
    print(f"Saved {result.width}x{result.height} grid to '{args.output}'")


if __name__ == "__main__":
    main()