"""
Colorwheel Grid Layout Tool
Takes a small set of transparent PNG images, repeats them randomly to fill a grid,
and recolors the white/bright pixels of each tile based on its angular position
relative to the center of the collage (mapping angle → HSL hue).

Usage:
  Rectangular:
    python grid_layout_color.py --input ./tiles --mode rect --cols 8 --rows 6 --output color_grid.png

  Hexagonal:
    python grid_layout_color.py --input ./tiles --mode hex --radius 4 --output color_hex.png

  Isometric:
    python grid_layout_color.py --input ./tiles --mode iso --cols 6 --rows 6 --output color_iso.png

Options:
  --input       Folder containing PNG files (your small set of source images)
  --output      Output image path (default: color_grid.png)
  --mode        Layout mode: 'rect', 'hex', or 'iso'
  --cols        Number of columns (rect/iso mode)
  --rows        Number of rows (rect/iso mode)
  --radius      Hex grid radius (hex mode)
  --padding     Pixels between tiles (default: 4)
  --tile-size   Force tile size in px (0 = use image size)
  --bg          Background color as #RRGGBB or #RRGGBBAA (default: transparent)
  --pointy      Hex mode: pointy-top orientation
  --seed        Random seed for reproducible results
  --saturation  HSL saturation for the recolor (0.0-1.0, default: 0.8)
  --lightness   HSL lightness for the recolor (0.0-1.0, default: 0.55)
  --threshold   Brightness threshold (0-255) above which pixels are considered "white" (default: 200)
"""

import argparse
import colorsys
import math
import random
import sys
from pathlib import Path

from PIL import Image
import numpy as np


def load_images(folder: str) -> list[tuple[str, Image.Image]]:
    """Load all PNG files from a folder."""
    folder = Path(folder)
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    files = sorted(folder.glob("*.png"))

    if not files:
        print(f"Error: No PNG files found in '{folder}'.")
        sys.exit(1)

    images = []
    for f in files:
        img = Image.open(f).convert("RGBA")
        images.append((f.stem, img))

    return images


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
        img.thumbnail((tile_size, tile_size), Image.LANCZOS)
        tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        offset_x = (tile_size - img.width) // 2
        offset_y = (tile_size - img.height) // 2
        tile.paste(img, (offset_x, offset_y), img)
        result.append((name, tile))
    return result


def recolor_white_pixels(img: Image.Image, hue: float, saturation: float, lightness: float, threshold: int) -> Image.Image:
    """
    Replace white/bright pixels with a color derived from the given HSL hue.
    Pixels where R, G, and B are all above `threshold` (and alpha > 0) are
    considered "white" and get recolored. The pixel's original brightness
    modulates the lightness so shading/gradients are preserved.
    """
    arr = np.array(img, dtype=np.float32)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

    # Mask: bright pixels that are visible
    is_white = (r >= threshold) & (g >= threshold) & (b >= threshold) & (a > 0)

    if not np.any(is_white):
        return img

    # Original brightness as a 0-1 factor to preserve shading
    brightness = (r + g + b) / (3.0 * 255.0)

    # Convert target HSL to RGB
    tr, tg, tb = colorsys.hls_to_rgb(hue, lightness, saturation)

    # Modulate target color by original brightness relative to pure white
    # This preserves highlights and soft shadows in the original
    out = arr.copy()
    out[:, :, 0] = np.where(is_white, np.clip(tr * 255.0 * brightness, 0, 255), r)
    out[:, :, 1] = np.where(is_white, np.clip(tg * 255.0 * brightness, 0, 255), g)
    out[:, :, 2] = np.where(is_white, np.clip(tb * 255.0 * brightness, 0, 255), b)

    return Image.fromarray(out.astype(np.uint8), "RGBA")


def fill_randomly(images: list[tuple[str, Image.Image]], count: int) -> list[tuple[str, Image.Image]]:
    """Repeat images randomly until we have `count` tiles."""
    result = []
    for _ in range(count):
        name, img = random.choice(images)
        result.append((name, img.copy()))
    return result


# ---------------------------------------------------------------------------
# Grid position generators — return list of (px, py) pixel positions
# and the canvas size needed
# ---------------------------------------------------------------------------

def rect_positions(cols, rows, tile_size, padding):
    positions = []
    for r in range(rows):
        for c in range(cols):
            x = padding + c * (tile_size + padding)
            y = padding + r * (tile_size + padding)
            positions.append((x, y))
    canvas_w = cols * tile_size + (cols + 1) * padding
    canvas_h = rows * tile_size + (rows + 1) * padding
    return positions, canvas_w, canvas_h


def iso_positions(cols, rows, tile_size, padding):
    step = tile_size + padding
    ax = step * math.sqrt(3) / 2
    ay = step * 0.5

    raw = []
    for r in range(rows):
        for c in range(cols):
            px = c * ax - r * ax
            py = c * ay + r * ay
            raw.append((px, py))

    min_px = min(p[0] for p in raw)
    min_py = min(p[1] for p in raw)
    max_px = max(p[0] for p in raw)
    max_py = max(p[1] for p in raw)

    margin = padding
    positions = [(int(px - min_px) + margin, int(py - min_py) + margin) for px, py in raw]
    canvas_w = int(max_px - min_px) + tile_size + margin * 2
    canvas_h = int(max_py - min_py) + tile_size + margin * 2
    return positions, canvas_w, canvas_h


def hex_grid_coords(radius, pointy_top=False):
    r = radius - 1
    positions = []
    for q in range(-r, r + 1):
        for s in range(-r, r + 1):
            cube_r = -q - s
            if abs(q) <= r and abs(s) <= r and abs(cube_r) <= r:
                positions.append((q, s))

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


def hex_positions(radius, tile_size, padding, pointy_top=False):
    coords = hex_grid_coords(radius, pointy_top)
    step = tile_size + padding

    raw = []
    for q, s in coords:
        if pointy_top:
            px = q * (math.sqrt(3) / 2) * step
            py = (q * 0.5 + s) * step
        else:
            px = (q + s * 0.5) * step
            py = s * (math.sqrt(3) / 2) * step
        raw.append((px, py))

    min_px = min(p[0] for p in raw)
    min_py = min(p[1] for p in raw)
    max_px = max(p[0] for p in raw)
    max_py = max(p[1] for p in raw)

    margin = padding
    positions = [(int(px - min_px) + margin, int(py - min_py) + margin) for px, py in raw]
    canvas_w = int(max_px - min_px) + tile_size + margin * 2
    canvas_h = int(max_py - min_py) + tile_size + margin * 2
    return positions, canvas_w, canvas_h


# ---------------------------------------------------------------------------
# Main compositing
# ---------------------------------------------------------------------------

def compose(
    source_images: list[tuple[str, Image.Image]],
    positions: list[tuple[int, int]],
    canvas_w: int,
    canvas_h: int,
    bg_color: tuple[int, int, int, int],
    saturation: float,
    lightness: float,
    threshold: int,
    ramp: float,
) -> Image.Image:
    """
    Fill positions with randomly repeated source images, recoloring white pixels
    based on each tile's angular position from the canvas center.
    Tiles near the center stay white; saturation increases with distance.
    """
    count = len(positions)
    tiles = fill_randomly(source_images, count)

    # Canvas center
    cx = canvas_w / 2.0
    cy = canvas_h / 2.0

    # Find the maximum distance any tile center has from the canvas center
    # so we can normalize distance to 0-1
    max_dist = 0.0
    for px, py in positions:
        tile_cx = px + tiles[0][1].width / 2.0
        tile_cy = py + tiles[0][1].height / 2.0
        dist = math.hypot(tile_cx - cx, tile_cy - cy)
        max_dist = max(max_dist, dist)

    if max_dist == 0:
        max_dist = 1.0  # single tile, avoid division by zero

    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_color)

    for idx, ((name, img), (px, py)) in enumerate(zip(tiles, positions)):
        # Tile center
        tile_cx = px + img.width / 2.0
        tile_cy = py + img.height / 2.0

        # Angle from canvas center → hue (0-1)
        dx = tile_cx - cx
        dy = tile_cy - cy
        angle = math.atan2(dy, dx)  # -π to π
        hue = (angle + math.pi) / (2 * math.pi)  # normalize to 0-1

        # Distance from center → 0 (center) to 1 (edge)
        dist_ratio = math.hypot(dx, dy) / max_dist

        # Apply ramp curve: values < 1 reach full saturation quicker,
        # values > 1 stay white longer before ramping up
        dist_ratio = min(dist_ratio ** ramp, 1.0)

        # Modulate: center = white (low saturation, high lightness)
        #           edge   = full color (target saturation and lightness)
        tile_saturation = dist_ratio * saturation
        tile_lightness = 1.0 - dist_ratio * (1.0 - lightness)

        recolored = recolor_white_pixels(img, hue, tile_saturation, tile_lightness, threshold)
        canvas.paste(recolored, (px, py), recolored)

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Fill a grid with repeated images, recolored by position on the HSL color wheel."
    )
    parser.add_argument("--input", required=True, help="Folder containing source PNG files")
    parser.add_argument("--output", default="color_grid.png", help="Output image path")
    parser.add_argument("--mode", choices=["rect", "hex", "iso"], required=True, help="Layout mode")
    parser.add_argument("--cols", type=int, default=5, help="Columns (rect/iso mode)")
    parser.add_argument("--rows", type=int, default=3, help="Rows (rect/iso mode)")
    parser.add_argument("--radius", type=int, default=3, help="Hex radius (hex mode)")
    parser.add_argument("--padding", type=int, default=4, help="Pixels between tiles")
    parser.add_argument("--tile-size", type=int, default=0, help="Force tile size in px (0 = use image size)")
    parser.add_argument("--bg", default="#00000000", help="Background color as #RRGGBB or #RRGGBBAA")
    parser.add_argument("--pointy", action="store_true", help="Hex mode: pointy-top orientation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible results")
    parser.add_argument("--saturation", type=float, default=0.8, help="HSL saturation (0.0-1.0, default: 0.8)")
    parser.add_argument("--lightness", type=float, default=0.55, help="HSL lightness (0.0-1.0, default: 0.55)")
    parser.add_argument("--threshold", type=int, default=200, help="Brightness threshold for 'white' pixels (0-255, default: 200)")
    parser.add_argument("--ramp", type=float, default=0.5, help="Saturation ramp curve power (default: 0.5). Lower = saturates quicker, higher = stays white longer. 1.0 = linear")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    bg_color = parse_color(args.bg)
    source_images = load_images(args.input)
    print(f"Loaded {len(source_images)} source images from '{args.input}'")

    tile_size = args.tile_size if args.tile_size > 0 else source_images[0][1].width
    source_images = make_uniform(source_images, tile_size)

    if args.mode == "rect":
        positions, cw, ch = rect_positions(args.cols, args.rows, tile_size, args.padding)
        capacity = args.cols * args.rows
    elif args.mode == "iso":
        positions, cw, ch = iso_positions(args.cols, args.rows, tile_size, args.padding)
        capacity = args.cols * args.rows
    else:
        positions, cw, ch = hex_positions(args.radius, tile_size, args.padding, args.pointy)
        capacity = len(positions)

    print(f"Grid has {capacity} slots, filling with {len(source_images)} source images repeated randomly")

    result = compose(source_images, positions, cw, ch, bg_color, args.saturation, args.lightness, args.threshold, args.ramp)

    result.save(args.output)
    print(f"Saved {result.width}x{result.height} grid to '{args.output}'")


if __name__ == "__main__":
    main()