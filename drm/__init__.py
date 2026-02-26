import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import trimesh

def detect_objects(binPath, model="votenet",scoreThr = 0.3, outputDir="../_output"):
    demoFile = "../../mmdetection3d/demo/pcd_demo.py"
    configFile = "/home/jvermandere/projects/mmdetection3d/configs/votenet/votenet_8xb8_scannet-3d.py"
    #configFile = "../../mmdetection3d/configs/votenet/votenet_8xb16_sunrgbd-3d.py"
    weightsFile = "/home/jvermandere/projects/DRM/_weights/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth"
    #weightsFile = "/home/jvermandere/projects/DRM/_weights/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth"

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
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for r in rows:
            if len(r) != 9:
                raise ValueError(f"Expected 9 values per row, got {len(r)}")

            f.write(" ".join(r) + "\n")


def txt_pcd_to_bin(txtPath, binPath = "", rotateX = False):
    #if binPath is empty, save at the same location
    if(binPath == ""):
        binPath = (str)(Path(txtPath).with_suffix(".bin"))
    # Load txt file
    # Assuming format: x y z nx ny nz r g b
    data = np.loadtxt(txtPath)

    # Keep only x, y, z, r, g, b for VoteNet/ScanNet format
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


