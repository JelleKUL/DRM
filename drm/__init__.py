import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import trimesh
import random
from scipy.spatial import ConvexHull, Delaunay,cKDTree


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


def txt_pcd_to_bin(txtPath, binPath = "", rotateX = False, containsNormals = False):
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
    Generate a 2D orthographic image of a plane with original colors,
    enlarged points for visual density, and a mask highlighting filled points.
    
    Parameters:
        original_pc (trimesh.points.PointCloud): Original plane points with colors.
        filled_pc (trimesh.points.PointCloud): Plane after filling holes.
        resolution (int): Output image resolution (HxW).
        point_radius (int): Radius in pixels to draw each point.
        
    Returns:
        img (np.ndarray): HxWx3 image with colored points enlarged.
        mask (np.ndarray): HxW binary mask of filled points.
    """
    points = original_pc.vertices
    colors = original_pc.colors.astype(np.uint8) if original_pc.colors is not None else np.full((len(points),3),255,dtype=np.uint8)
    filled_points = filled_pc.vertices

    # 1. PCA for plane alignment
    centroid = points.mean(axis=0)
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    u = eigvecs[:,2]
    v = eigvecs[:,1]

    # 2. Project to 2D
    points_2d = np.dot(points - centroid, np.column_stack((u,v)))
    filled_2d = np.dot(filled_points - centroid, np.column_stack((u,v)))

    # 3. Convex hull bounding box for scaling
    hull = ConvexHull(filled_2d)
    min_xy = filled_2d.min(axis=0)
    max_xy = filled_2d.max(axis=0)

    scale = resolution / (max_xy - min_xy)
    scale_factor = np.min(scale)

    def to_img_coords(p):
        return np.clip(((p - min_xy) * scale_factor).astype(int), 0, resolution-1)

    orig_coords = to_img_coords(points_2d)
    filled_coords = to_img_coords(filled_2d)

    # 4. Initialize image and mask
    img = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    mask = np.zeros((resolution, resolution), dtype=np.uint8)

    # 5. Function to draw a square around a point
    def draw_point(x, y, img_array, color):
        x_min = max(x - point_radius, 0)
        x_max = min(x + point_radius + 1, resolution)
        y_min = max(y - point_radius, 0)
        y_max = min(y + point_radius + 1, resolution)
        img_array[y_min:y_max, x_min:x_max] = color

    # 6. Draw original points with color
    for idx, (x,y) in enumerate(orig_coords):
        draw_point(x, y, img, colors[idx])

    # 7. Determine holes (new points in filled but not original)
    orig_kdtree = cKDTree(points)
    dists, _ = orig_kdtree.query(filled_points)
    hole_mask = dists > 0.05

    # 8. Draw holes in mask
    for idx, is_hole in enumerate(hole_mask):
        if is_hole:
            x, y = filled_coords[idx]
            draw_point(x, y, mask, 1)

    return img, mask,filled_coords
