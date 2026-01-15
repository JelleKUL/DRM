import numpy as np
import matplotlib.pyplot as plt

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
