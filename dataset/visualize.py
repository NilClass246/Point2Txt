import open3d as o3d
import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

def visualize_pointcloud_o3d(pc):
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])

def visualize_pointcloud(pc, title="Point cloud", max_points=5000):
    """
    pc: torch.Tensor or np.ndarray of shape (N, 3)
    """
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()

    # Optionally subsample if it's huge
    N = pc.shape[0]
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        pc = pc[idx]

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=1)  # s = point size

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.show()