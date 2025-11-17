import os
import json
import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm

def preprocess_ply_folder(
    pcs_dir,
    out_points_path="points.pt",
    out_ids_path="ids.json",
    normalize=True
):
    """
    Preprocesses all PLY files in a given directory by loading, normalizing,
    and saving them as a single tensor file along with their identifiers.

    Args:
        pcs_dir (str): Directory containing PLY files.
        out_points_path (str): Path to save the processed points tensor.
        out_ids_path (str): Path to save the identifiers JSON file.
        normalize (bool): Whether to normalize the point clouds.
    """
    all_points = []
    all_ids = []

    ply_files = [f for f in os.listdir(pcs_dir) if f.endswith('.ply')]
    
    for ply_file in tqdm(ply_files, desc="Processing PLY files"):
        file_path = os.path.join(pcs_dir, ply_file)
        pcd = o3d.io.read_point_cloud(file_path)
        points = torch.from_numpy(np.asarray(pcd.points)).float()

        if normalize:
            centroid = points.mean(dim=0)
            points -= centroid
            max_dist = torch.max(torch.norm(points, dim=1))
            points /= max_dist

        all_points.append(points)
        all_ids.append(os.path.splitext(ply_file)[0])

    # all_points_tensor = torch.cat(all_points, dim=0)
    torch.save(all_points, out_points_path)

    with open(out_ids_path, 'w') as f:
        json.dump(all_ids, f)

    print(f"Saved processed points to {out_points_path}")
    print(f"Saved identifiers to {out_ids_path}")

if __name__ == "__main__":
    pcs_directory = "ShapeNet_pcs"
    preprocess_ply_folder(
        pcs_dir=pcs_directory,
        out_points_path="data/processed_points.pt",
        out_ids_path="data/point_ids.json",
        normalize=True
    )
