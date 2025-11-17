import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from visualize import visualize_pointcloud_o3d, visualize_pointcloud

class Cap3DShapeNetPreprocessed(Dataset):
    def __init__(self, points_path, ids_path, csv_path, transform=None):
        self.pointclouds = torch.load(points_path)  # list of (N,3) tensors
        self.ids = json.load(open(ids_path))
        self.transform = transform

        df = pd.read_csv(csv_path, names=["id", "caption"])
        print(df.head())
        cap_dict = dict(zip(df["id"], df["caption"]))
        self.captions = [cap_dict[i] for i in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pc = self.pointclouds[idx]
        caption = self.captions[idx]

        if self.transform:
            pc = self.transform(pc)

        return {
            "id": self.ids[idx],
            "points": pc,
            "caption": caption,
        }
    
if __name__ == "__main__":
    dataset = Cap3DShapeNetPreprocessed(
        points_path="data/processed_points.pt",
        ids_path="data/point_ids.json",
        csv_path="data/Cap3D_automated_ShapeNet.csv"
    )
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[1]
    print(f"Sample ID: {sample['id']}")
    print(f"Sample Points Shape: {sample['points'].shape}")
    print(f"Sample Caption: {sample['caption']}")

    # visualize
    # visualize_pointcloud_o3d(sample["points"])
    # visualize_pointcloud(sample["points"], title=sample["id"])