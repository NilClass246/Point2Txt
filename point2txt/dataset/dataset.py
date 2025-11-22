import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple

class Cap3DShapeNetPreprocessed(Dataset):
    def __init__(self, points_path, ids_path, csv_path, device, transform=None):
        self.pointclouds = torch.load(points_path)  # list of (N,6) tensors
        self.ids = json.load(open(ids_path))
        self.transform = transform
        self.device = device

        df = pd.read_csv(csv_path, names=["id", "caption"])
        print(df.head())
        cap_dict = dict(zip(df["id"], df["caption"]))
        self.captions = [cap_dict[i] for i in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pc = self.pointclouds[idx].to(self.device)
        caption = self.captions[idx]

        if self.transform:
            pc = self.transform(pc)

        return pc, caption

class Cap3DObjaversePreprocessed(Dataset):
    def __init__(self, points_path, ids_path, csv_path, device):
        self.points_path = points_path
        self.device = device

        with open(ids_path, 'r') as f:
            data = json.load(f)
        self.ids = [item["object_id"] for item in data]
        

        df = pd.read_csv(csv_path, names=["id", "caption"])
        cap_dict = dict(zip(df["id"], df["caption"]))
        self.captions = [cap_dict[i] for i in self.ids]


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_path = f"{self.points_path}/{self.ids[idx]}_8192.npy"
        pc = np.load(file_path)
        pc = torch.from_numpy(pc).to(self.device)
        caption = self.captions[idx]

        return pc, caption

def get_collate_fn(tokenizer, device):
    def collate_fn(batch: List[Tuple[torch.Tensor, str]]):
        """
        Collate function to:
        - stack point clouds
        - tokenize captions
        """
        pts_list, captions = zip(*batch)

        # Stack point clouds -> (B, N, 6)
        pts_batch = torch.stack(pts_list, dim=0).float()

        # Tokenize captions
        enc = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Use input_ids as labels (standard LM training)
        labels = input_ids.clone()

        return pts_batch, input_ids, attention_mask, labels, captions

    return collate_fn