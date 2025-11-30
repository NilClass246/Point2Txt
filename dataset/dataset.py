import json
import torch
import pandas as pd
import os
import random
from torch.utils.data import Dataset, IterableDataset
from typing import List, Tuple
from collections import defaultdict

def fix_pointcloud(pc: torch.Tensor, target_n=8192):
    pc = torch.nan_to_num(pc, nan=0.0)
    pc[:, :3] = torch.clamp(pc[:, :3], -100, 100)
    N = pc.shape[0]
    if N == 0:
        pc = torch.zeros((target_n, pc.shape[1]), dtype=pc.dtype)
        N = target_n
    if N != target_n:
        idx = torch.randint(0, N, (target_n,), device=pc.device)
        pc = pc[idx]
    return pc

class Cap3DShapeNetPreprocessed(Dataset):
    def __init__(self, points_path, ids_path, csv_path, device, transform=None):
        self.pointclouds = torch.load(points_path)
        self.ids = json.load(open(ids_path))
        self.transform = transform
        self.device = device

        df = pd.read_csv(csv_path, names=["id", "caption"])
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

class ObjaverseStreamingDataset(IterableDataset):
    def __init__(self, point_map_path, id_map_path, csv_path, device, 
                 buffer_size=1, split='train', val_ratio=0.1):
        """
        split: 'train', 'val', or 'all'
        val_ratio: Fraction of chunks to reserve for validation (e.g., 0.1 = 10%)
        """
        self.device = device
        self.buffer_size = buffer_size
        
        print(f"Loading metadata for split: {split}...")
        
        with open(point_map_path, 'r') as f:
            self.chunk_paths = json.load(f)
            
        with open(id_map_path, 'r') as f:
            records = json.load(f)

        df = pd.read_csv(csv_path, names=["id", "caption"], on_bad_lines='skip')
        self.cap_dict = dict(zip(df["id"].astype(str), df["caption"].astype(str)))

        self.chunk_metadata = defaultdict(list)
        for oid, info in records.items():
            chunk_name = info['chunk']
            idx = info['index']
            if oid in self.cap_dict:
                self.chunk_metadata[chunk_name].append((idx, oid))
        
        for c in self.chunk_metadata:
            self.chunk_metadata[c].sort(key=lambda x: x[0])

        all_chunks = list(self.chunk_paths.keys())
        valid_chunks = [
            c for c in all_chunks 
            if c in self.chunk_metadata and len(self.chunk_metadata[c]) > 0
        ]
        
        valid_chunks.sort() 
        
        num_val = int(len(valid_chunks) * val_ratio)
        num_train = len(valid_chunks) - num_val
        
        if split == 'train':
            self.available_chunks = valid_chunks[:num_train]
        elif split == 'val':
            self.available_chunks = valid_chunks[num_train:]
        else:
            self.available_chunks = valid_chunks
            
        self.total_samples = 0
        for c in self.available_chunks:
            self.total_samples += len(self.chunk_metadata[c])
            
        print(f"[{split.upper()}] Dataset ready. Using {len(self.available_chunks)} chunks ({self.total_samples} samples).")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            my_chunks = list(self.available_chunks)
        else:
            my_chunks = [
                c for i, c in enumerate(self.available_chunks) 
                if i % worker_info.num_workers == worker_info.id
            ]
            
        random.shuffle(my_chunks)
        
        for chunk_name in my_chunks:
            dir_path = self.chunk_paths[chunk_name]
            full_path = os.path.join(dir_path, chunk_name)
            
            if not os.path.exists(full_path):
                continue
                
            try:
                chunk_tensor = torch.load(full_path, map_location='cpu', weights_only=True)
                metadata = self.chunk_metadata[chunk_name]
                
                local_indices = list(range(len(metadata)))
                random.shuffle(local_indices)
                
                for i in local_indices:
                    tensor_idx, oid = metadata[i]
                    if tensor_idx >= len(chunk_tensor): continue
                    
                    pc = chunk_tensor[tensor_idx]
                    caption = self.cap_dict[oid]
                    yield pc, caption
                    
            except Exception as e:
                print(f"Error reading chunk {chunk_name}: {e}")
                continue

def get_collate_fn(tokenizer, device, max_length=32):
    def collate_fn(batch: List[Tuple[torch.Tensor, str]]):
        """
        Collate function to:
        - stack point clouds
        - tokenize captions
        """
        pts_list, captions = zip(*batch)

        pts_batch = torch.stack(pts_list, dim=0).float() 

        enc = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return pts_batch, input_ids, attention_mask, labels, captions

    return collate_fn