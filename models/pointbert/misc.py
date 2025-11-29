import torch
import numpy as np

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def fps(xyz, npoint):
    """
    Furthest Point Sampling
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        sampled_points: [B, npoint, 3]
    """
    with torch.amp.autocast('cuda', enabled=False):
        xyz = xyz.float()
        device = xyz.device
        B, N, C = xyz.shape
        
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            
            dist = torch.cdist(xyz, centroid, p=2).squeeze(-1) ** 2
            
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]
    
    return index_points(xyz, centroids) 

def square_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    with torch.amp.autocast('cuda', enabled=False):
        src = src.float()
        dst = dst.float()
        return torch.cdist(src, dst, p=2) ** 2

def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx