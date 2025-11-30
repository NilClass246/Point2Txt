import torch
import torch.nn as nn
from . import misc

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            output: B G M 3
            center : B G 3
        '''
        if xyz.dtype != torch.float32:
            xyz = xyz.float()
        
        if torch.isnan(xyz).any() or torch.isinf(xyz).any():
            xyz = torch.nan_to_num(xyz, nan=0.0, posinf=1.0, neginf=-1.0)
        
        xyz = torch.clamp(xyz, min=-100.0, max=100.0)

        B, N, C = xyz.shape
        if C > 3:
            data = xyz
            xyz = data[:,:,:3].contiguous()
            rgb = data[:, :, 3:].contiguous()
        else:
            if not xyz.is_contiguous():
                xyz = xyz.contiguous()
        
        batch_size, num_points, _ = xyz.shape
        
        # FPS
        center = misc.fps(xyz, self.num_group) # B G 3

        # KNN
        idx = misc.knn_point(self.group_size, xyz, center) # B G M
        
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood_xyz = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood_xyz = neighborhood_xyz.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        
        if C > 3:
            neighborhood_rgb = rgb.view(batch_size * num_points, -1)[idx, :]
            neighborhood_rgb = neighborhood_rgb.view(batch_size, self.num_group, self.group_size, -1).contiguous()

        neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)
        
        if C > 3:
            neighborhood = torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)
        else:
            neighborhood = neighborhood_xyz
            
        return neighborhood, center

class Encoder(nn.Module):
    def __init__(self, encoder_channel, point_input_dims=3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.point_input_dims = point_input_dims
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.point_input_dims, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            feature_global : B G C
        '''
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        
        return feature_global.reshape(bs, g, self.encoder_channel)