import yaml
import os

from models import PointTransformer
from easydict import EasyDict

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def load_point_encoder(config_path, ckpt_path, device):
    print(f"Loading PointBERT config from {config_path}.")
    point_bert_config = cfg_from_yaml_file(config_path)

    point_bert_config.model.point_dims = 6  # Use 6D points (XYZ + RGB)
    use_max_pool = False
    point_encoder = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool).to(device)
    print(f"Using {point_encoder.point_dims} dim of points.")

    point_encoder.load_checkpoint(ckpt_path)

    backbone_output_dim = point_bert_config.model.trans_dim
    print(f"Using {backbone_output_dim} output dim of points from PointBERT.")

    # freeze PointBERT parameters
    for param in point_encoder.parameters():
        param.requires_grad = False
    point_encoder.eval()

    return point_encoder, backbone_output_dim
