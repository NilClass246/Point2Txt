import torch
import open3d as o3d
import numpy as np
import argparse
import os
from os import path

from transformers import GPT2TokenizerFast
from dataset.dataset import Cap3DShapeNetPreprocessed
from dataset.visualize import visualize_pointcloud_o3d
from models.point2txt import Point2Txt
from models.llm import load_gpt2
from models.encoder import load_point_encoder
from models.pointbert.misc import fps

def parse_args():
    parser = argparse.ArgumentParser(description="Point2Text Inference")
    
    # Model Paths
    parser.add_argument("--config_path", type=str, default="models/pointbert/PointTransformer_8192point_2layer.yaml", help="Path to PointBERT config")
    parser.add_argument("--encoder_ckpt", type=str, default="models/pointbert/point_bert_v1.2.pt", help="Path to PointBERT checkpoint")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to trained Point2Txt weights")
    
    # Data Paths
    parser.add_argument("--data_path", type=str, default="data/shapenet", help="Root directory for ShapeNet data")
    
    # Input Selection
    parser.add_argument("--test_idx", type=int, default=100, help="Index from the dataset to test")
    parser.add_argument("--ply_path", type=str, default=None, help="Path to a specific .ply file (overrides dataset index)")
    
    # Generation Parameters
    parser.add_argument("--max_len", type=int, default=30, help="Max new tokens to generate")
    parser.add_argument("--beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy/beam search")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling")
    
    # Misc
    parser.add_argument("--no_vis", action="store_true", help="Disable Open3D visualization")

    return parser.parse_args()

@torch.no_grad()
def generate_caption_from_points(
    model: Point2Txt,
    tokenizer: GPT2TokenizerFast,
    pts: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 30,
    num_beams: int = 5,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    """
    Robust generation function for CLIPCap/Prefix-tuning style models.
    """
    model.eval()

    # 1. Prepare Points: Ensure (1, N, C) shape
    if pts.dim() == 2:
        pts = pts.unsqueeze(0)
    pts = pts.to(device)
    
    # 2. Encode Prefix (Soft Prompts)
    prefix_embeds = model.encode_prefix(pts) # (1, prefix_len, hidden_dim)
    
    # 3. Create Attention Mask
    attention_mask = torch.ones(
        (prefix_embeds.shape[0], prefix_embeds.shape[1]), 
        dtype=torch.long, 
        device=device
    )

    # 4. Generate
    output_ids = model.gpt2.generate(
        inputs_embeds=prefix_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        use_cache=True, 
        early_stopping=(num_beams > 1)
    )

    # 5. Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text.strip()

def load_custom_ply(ply_path: str, device: torch.device) -> torch.Tensor:
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
        
    pcd = o3d.io.read_point_cloud(ply_path)
    points = torch.from_numpy(np.asarray(pcd.points)).float()
    
    # Handle colors
    if pcd.has_colors():
        colors = torch.from_numpy(np.asarray(pcd.colors)).float()
    else:
        print("Warning: Point cloud has no colors. Padding with zeros.")
        colors = torch.zeros_like(points)

    # Normalize to unit sphere (matching ShapeNet preprocessing)
    centroid = points.mean(dim=0)
    points -= centroid
    max_dist = torch.max(torch.norm(points, dim=1))
    if max_dist > 0:
        points /= max_dist

    # Combine points and colors (N, 6)
    colored_points = torch.cat([points, colors], dim=1) 
    
    target_points = 8192
    if colored_points.shape[0] > target_points:
        print(f"Downsampling from {colored_points.shape[0]} to {target_points} using FPS...")
        pts_batch = colored_points.unsqueeze(0).to(device) 
        sampled_pts = fps(pts_batch, target_points) # (1, 8192, 6)
        colored_points = sampled_pts.squeeze(0).cpu()
    
    return colored_points

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Models ---
    print("Loading Point Encoder...")
    point_encoder, backbone_output_dim = load_point_encoder(
        config_path=args.config_path,
        ckpt_path=args.encoder_ckpt,
        device=device
    )
    
    print("Loading GPT-2...")
    gpt2, tokenizer = load_gpt2(device)
    
    # Initialize Wrapper
    model = Point2Txt(point_encoder, gpt2, backbone_output_dim=backbone_output_dim, prefix_len=10).to(device)

    # Load Trained Weights
    if path.exists(args.model_path):
        try:
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(args.model_path, map_location=device)
            
        model.load_state_dict(checkpoint)
        print(f"Pretrained weights loaded from {args.model_path}.")
    else:
        print(f"Warning: {args.model_path} not found. Running with random weights.")

    # --- Prepare Input Data ---
    pts = None
    gt_caption = "N/A"

    if args.ply_path:
        # Mode A: Custom File
        print(f"Loading custom file: {args.ply_path}")
        pts = load_custom_ply(args.ply_path, device)
        gt_caption = "(Custom PLY - No GT)"
    else:
        # Mode B: Dataset
        print(f"Loading dataset from {args.data_path}...")
        try:
            dataset = Cap3DShapeNetPreprocessed(
                points_path=path.join(args.data_path, "processed_points.pt"),
                ids_path=path.join(args.data_path, "point_ids.json"),
                csv_path=path.join(args.data_path, "Cap3D_automated_ShapeNet.csv"),
                device=torch.device("cpu"),
            )
            
            idx = args.test_idx
            if idx >= len(dataset):
                print(f"Index {idx} out of bounds. Using index 0.")
                idx = 0
                
            pts, gt_caption = dataset[idx]
            print(f"Loaded sample {idx}.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

    # --- Generate ---
    print(f"\nGround Truth: {gt_caption}")
    
    generated_text = generate_caption_from_points(
        model, tokenizer, pts, 
        device=device,
        max_new_tokens=args.max_len,
        num_beams=args.beams,
        do_sample=args.do_sample,
        temperature=args.temp,
        top_k=args.top_k
    )
    
    print(f"Generated:    {generated_text}")

    # --- Visualization ---
    if not args.no_vis:
        print("Visualizing... (Close window to exit)")
        visualize_pointcloud_o3d(pts)

if __name__ == "__main__":
    main()