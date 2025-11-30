import torch
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.point2txt import Point2Txt
from models.llm import load_llm
from models.encoder import load_point_encoder
from dataset.dataset import Cap3DShapeNetPreprocessed

def set_seed(seed=42):
    """Ensure deterministic selection and generation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic Qualitative Generator")
    
    # Model Architecture Args
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--config_path", type=str, default="models/pointbert/PointTransformer_8192point_2layer.yaml")
    parser.add_argument("--encoder_ckpt", type=str, default="models/pointbert/point_bert_v1.2.pt")
    parser.add_argument("--llm_name", type=str, default="gpt2")
    parser.add_argument("--mapper_type", type=str, default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--mapper_layers", type=int, default=2)
    parser.add_argument("--mapper_heads", type=int, default=8)
    parser.add_argument("--prefix_len", type=int, default=10)
    
    # Generation Args
    parser.add_argument("--max_len", type=int, default=100, help="Max new tokens for generation")
    
    # Data Args
    parser.add_argument("--dataset_type", type=str, default="objaverse", choices=["shapenet", "objaverse"])
    parser.add_argument("--data_path", type=str, default="data/objaverse/validation/test_set") 
    parser.add_argument("--json_path", type=str, default="data/objaverse/validation/test_set_captions.json")
    
    # Output Args
    parser.add_argument("--output_dir", type=str, default="qualitative_results")
    
    return parser.parse_args()

@torch.no_grad()
def generate_caption(model, tokenizer, pts, device, max_new_tokens=30):
    model.eval()
    if pts.dim() == 2:
        pts = pts.unsqueeze(0) # (1, N, 6)
    pts = pts.to(device)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.amp.autocast('cuda', dtype=dtype):
        prefix_embeds = model.encode_prefix(pts)
        
        output_ids = model.llm.generate(
            inputs_embeds=prefix_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1, 
            do_sample=False,
            repetition_penalty=1.2
        )
        
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def save_dual_view(points, idx, output_dir):
    """
    Saves a clean PNG with two viewing angles of the point cloud.
    Background: Grey.
    Points: White (or original color).
    No Text.
    """
    bg_color = "#515151" # Dark Grey
    fig = plt.figure(figsize=(10, 5))
    fig.patch.set_facecolor(bg_color)
    
    if points.shape[0] > 8192:
        choice = np.random.choice(points.shape[0], 8192, replace=False)
        points = points[choice]

    x, y, z = points[:, 0], points[:, 2], points[:, 1]
    
    if points.shape[1] >= 6:
        colors = points[:, 3:6]
        colors = np.clip(colors, 0, 1)
    else:
        colors = 'white' 

    def setup_axis(ax, azim):
        ax.set_facecolor(bg_color)
        ax.xaxis.set_pane_color(bg_color)
        ax.yaxis.set_pane_color(bg_color)
        ax.zaxis.set_pane_color(bg_color)
        ax.grid(False)
        ax.axis('off')
        ax.view_init(elev=20, azim=azim)

    # --- View 1 (Front/Iso) ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    setup_axis(ax1, 45)
    ax1.scatter(x, y, z, c=colors, s=1.0, alpha=0.9, linewidth=0)

    # --- View 2 (Side/Back) ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    setup_axis(ax2, 135) # Rotated 90 degrees
    ax2.scatter(x, y, z, c=colors, s=1.0, alpha=0.9, linewidth=0)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
    
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=200, facecolor=bg_color)
    plt.close()
    return save_path

def main():
    args = parse_args()
    set_seed(42) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")

    print(f"Loading Encoder...")
    point_encoder, backbone_output_dim = load_point_encoder(args.config_path, args.encoder_ckpt, device)
    
    print(f"Loading LLM...")
    llm, tokenizer = load_llm(args.llm_name, device, freeze=True)
    
    model = Point2Txt(
        point_encoder=point_encoder,
        llm=llm,
        backbone_output_dim=backbone_output_dim,
        prefix_len=args.prefix_len,
        mapper_type=args.mapper_type,
        mapper_layers=args.mapper_layers,
        mapper_heads=args.mapper_heads
    ).to(device)

    print(f"Loading Weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    print(f"Loading Dataset...")
    if args.dataset_type == "objaverse":
        from test import ObjaverseDataset 
        dataset = ObjaverseDataset(npy_dir=args.data_path, json_path=args.json_path)
    else:
        dataset = Cap3DShapeNetPreprocessed(
            points_path=os.path.join(args.data_path, "processed_points.pt"),
            ids_path=os.path.join(args.data_path, "point_ids.json"),
            csv_path=os.path.join(args.data_path, "Cap3D_automated_ShapeNet.csv"),
            device=torch.device("cpu")
        )

    total_samples = len(dataset)
    indices = np.linspace(0, total_samples - 1, 6, dtype=int)
    
    results_txt_path = os.path.join(args.output_dir, "results_summary.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating visualizations for indices: {indices}")

    with open(results_txt_path, "w") as f:
        f.write(f"Model: {args.model_path}\n\n")
        
        for idx in tqdm(indices):
            data = dataset[idx]
            if len(data) == 5: 
                pts, _, _, uid, gt_caption = data
            else: 
                pts, gt_caption = data
                uid = str(idx)

            pred_caption = generate_caption(model, tokenizer, pts, device, max_new_tokens=args.max_len)
            
            save_dual_view(pts.numpy(), idx, args.output_dir)
            
            f.write(f"Index: {idx} | ID: {uid}\n")
            f.write(f"GT:   {gt_caption}\n")
            f.write(f"Pred: {pred_caption}\n")
            f.write("-" * 40 + "\n")

    print(f"\nDone! Visualizations (PNG) and Text (TXT) saved to: {args.output_dir}")

if __name__ == "__main__":
    main()