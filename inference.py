import torch
import open3d as o3d
import numpy as np

from os import path
from transformers import GPT2TokenizerFast
from datasets.dataset import Cap3DObjaversePreprocessed, Cap3DShapeNetPreprocessed
from datasets.visualize import visualize_pointcloud_o3d
from models.point2txt import Point2Txt
from models.llm import load_gpt2
from models.encoder import load_point_encoder

config = {
    "encoder_config_path": "models/pointbert/PointTransformer_8192point_2layer.yaml",
    "encoder_ckpt_path": "models/pointbert/point_bert_v1.2.pt",
    "data_path": "data/shapenet",
}

@torch.no_grad()
def generate_caption_from_points(
    model: Point2Txt,
    tokenizer: GPT2TokenizerFast,
    pts: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 30,
    num_beams: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> str:
    """
    Generates a caption from a point cloud using Hugging Face's generate() method.
    
    Args:
        model: The Point2Txt model.
        tokenizer: GPT2 tokenizer.
        pts: (N, C) tensor of point cloud data.
        device: Torch device.
        max_new_tokens: Maximum number of tokens to generate.
        num_beams: >1 enables beam search (better quality, slower).
        do_sample: True enables sampling (more diverse, less repetitive).
        temperature: Softmax temperature (lower = more deterministic).
        top_k: Filter top-k tokens before sampling.
        top_p: Nucleus sampling (filter by cumulative probability).
        repetition_penalty: >1.0 penalizes repetition.
    """
    model.eval()

    pts = pts.unsqueeze(0).to(device)  # (1, N, C)
    
    prefix_embeds = model.encode_prefix(pts)
    
    prefix_mask = torch.ones(
        (prefix_embeds.shape[0], prefix_embeds.shape[1]), 
        dtype=torch.long, 
        device=device
    )

    output_ids = model.gpt2.generate(
        inputs_embeds=prefix_embeds,
        attention_mask=prefix_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        use_cache=True, 
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text.strip()

def load_ply(ply_path: str) -> torch.Tensor:
    pcd = o3d.io.read_point_cloud(ply_path)
    points = torch.from_numpy(np.asarray(pcd.points)).float()
    colors = torch.from_numpy(np.asarray(pcd.colors)).float()

    # normalize
    centroid = points.mean(dim=0)
    points -= centroid
    max_dist = torch.max(torch.norm(points, dim=1))
    points /= max_dist

    colored_points = torch.cat([points, colors], dim=1)  # Combine points and colors
    
    return colored_points


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset = Cap3DObjaversePreprocessed(
    #     points_path="data/objaverse/eval_set",
    #     ids_path="data/objaverse/PointLLM_brief_description_val_3000_GT.json",
    #     csv_path="data/objaverse/Cap3D_automated_Objaverse_full.csv",
    #     device=device,
    # )

    dataset = Cap3DShapeNetPreprocessed(
        points_path=path.join(config["data_path"], "processed_points.pt"),
        ids_path=path.join(config["data_path"], "point_ids.json"),
        csv_path=path.join(config["data_path"], "Cap3D_automated_ShapeNet.csv"),
        device=device,
    )

    # load models
    point_encoder, backbone_output_dim = load_point_encoder(
        config_path=config["encoder_config_path"],
        ckpt_path=config["encoder_ckpt_path"],
        device=device
    )
    gpt2, tokenizer = load_gpt2(device)
    model = Point2Txt(point_encoder, gpt2, backbone_output_dim=backbone_output_dim, prefix_len=10).to(device)
    print("Model ready.")

    # Load pretrained weights
    ckpt_path = "checkpoints/test_model.pth"
    if path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Pretrained weights loaded from {ckpt_path}.")
    else:
        print(f"Warning: {ckpt_path} not found. Using random weights.")

    # Select a test sample
    test_idx = 10000
    if test_idx >= len(dataset):
        test_idx = 0 # Fallback
    
    test_pts, test_caption = dataset[test_idx]
    print("\nGround truth caption:", test_caption)

    # --- Inference Strategy 1: Beam Search (Best for quality/grammar) ---
    gen_caption_beam = generate_caption_from_points(
        model, tokenizer, test_pts, device=device, 
        max_new_tokens=100,
        num_beams=5,           # Beam search with 5 beams
        repetition_penalty=1.2 # Penalty for repeating words
    )
    print("Generated (Beam Search):", gen_caption_beam)

    # --- Inference Strategy 2: Sampling (Best for diversity) ---
    # gen_caption_sample = generate_caption_from_points(
    #     model, tokenizer, test_pts, device=device, 
    #     max_new_tokens=100,
    #     do_sample=True,        # Enable sampling
    #     temperature=0.8,       # Add some randomness
    #     top_k=50               # Limit vocabulary to top 50 tokens
    # )
    # print("Generated (Sampling):   ", gen_caption_sample)

    visualize_pointcloud_o3d(test_pts)

if __name__ == "__main__":
    main()