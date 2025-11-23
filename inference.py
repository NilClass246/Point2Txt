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
    temperature: float = 1.0,
    top_k: int = 0,
) -> str:
    """
    pts: (N, C) tensor on CPU
    """
    model.eval()

    pts = pts.unsqueeze(0).to(device)  # (1, N, C)
    prefix = model.encode_prefix(pts)  # (1, prefix_len, H)

    # Start with BOS token
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    generated = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Token embeddings for current generated sequence
        token_embeds = model.gpt2.transformer.wte(generated)  # (1, t, H)

        # Concatenate prefix + tokens
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)  # (1, prefix_len + t, H)

        outputs = model.gpt2(inputs_embeds=inputs_embeds)
        next_token_logits = outputs.logits[:, -1, :]  # (1, vocab)

        # Optionally apply temperature & top-k
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        if top_k > 0:
            values, indices = torch.topk(next_token_logits, top_k)
            probs = torch.softmax(values, dim=-1)
            next_token = indices[0, torch.multinomial(probs[0], num_samples=1)]
            next_token = next_token.unsqueeze(0).unsqueeze(0)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    # Drop BOS, decode
    caption_ids = generated[0, 1:]
    caption = tokenizer.decode(caption_ids, skip_special_tokens=True)
    return caption.strip()

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
    checkpoint = torch.load("checkpoints/test_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    print("Pretrained weights loaded.")

    test_pts, test_caption = dataset[10000]
    print("Ground truth caption:", test_caption)

    gen_caption = generate_caption_from_points(model, tokenizer, test_pts, device=device, max_new_tokens=20)
    print("Generated caption:", gen_caption)

    visualize_pointcloud_o3d(test_pts)

if __name__ == "__main__":
    main()