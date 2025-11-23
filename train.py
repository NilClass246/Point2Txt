import torch
import argparse
import os
import math
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.point2txt import Point2Txt
from models.llm import load_gpt2
from models.encoder import load_point_encoder
from datasets.dataset import Cap3DShapeNetPreprocessed, get_collate_fn

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--data_path", type=str, default="data/shapenet")
    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc=f"Train Epoch {epoch}")
    
    for pts, input_ids, attention_mask, labels, _ in progress:
        pts, input_ids, labels = pts.to(device), input_ids.to(device), labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        
        # Forward
        outputs = model(pts, input_ids, attention_mask, labels)
        loss = outputs.loss
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": loss.item()})
        
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    
    for pts, input_ids, attention_mask, labels, _ in tqdm(loader, desc="Validating"):
        pts, input_ids, labels = pts.to(device), input_ids.to(device), labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(pts, input_ids, attention_mask, labels)
        total_loss += outputs.loss.item()
        
    return total_loss / len(loader)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # --- 1. Models ---
    config_enc = "models/pointbert/PointTransformer_8192point_2layer.yaml"
    ckpt_enc = "models/pointbert/point_bert_v1.2.pt"
    
    point_encoder, backbone_output_dim = load_point_encoder(config_enc, ckpt_enc, device)
    gpt2, tokenizer = load_gpt2(device)
    
    model = Point2Txt(point_encoder, gpt2, backbone_output_dim, prefix_len=10).to(device)

    # --- 2. Data ---
    full_dataset = Cap3DShapeNetPreprocessed(
        points_path=os.path.join(args.data_path, "processed_points.pt"), 
        ids_path=os.path.join(args.data_path, "point_ids.json"),
        csv_path=os.path.join(args.data_path, "Cap3D_automated_ShapeNet.csv"),
        device=device
    )
    
    # Split Train/Val (90/10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    collate_fn = get_collate_fn(tokenizer, device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- 3. Optimization ---
    # Only train Mapper (and GPT2 if not frozen)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')

    # --- 4. Loop ---
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        
        scheduler.step()
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"Saved best model (Val Loss: {val_loss:.4f})")
        
        # Save Latest
        torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model.pth"))

if __name__ == "__main__":
    main()