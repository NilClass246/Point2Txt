import torch
import argparse
import os
import math
import numpy as np
import random
import evaluate
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

from models.point2txt import Point2Txt
from models.llm import load_gpt2
from models.encoder import load_point_encoder
from dataset.dataset import Cap3DShapeNetPreprocessed, get_collate_fn

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--data_path", type=str, default="data/shapenet")
    parser.add_argument("--config_path", type=str, default="models/pointbert/PointTransformer_8192point_2layer.yaml")
    parser.add_argument("--ckpt_path", type=str, default="models/pointbert/point_bert_v1.2.pt")
    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc=f"Train Epoch {epoch}")
    
    for pts, input_ids, attention_mask, labels, _ in progress:
        pts = pts.to(device)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            labels[attention_mask == 0] = -100

        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(pts, input_ids, attention_mask, labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress.set_postfix({"loss": loss.item()})
        
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, tokenizer, device, max_gen_batches=5):
    model.eval()
    total_loss = 0
    
    try:
        bleu_metric = evaluate.load("bleu")
    except Exception:
        print("Warning: Could not load 'evaluate'. BLEU will be 0.")
        bleu_metric = None

    predictions = []
    references = []
    
    progress = tqdm(loader, desc="Validating")
    
    for batch_idx, (pts, input_ids, attention_mask, labels, raw_captions) in enumerate(progress):
        pts = pts.to(device)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            labels[attention_mask == 0] = -100

        # 1. Calculate Loss
        with autocast('cuda'):
            outputs = model(pts, input_ids, attention_mask, labels)
            total_loss += outputs.loss.item()

        # 2. Generate Captions
        if batch_idx < max_gen_batches:
            with autocast('cuda'):
                prefix_embeds = model.encode_prefix(pts)
                
                B, PL, H = prefix_embeds.shape
                gen_mask = torch.ones((B, PL), dtype=torch.long, device=device)
                
                generated_ids = model.gpt2.generate(
                    inputs_embeds=prefix_embeds,
                    attention_mask=gen_mask,
                    max_new_tokens=30,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=3,
                    early_stopping=True
                )
                
                gen_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                predictions.extend([t.strip() for t in gen_texts])
                references.extend([[r] for r in raw_captions])

                if batch_idx == 0:
                    print(f"\n[Sanity Check] GT:   {raw_captions[0]}")
                    print(f"[Sanity Check] Pred: {gen_texts[0]}\n")

    avg_loss = total_loss / len(loader)
    bleu_score = 0.0
    
    if bleu_metric and predictions:
        results = bleu_metric.compute(predictions=predictions, references=references)
        bleu_score = results['bleu']
        
    return avg_loss, bleu_score

def main():
    args = parse_args()
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # --- 1. Models ---
    point_encoder, backbone_output_dim = load_point_encoder(args.config_path, args.ckpt_path, device)
    gpt2, tokenizer = load_gpt2(device)
    
    model = Point2Txt(point_encoder, gpt2, backbone_output_dim, prefix_len=10).to(device)

    # --- 2. Data ---
    full_dataset = Cap3DShapeNetPreprocessed(
        points_path=os.path.join(args.data_path, "processed_points.pt"), 
        ids_path=os.path.join(args.data_path, "point_ids.json"),
        csv_path=os.path.join(args.data_path, "Cap3D_automated_ShapeNet.csv"),
        device=torch.device("cpu")
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    collate_fn = get_collate_fn(tokenizer, device)
    
    # --- 3. Loaders ---
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=0
    )
    
    val_batch_size = 32 
    val_loader = DataLoader(
        val_set, 
        batch_size=val_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=0
    )

    # --- 4. Optimization ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = GradScaler('cuda') 

    best_val_bleu = 0.0

    # --- 5. Loop ---
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        
        # Clear VRAM before validation starts
        torch.cuda.empty_cache() 
        
        val_loss, val_bleu = validate(model, val_loader, tokenizer, device)
        
        scheduler.step()
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.4f}")

        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"Saved best model (BLEU: {val_bleu:.4f})")
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model.pth"))

if __name__ == "__main__":
    main()