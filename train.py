import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

import argparse
import os
import numpy as np
import random
import evaluate
from tqdm import tqdm
import json

import wandb

from models.point2txt import Point2Txt
from models.llm import load_llm
from models.encoder import load_point_encoder
from dataset.dataset import Cap3DShapeNetPreprocessed, get_collate_fn, ObjaverseStreamingDataset

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # --- Training Hyperparams ---
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    # --- Data ---
    parser.add_argument("--dataset", type=str, default="objaverse", choices=["shapenet", "objaverse"])
    parser.add_argument("--data_path", type=str, default="data/objaverse")
    
    # --- Encoder ---
    parser.add_argument("--config_path", type=str, default="models/pointbert/PointTransformer_8192point_2layer.yaml")
    parser.add_argument("--ckpt_path", type=str, default="models/pointbert/point_bert_v1.2.pt")
    
    # --- Experiment Config (Naming) ---
    # 1.1 Max Length (Token limit for tokenizer)
    parser.add_argument("--max_len", type=int, default=32, help="Max sequence length for caption tokenization")
    
    # 2. Mapper
    parser.add_argument("--mapper_type", type=str, default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--mapper_layers", type=int, default=4)
    parser.add_argument("--mapper_heads", type=int, default=8)
    parser.add_argument("--prefix_len", type=int, default=10)
    
    # 3. LLM
    parser.add_argument("--llm_name", type=str, default="gpt2", help="HF Model ID (e.g. 'gpt2', 'Qwen/Qwen2.5-3B-Instruct')")
    parser.add_argument("--freeze_llm", action="store_true", help="Freeze LLM weights")
    
    # Misc
    parser.add_argument("--wandb_project", type=str, default="Point2Text")
    
    return parser.parse_args()

def generate_run_name(args):
    """
    Generates run name: pointbert_mlp-l2-pl10_gpt2-ml32-finetune
    """
    enc_str = "pointbert"
    
    # 2. Mapper part
    if args.mapper_type == "mlp":
        map_str = f"mlp-l{args.mapper_layers}-pl{args.prefix_len}"
    else:
        map_str = f"transformer-l{args.mapper_layers}-h{args.mapper_heads}-pl{args.prefix_len}"
        
    # 3. LLM part
    llm_short = args.llm_name.split('/')[-1].split('-')[0].lower()
    train_status = "freeze" if args.freeze_llm else "finetune"
    
    llm_str = f"{llm_short}-ml{args.max_len}-{train_status}"
    
    return f"{enc_str}_{map_str}_{llm_str}"

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, total_steps=None):
    model.train()
    total_loss = 0
    batch_count = 0
    
    progress = tqdm(loader, desc=f"Train Epoch {epoch}", total=total_steps)
    
    for pts, input_ids, attention_mask, labels, _ in progress:
        pts = pts.to(device)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            labels[attention_mask == 0] = -100

        optimizer.zero_grad()
        
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(pts, input_ids, attention_mask, labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batch_count += 1
        progress.set_postfix({"loss": loss.item()})
        
    return total_loss / max(1, batch_count)

@torch.no_grad()
def validate(model, loader, tokenizer, device, epoch, max_gen_batches=3):
    model.eval()
    total_loss = 0
    batch_count = 0
    bleu_metric = evaluate.load("bleu")
    
    progress = tqdm(loader, desc="Validating")
    
    table_data = []

    for batch_idx, (pts, input_ids, attention_mask, labels, raw_captions) in enumerate(progress):
        pts = pts.to(device)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            labels[attention_mask == 0] = -100

        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(pts, input_ids, attention_mask, labels)
            total_loss += outputs.loss.item()
        
        batch_count += 1

        # Generate specific samples for WandB table
        if batch_idx < max_gen_batches:
            with autocast('cuda', dtype=torch.bfloat16):
                prefix_embeds = model.encode_prefix(pts)
                B, PL, H = prefix_embeds.shape
                gen_mask = torch.ones((B, PL), dtype=torch.long, device=device)
                
                # We need to call .generate on the underlying LLM
                generated_ids = model.llm.generate(
                    inputs_embeds=prefix_embeds,
                    attention_mask=gen_mask,
                    max_new_tokens=30,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=1,
                    do_sample=False
                )
                
                gen_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                predictions = [t.strip() for t in gen_texts]
                references = [[r] for r in raw_captions]

                bleu_metric.add_batch(predictions=predictions, references=references)

                for gt, pred in zip(raw_captions, predictions):
                    table_data.append([epoch, batch_idx, gt, pred])
        
    avg_loss = total_loss / max(1, batch_count)
    
    metrics = {"bleu1": 0.0}
    try:
        results = bleu_metric.compute(max_order=1) 
        metrics['bleu1'] = results['bleu']
    except:
        print("Warning: BLEU computation failed (maybe empty prediction)")
        
    return avg_loss, metrics, table_data

def main():
    args = parse_args()
    set_seed(42)
    
    run_name = generate_run_name(args)
    print(f"Starting Run: {run_name}")
    
    # Initialize WandB with custom run name
    wandb.init(project=args.wandb_project, name=run_name, config=args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Save dir specific to run
    run_save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_save_dir, exist_ok=True)

    print("Loading Point Encoder...")
    point_encoder, backbone_output_dim = load_point_encoder(args.config_path, args.ckpt_path, device)
    
    # Modified LLM Loader
    llm, tokenizer = load_llm(args.llm_name, device, freeze=args.freeze_llm)
    
    # Instantiate Model
    model = Point2Txt(
        point_encoder=point_encoder, 
        llm=llm, 
        backbone_output_dim=backbone_output_dim, 
        prefix_len=args.prefix_len,
        mapper_type=args.mapper_type,
        mapper_layers=args.mapper_layers,
        mapper_heads=args.mapper_heads
    ).to(device)

    # Pass max_len to collate
    collate_fn = get_collate_fn(tokenizer, device, max_length=args.max_len)

    print("Loading Dataset...")
    if args.dataset == "shapenet":
        full_dataset = Cap3DShapeNetPreprocessed(
            points_path=os.path.join(args.data_path, "processed_points.pt"), 
            ids_path=os.path.join(args.data_path, "point_ids.json"),
            csv_path=os.path.join(args.data_path, "Cap3D_automated_ShapeNet.csv"),
            device=torch.device("cpu")
        )
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    else:
        dataset_args = dict(
            point_map_path=os.path.join(args.data_path, "chunk_paths.json"),
            id_map_path=os.path.join(args.data_path, "chunk_records.json"),
            csv_path=os.path.join(args.data_path, "Cap3D_automated_Objaverse_full.csv"),
            device=torch.device("cpu"),
            val_ratio=0.1
        )

        train_dataset = ObjaverseStreamingDataset(**dataset_args, split='train')
        val_dataset = ObjaverseStreamingDataset(**dataset_args, split='val')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=2, 
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=1,
            pin_memory=True
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = GradScaler('cuda', enabled=False) 

    best_val_score = 0.0

    for epoch in range(args.epochs):
        if isinstance(train_loader.dataset, ObjaverseStreamingDataset):
            total_steps = train_loader.dataset.total_samples // train_loader.batch_size
        else:
            total_steps = len(train_loader)
            
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, total_steps=total_steps)
        
        torch.cuda.empty_cache() 
        
        val_loss, metrics, table_data = validate(model, val_loader, tokenizer, device, epoch)
        
        scheduler.step()
        
        print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | BLEU-1: {metrics['bleu1']:.4f}")

        # --- WandB Logging Fix ---
        # Log everything ONCE per epoch to keep steps aligned
        
        # 1. Log Metrics
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "bleu1": metrics['bleu1']
        }
        
        # 2. Log Table (limit rows to avoid bloat)
        columns = ["epoch", "batch_idx", "ground_truth", "prediction"]
        # Take first 10 rows for table
        log_dict["val_samples"] = wandb.Table(data=table_data[:10], columns=columns)
        
        wandb.log(log_dict)
        # -------------------------

        # Save Best Model
        if metrics['bleu1'] > best_val_score:
            best_val_score = metrics['bleu1']
            torch.save(model.state_dict(), os.path.join(run_save_dir, "best_model.pth"))
            print(f"Saved best model to {run_save_dir}")
        
        torch.save(model.state_dict(), os.path.join(run_save_dir, "last_model.pth"))

if __name__ == "__main__":
    main()