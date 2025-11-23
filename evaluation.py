import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import evaluate
from sentence_transformers import SentenceTransformer, util

from transformers import GPT2TokenizerFast
from models.point2txt import Point2Txt
from models.llm import load_gpt2
from models.encoder import load_point_encoder
from dataset.dataset import Cap3DShapeNetPreprocessed, get_collate_fn
from os import path

def parse_args():
    parser = argparse.ArgumentParser(description="Point2Text Evaluation (Table 3 Metrics)")
    
    # Paths
    parser.add_argument("--config_path", type=str, default="models/pointbert/PointTransformer_8192point_2layer.yaml")
    parser.add_argument("--encoder_ckpt", type=str, default="models/pointbert/point_bert_v1.2.pt")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to trained weights")
    parser.add_argument("--data_path", type=str, default="data/shapenet")
    
    # Settings
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation (if supported) or dataloader")
    parser.add_argument("--num_samples", type=int, default=-1, help="Limit number of samples for quick testing (-1 for all)")
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--beam_size", type=int, default=1, help="Use 1 for greedy (faster), 3+ for better quality")
    
    return parser.parse_args()

@torch.no_grad()
def generate_batch(model, tokenizer, pts, device, max_len=30, beam_size=1):
    """
    Generates captions for a batch of point clouds.
    """
    model.eval()
    pts = pts.to(device) # (B, N, 6)
    
    # 1. Encode Prefix
    prefix_embeds = model.encode_prefix(pts) # (B, prefix_len, H)
    
    # 2. Create Mask
    B, PL, H = prefix_embeds.shape
    attention_mask = torch.ones((B, PL), dtype=torch.long, device=device)
    
    # 3. Generate
    output_ids = model.gpt2.generate(
        inputs_embeds=prefix_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_len,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=beam_size,
        do_sample=False,
        use_cache=True
    )
    
    # 4. Decode
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [c.strip() for c in captions]

def compute_semantic_similarity(model_name, preds, refs, device):
    """
    Computes average Cosine Similarity using a SentenceTransformer model.
    Used for both S-BERT and SimCSE scores.
    """
    print(f"Loading {model_name} for similarity metric...")
    st_model = SentenceTransformer(model_name, device=device)
    
    pred_embeddings = st_model.encode(preds, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
    ref_embeddings = st_model.encode(refs, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
    
    cosine_scores = util.pairwise_cos_sim(pred_embeddings, ref_embeddings)
    
    return torch.mean(cosine_scores).item()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    # --- 1. Load Models ---
    point_encoder, backbone_output_dim = load_point_encoder(args.config_path, args.encoder_ckpt, device)
    gpt2, tokenizer = load_gpt2(device)
    model = Point2Txt(point_encoder, gpt2, backbone_output_dim, prefix_len=10).to(device)

    if path.exists(args.model_path):
        try:
            ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
        except:
            ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Loaded weights from {args.model_path}")
    else:
        print(f"WARNING: No weights found at {args.model_path}, evaluating random model!")

    # --- 2. Load Data ---
    dataset = Cap3DShapeNetPreprocessed(
        points_path=path.join(args.data_path, "processed_points.pt"),
        ids_path=path.join(args.data_path, "point_ids.json"),
        csv_path=path.join(args.data_path, "Cap3D_automated_ShapeNet.csv"),
        device=torch.device("cpu"),
    )

    if args.num_samples > 0:
        indices = range(min(len(dataset), args.num_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    collate_fn = get_collate_fn(tokenizer, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- 3. Generation Loop ---
    predictions = []
    references = []
    
    print(f"Generating captions for {len(dataset)} samples...")
    for batch in tqdm(loader):
        pts, _, _, _, caps = batch
        
        batch_preds = generate_batch(model, tokenizer, pts, device, max_len=args.max_len, beam_size=args.beam_size)
        
        predictions.extend(batch_preds)
        references.extend(caps)

    print("\n--- Sample Generations ---")
    for i in range(3):
        print(f"GT:   {references[i]}")
        print(f"Pred: {predictions[i]}")
        print("-" * 20)

    # --- 4. Compute Metrics ---
    results = {}

    # A. Lexical Metrics (BLEU-1, ROUGE-L, METEOR)
    print("Computing Lexical Metrics...")
    
    # BLEU-1
    bleu = evaluate.load("bleu")
    # max_order=1 ensures we are calculating BLEU-1 (unigram overlap)
    b1_score = bleu.compute(predictions=predictions, references=[[r] for r in references], max_order=1)
    results['BLEU-1'] = b1_score['bleu']

    # ROUGE-L
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=predictions, references=references)
    results['ROUGE-L'] = rouge_score['rougeL']

    # METEOR
    meteor = evaluate.load("meteor")
    meteor_score = meteor.compute(predictions=predictions, references=references)
    results['METEOR'] = meteor_score['meteor']

    # B. Semantic Metrics (Sentence-BERT, SimCSE)
    
    # Sentence-BERT (Standard baseline: all-MiniLM-L6-v2)
    results['S-BERT'] = compute_semantic_similarity(
        'sentence-transformers/all-MiniLM-L6-v2', 
        predictions, 
        references, 
        device
    )

    # SimCSE (Supervised BERT Base is a common standard for this metric)
    # Using Princeton-NLP's supervised SimCSE model via SentenceTransformers
    results['SimCSE'] = compute_semantic_similarity(
        'princeton-nlp/sup-simcse-bert-base-uncased', 
        predictions, 
        references, 
        device
    )

    # --- 5. Print Table 3 Results ---
    print("\n" + "="*30)
    print("       Evaluation Results      ")
    print("="*30)
    print(f"S-BERT:   {results['S-BERT']:.4f}")
    print(f"SimCSE:   {results['SimCSE']:.4f}")
    print(f"BLEU-1:   {results['BLEU-1']:.4f}")
    print(f"ROUGE-L:  {results['ROUGE-L']:.4f}")
    print(f"METEOR:   {results['METEOR']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()