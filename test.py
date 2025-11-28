import torch
import torch.nn.functional as F
import argparse
import numpy as np
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import evaluate
from sentence_transformers import SentenceTransformer, models

from models.point2txt import Point2Txt
from models.llm import load_gpt2
from models.encoder import load_point_encoder
from dataset.dataset import Cap3DShapeNetPreprocessed, get_collate_fn 

class ObjaverseDataset(Dataset):
    def __init__(self, npy_dir, json_path, num_points=8192):
        """
        Args:
            npy_dir: Folder containing .npy files (e.g., .../eval_set)
            json_path: Path to .json file { "uid": "caption", ... }
            num_points: Number of points to sample/ensure.
        """
        self.npy_dir = npy_dir
        self.num_points = num_points
        
        print(f"Loading captions from {json_path}...")
        with open(json_path, 'r') as f:
            self.captions_map = json.load(f)
        
        self.valid_uids = []
        all_npy = set([f.split('.')[0] for f in os.listdir(npy_dir) if f.endswith('.npy')])
        
        for uid in self.captions_map.keys():
            if uid + "_8192" in all_npy:
                self.valid_uids.append(uid)
        
        print(f"Found {len(self.valid_uids)} valid samples (Matching NPY + JSON).")

    def __len__(self):
        return len(self.valid_uids)

    def __getitem__(self, idx):
        uid = self.valid_uids[idx]
        caption = self.captions_map[uid]
        npy_path = os.path.join(self.npy_dir, f"{uid}_8192.npy")
        
        try:
            points = np.load(npy_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {uid}: {e}")
            points = np.zeros((self.num_points, 6), dtype=np.float32)

        N = points.shape[0]
        if N != self.num_points:
            if N > self.num_points:
                choice = np.random.choice(N, self.num_points, replace=False)
                points = points[choice]
            else:
                choice = np.random.choice(N, self.num_points, replace=True)
                points = points[choice]

        return torch.from_numpy(points), None, None, uid, caption

def parse_args():
    parser = argparse.ArgumentParser(description="Point2Text Evaluation")
    
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="objaverse",
        choices=["shapenet", "objaverse"], 
        help="Choose 'shapenet' for .pt file or 'objaverse' for individual .npy files"
    )

    parser.add_argument("--config_path", type=str, default="models/pointbert/PointTransformer_8192point_2layer.yaml")
    parser.add_argument("--encoder_ckpt", type=str, default="models/pointbert/point_bert_v1.2.pt")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    
    parser.add_argument("--shapenet_data_path", type=str, default="data/shapenet")
    
    parser.add_argument("--objaverse_npy_path", type=str, default="data/objaverse/validation/eval_set")
    parser.add_argument("--objaverse_json_path", type=str, default="data/objaverse/validation/captions.json")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--beam_size", type=int, default=1)
    
    return parser.parse_args()

@torch.no_grad()
def generate_batch(model, tokenizer, pts, device, max_len=30, beam_size=1):
    model.eval()
    pts = pts.to(device)
    
    prefix_embeds = model.encode_prefix(pts)
    B, PL, H = prefix_embeds.shape
    attention_mask = torch.ones((B, PL), dtype=torch.long, device=device)
    
    output_ids = model.gpt2.generate(
        inputs_embeds=prefix_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_len,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=beam_size,
        
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        length_penalty=1.0,
        do_sample=False,
        use_cache=True
    )
    
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [c.strip() for c in captions]

def compute_semantic_similarity(model_name, preds, refs, device):
    print(f"Loading {model_name}...")
    
    try:
        if "simcse" in model_name.lower():
            word_embedding_model = models.Transformer(model_name, max_seq_length=100)
            
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(), 
                pooling_mode='cls'
            )
            
            st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
            print(f"-> Successfully loaded {model_name} with CLS Pooling.")
            
        else:
            st_model = SentenceTransformer(model_name, device=device)

    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return 0.0

    pred_embeddings = st_model.encode(preds, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
    ref_embeddings = st_model.encode(refs, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
    
    pred_norm = F.normalize(pred_embeddings, p=2, dim=1)
    ref_norm = F.normalize(ref_embeddings, p=2, dim=1)
    
    scores = (pred_norm * ref_norm).sum(dim=1)
    
    return scores.mean().item()

def custom_eval_collate_fn(batch):
    """
    Custom collate for evaluation that handles the 5-item tuple from ObjaverseDataset.
    Batch structure: [(points, None, None, uid, caption), ...]
    """
    transposed = list(zip(*batch))
    
    points = torch.stack(transposed[0])
    
    uids = list(transposed[3])
    captions = list(transposed[4])
    
    return points, None, None, uids, captions

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device} | Mode: {args.dataset_type}")

    point_encoder, backbone_output_dim = load_point_encoder(args.config_path, args.encoder_ckpt, device)
    gpt2, tokenizer = load_gpt2(device)
    model = Point2Txt(point_encoder, gpt2, backbone_output_dim, prefix_len=10).to(device)

    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Loaded weights from {args.model_path}")
    else:
        print(f"WARNING: No weights found at {args.model_path}, using random init.")

    if args.dataset_type == "shapenet":
        print("Initializing ShapeNet Dataset...")
        dataset = Cap3DShapeNetPreprocessed(
            points_path=os.path.join(args.shapenet_data_path, "processed_points.pt"),
            ids_path=os.path.join(args.shapenet_data_path, "point_ids.json"),
            csv_path=os.path.join(args.shapenet_data_path, "Cap3D_automated_ShapeNet.csv"),
            device=torch.device("cpu"),
        )
    else:
        print("Initializing Objaverse Dataset...")
        dataset = ObjaverseDataset(
            npy_dir=args.objaverse_npy_path,
            json_path=args.objaverse_json_path,
            num_points=8192
        )

    if args.num_samples > 0:
        indices = range(min(len(dataset), args.num_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    if args.dataset_type == "shapenet":
        collate_fn = get_collate_fn(tokenizer, device)
    else:
        collate_fn = custom_eval_collate_fn

    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=4
    )

    predictions = []
    references = []
    
    print(f"Generating captions for {len(dataset)} samples...")
    for batch in tqdm(loader):
        pts = batch[0]
        caps = batch[4]
        
        batch_preds = generate_batch(model, tokenizer, pts, device, max_len=args.max_len, beam_size=args.beam_size)
        
        predictions.extend(batch_preds)
        references.extend(caps)

    print("\n--- Sample Generations ---")
    for i in range(min(3, len(predictions))):
        print(f"GT:   {references[i]}")
        print(f"Pred: {predictions[i]}")
        print("-" * 20)

    results = {}

    print("Computing Lexical Metrics...")
    bleu = evaluate.load("bleu")
    b1_score = bleu.compute(predictions=predictions, references=[[r] for r in references], max_order=1)
    results['BLEU-1'] = b1_score['bleu']

    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=predictions, references=references)
    results['ROUGE-L'] = rouge_score['rougeL']

    meteor = evaluate.load("meteor")
    meteor_score = meteor.compute(predictions=predictions, references=references)
    results['METEOR'] = meteor_score['meteor']

    print("Computing Semantic Metrics...")
    
    results['S-BERT'] = compute_semantic_similarity(
        'sentence-transformers/all-MiniLM-L6-v2', 
        predictions, references, device
    )

    results['SimCSE'] = compute_semantic_similarity(
        'princeton-nlp/sup-simcse-bert-base-uncased', 
        predictions, references, device
    )

    print("\n" + "="*30)
    print(f"  Evaluation Results ({args.dataset_type})  ")
    print("="*30)
    print(f"S-BERT:   {results['S-BERT']:.4f}")
    print(f"SimCSE:   {results['SimCSE']:.4f}")
    print(f"BLEU-1:   {results['BLEU-1']:.4f}")
    print(f"ROUGE-L:  {results['ROUGE-L']:.4f}")
    print(f"METEOR:   {results['METEOR']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()