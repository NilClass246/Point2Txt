<h1 align="center">Point2Text: Lightweight 3D Point Cloud Captioning via Large Language Models</h1>

<p align="center"><img width="100%" src="assets/example.png" /></p>
This repo is a project for CSC2503 at the University of Toronto. It implements a captioning pipeline connecting a PointBERT encoder with a GPT-2 decoder to generate descriptions for 3D point clouds.

## 1. Environment Setup
To set up the environment for training and evaluation:

```bash
conda create -n point2text python=3.10.14 -y
conda activate point2text
pip install -r requirements.txt
````

## 2. Data Preparation

We utilize the Objaverse dataset. All necessary data files, metadata, and pretrained weights are hosted on Hugging Face.

### Directory Structure

First, create the necessary directories:

```bash
mkdir -p data/objaverse/validation
mkdir -p models/pointbert
mkdir -p checkpoints
```

### Download Data

Download the dataset files from [NilClass/Point2Txt](https://huggingface.co/datasets/NilClass/Point2Txt). You can do this manually or via Python/CLI.

**Required files:**

1.  `Cap3D_automated_Objaverse_full.csv` (Captions)
2.  `chunk_paths.json` (Data mapping)
3.  `chunk_records.json` (ID mapping)
4.  `objaverse_part1.zip` (Training shards)
5.  `objaverse_part2.zip` (Training shards)
6.  `test_set.zip` (Validation/Test data)
7.  `test_set_captions.json` (Test captions)

**Move them to `data/objaverse/` and extract:**

```bash
# 1. Unzip training data
# This will extract chunk folders into data/objaverse/
unzip data/objaverse/objaverse_part1.zip -d data/objaverse/part1/
unzip data/objaverse/objaverse_part2.zip -d data/objaverse/part2/

# 2. Unzip validation/test data
# This will extract .npy files into data/objaverse/validation/test_set/
unzip data/objaverse/test_set.zip -d data/objaverse/validation/

# 3. Place metadata
mv data/objaverse/test_set_captions.json data/objaverse/validation/
```

## 3. Model Weights

### PointBERT Encoder

Download the pretrained PointBERT weights and place them in `models/pointbert/`:

```bash
wget https://huggingface.co/RunsenXu/PointLLM_7B_v1.1_init/resolve/main/point_bert_v1.2.pt -P models/pointbert/
```

### Point2Text Checkpoint

Download our trained `best_model.pth` from the Hugging Face repo and place it in `checkpoints/`:

```bash
# Assuming you downloaded best_model.pth from NilClass/Point2Txt
mv best_model.pth checkpoints/
```

## 4. Training

To train the model from scratch using the Objaverse dataset:

```bash
python train.py \
  --dataset objaverse \
  --data_path data/objaverse \
  --batch_size 128 \
  --epochs 10 \
  --save_dir checkpoints \
  --wandb_project Point2Text
```

*Note: You may need to log in to WandB (`wandb login`) to track experiments, or set `WANDB_MODE=offline`.*

## 5. Evaluation

To evaluate the model using BLEU, ROUGE, METEOR, and Semantic Similarity (S-BERT/SimCSE) on the test set:

```bash
python test.py \
  --dataset_type objaverse \
  --objaverse_npy_path data/objaverse/validation/test_set \
  --objaverse_json_path data/objaverse/validation/test_set_captions.json \
  --model_path checkpoints/best_model.pth \
  --batch_size 32
```

## 6. Inference & Visualization

You can generate captions for specific samples or custom point cloud files.

**Test on a dataset sample (with Open3D visualization):**

```bash
python inference.py \
  --data_path data/objaverse \
  --model_path checkpoints/best_model.pth \
  --test_idx 42 \
  --beams 5
```

## 7. Qualitative Results

To generate a batch of qualitative results (rendered images and captions) saved to disk:

```bash
python generate_qualitative.py \
  --dataset_type objaverse \
  --data_path data/objaverse/validation/test_set \
  --json_path data/objaverse/validation/test_set_captions.json \
  --model_path checkpoints/best_model.pth \
  --output_dir qualitative_results
```