import torch

from os import path, makedirs
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.point2txt import Point2Txt
from models.llm import load_gpt2
from models.encoder import load_point_encoder
from datasets.dataset import Cap3DShapeNetPreprocessed, get_collate_fn

num_epochs = 3
lr = 1e-4
prefix_len = 10
batch_size = 32

config = {
    "encoder_config_path": "models/pointbert/PointTransformer_8192point_2layer.yaml",
    "encoder_ckpt_path": "models/pointbert/point_bert_v1.2.pt",
    "data_path": "data/shapenet"
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load models
    point_encoder, backbone_output_dim = load_point_encoder(
        config_path=config["encoder_config_path"],
        ckpt_path=config["encoder_ckpt_path"],
        device=device
    )
    gpt2, tokenizer = load_gpt2(device)
    model = Point2Txt(point_encoder, gpt2, backbone_output_dim=backbone_output_dim, prefix_len=prefix_len).to(device)
    print("Model ready.")

    # load dataset
    dataset = Cap3DShapeNetPreprocessed(
        points_path=path.join(config["data_path"], "processed_points.pt"),
        ids_path=path.join(config["data_path"],"point_ids.json"),
        csv_path=path.join(config["data_path"],"Cap3D_automated_ShapeNet.csv"),
        device =device,
    )
    collate_fn = get_collate_fn(tokenizer, device)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print("DataLoader ready.")

    # set up optimizer
    # Only train the parts that require grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, (pts, input_ids, attention_mask, labels, raw_caps) in enumerate(tqdm(train_loader)):
            pts = pts.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                pts=pts,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} finished. Avg loss: {total_loss / len(train_loader):.4f}")

    makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/test_model.pth")

if __name__ == "__main__":
    main()