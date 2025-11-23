import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel


class Point2Txt(nn.Module):
    """
    CLIPCap-style model:
        point cloud -> point encoder (Point-BERT) -> MLP mapper -> GPT-2 prefix
    """
     
    def __init__(self, point_encoder: nn.Module, gpt2: GPT2LMHeadModel, backbone_output_dim: int, prefix_len: int = 10):
        super().__init__()
        self.point_encoder = point_encoder
        self.gpt2 = gpt2
        self.prefix_len = prefix_len

        # Map global point embedding -> (prefix_len * gpt_emb_dim)
        self.mapper = nn.Sequential(
            nn.Linear(backbone_output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, prefix_len * gpt2.config.n_embd)
        )

    def encode_prefix(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (B, N, C)
        Returns:
            prefix: (B, prefix_len, gpt_emb_dim)
        """
        B = pts.size(0)
        feats = self.point_encoder(pts)  # (B, D)
        # print(global_feat.shape)
        global_feat = feats.mean(dim=1)
        mapped = self.mapper(global_feat)      # (B, prefix_len * H)
        prefix = mapped.view(B, self.prefix_len, self.gpt2.config.n_embd)
        return prefix
    
    def forward(
        self,
        pts: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        pts: (B, N, C)
        input_ids: (B, T)
        attention_mask: (B, T), optional
        labels: (B, T), optional
        """

        B = pts.size(0)
        prefix = self.encode_prefix(pts)  # (B, prefix_len, gpt_emb_dim)

        # Token embeddings from GPT-2
        token_embeds = self.gpt2.transformer.wte(input_ids)  # (B, T, H)

        # Concatenate prefix + tokens along sequence dimension
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)  # (B, prefix_len + T, H)

        # Build attention mask, padding ones for the prefix
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (B, self.prefix_len), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask_full = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            attention_mask_full = None

        # Build labels, ignoring prefix positions with -100
        if labels is not None:
            prefix_labels = torch.full(
                (B, self.prefix_len), -100, dtype=labels.dtype, device=labels.device
            )
            labels_full = torch.cat([prefix_labels, labels], dim=1)
        else:
            labels_full = None

        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_full,
            labels=labels_full,
        )
        return outputs