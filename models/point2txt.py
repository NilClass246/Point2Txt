import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel

class TransformerMapper(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, prefix_len: int, num_layers: int = 4, num_heads: int = 8):
        """
        Args:
            input_dim: Dimension of the PointBERT output (e.g., 384)
            output_dim: Dimension of the GPT-2 embedding (e.g., 768)
            prefix_len: Number of tokens to generate for the prefix
        """
        super().__init__()
        self.prefix_len = prefix_len
        self.output_dim = output_dim

        self.linear_in = nn.Linear(input_dim, prefix_len * output_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim, 
            nhead=num_heads, 
            dim_feedforward=output_dim * 4,
            batch_first=True, 
            activation="gelu",
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.linear_out = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        x: (B, input_dim) or (B, 1, input_dim) - Global Point Feature
        """
        if x.dim() == 3:
            x = x.squeeze(1)

        B = x.shape[0]

        x = self.linear_in(x).view(B, self.prefix_len, self.output_dim)
        
        x = self.transformer(x)
        
        x = self.linear_out(x)
        x = self.dropout(x)
        
        return x


class Point2Txt(nn.Module):
    """
    CLIPCap-style model with Transformer Mapper:
        point cloud -> point encoder (Point-BERT) -> Transformer Mapper -> GPT-2 prefix
    """
      
    def __init__(self, point_encoder: nn.Module, gpt2: GPT2LMHeadModel, backbone_output_dim: int, prefix_len: int = 10):
        super().__init__()
        self.point_encoder = point_encoder
        self.gpt2 = gpt2
        self.prefix_len = prefix_len
        
        gpt_emb_dim = gpt2.config.n_embd

        self.mapper = TransformerMapper(
            input_dim=backbone_output_dim,
            output_dim=gpt_emb_dim,
            prefix_len=prefix_len,
            num_layers=4,
            num_heads=8
        )

    def encode_prefix(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (B, N, C)
        Returns:
            prefix: (B, prefix_len, gpt_emb_dim)
        """
        feats = self.point_encoder(pts)
        
        if feats.dim() == 3:
            global_feat = feats[:, 0, :] 
        else:
            global_feat = feats

        prefix = self.mapper(global_feat)
        
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