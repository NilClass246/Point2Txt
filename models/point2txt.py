import torch
import torch.nn as nn
from transformers import PreTrainedModel

class MLPMapper(nn.Module):
    def __init__(self, input_dim, output_dim, prefix_len, num_layers=3):
        super().__init__()
        self.prefix_len = prefix_len
        self.output_dim = output_dim
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, output_dim * prefix_len))
        layers.append(nn.GELU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
             layers.append(nn.Linear(output_dim * prefix_len, output_dim * prefix_len))
             layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        B = x.shape[0]
        return self.net(x).view(B, self.prefix_len, self.output_dim)


class TransformerMapper(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, prefix_len: int, num_layers: int = 4, num_heads: int = 8):
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
        if x.dim() == 3:
            x = x.squeeze(1)

        B = x.shape[0]

        x = self.linear_in(x).view(B, self.prefix_len, self.output_dim)
        
        x = self.transformer(x)
        
        x = self.linear_out(x)
        x = self.dropout(x)
        
        return x


class Point2Txt(nn.Module):
    def __init__(self, point_encoder: nn.Module, llm: PreTrainedModel, backbone_output_dim: int, prefix_len: int = 10, mapper_type: str = "transformer", mapper_layers: int = 4, mapper_heads: int = 8):
        super().__init__()
        self.point_encoder = point_encoder
        self.llm = llm
        self.prefix_len = prefix_len
        
        # Determine embedding dimension safely
        try:
            llm_emb_dim = getattr(llm.config, "n_embd", getattr(llm.config, "hidden_size", None))
        except:
            llm_emb_dim = llm.get_input_embeddings().weight.shape[1]
            
        if llm_emb_dim is None:
             raise ValueError("Could not determine LLM embedding dimension.")

        if mapper_type == "mlp":
            self.mapper = MLPMapper(
                input_dim=backbone_output_dim,
                output_dim=llm_emb_dim,
                prefix_len=prefix_len,
                num_layers=mapper_layers
            )
        elif mapper_type == "transformer":
            self.mapper = TransformerMapper(
                input_dim=backbone_output_dim,
                output_dim=llm_emb_dim,
                prefix_len=prefix_len,
                num_layers=mapper_layers,
                num_heads=mapper_heads
            )
        else:
            raise ValueError(f"Unknown mapper type: {mapper_type}")

    def encode_prefix(self, pts: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            pts = pts.float() 
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
        B = pts.size(0)
        prefix = self.encode_prefix(pts) 

        # Get embeddings from the LLM
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # Concatenate prefix + tokens
        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1) 

        # Build attention mask
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (B, self.prefix_len), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask_full = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            attention_mask_full = None

        # Build labels
        if labels is not None:
            prefix_labels = torch.full(
                (B, self.prefix_len), -100, dtype=labels.dtype, device=labels.device
            )
            labels_full = torch.cat([prefix_labels, labels], dim=1)
        else:
            labels_full = None

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_full,
            labels=labels_full,
        )
        return outputs