import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, C = q.shape

        q = self.q_proj(q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, heads, dropout)
        self.cross_mha = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        self_attn = self.self_mha(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn))

        cross_attn = self.cross_mha(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attn))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, N=3, heads=8, d_ff=2048, max_len=100):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads, d_ff) for _ in range(N)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads, d_ff) for _ in range(N)])

        self.out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, T):
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        return mask

    def forward(self, src, tgt, src_mask=None):
        src = self.pos(self.token_emb(src))
        tgt = self.pos(self.token_emb(tgt))

        # encoder
        for layer in self.encoder:
            src = layer(src, src_mask)

        # decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        for layer in self.decoder:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        return self.out(tgt)
    
