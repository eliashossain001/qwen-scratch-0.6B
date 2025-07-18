# ==================== src/model.py ====================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ———— Begin in-file RotaryEmbedding implementation ————
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim: total rotary dimension (must be even)
        base: RoPE base frequency
        """
        super().__init__()
        # Build inverse frequencies for half the dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: [..., T, dim]
        *prefix, T, dim = x.shape
        half = dim // 2

        # Compute rotation frequencies
        t = torch.arange(T, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, half]
        cos = freqs.cos()[None, :, :]
        sin = freqs.sin()[None, :, :]

        # Separate even and odd indices
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]

        # Apply RoPE rotation
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        # Interleave back
        x_rot = torch.zeros_like(x)
        x_rot[..., ::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd
        return x_rot
# —————— End RotaryEmbedding implementation ——————

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, T, D]
        norm = x.norm(dim=-1, keepdim=True)
        return x * (self.scale / (norm + self.eps))

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, num_heads, rope_dim, rotary_base):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        # Rotary
        self.rotary = RotaryEmbedding(self.head_dim * rope_dim, base=rotary_base)
        self.rope_dim = int(self.head_dim * rope_dim)

    def forward(self, x):  # x: [B, T, D]
        B, T, D = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        # reshape to [B, H, T, Hd]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # apply rotary to first rope_dim dims
        qf, ql = q[..., :self.rope_dim], q[..., self.rope_dim:]
        kf, kl = k[..., :self.rope_dim], k[..., self.rope_dim:]
        qf = self.rotary(qf)
        kf = self.rotary(kf)
        q = torch.cat([qf, ql], dim=-1)
        k = torch.cat([kf, kl], dim=-1)
        # scaled dot-product
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        out = attn @ v  # [B, H, T, Hd]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config['hidden_size']
        self.norm1 = RMSNorm(d, eps=config['rms_norm_eps'])
        self.attn = GroupedQueryAttention(
            dim=d,
            num_heads=config['num_attention_heads'],
            rope_dim=config['rotary_pct'],
            rotary_base=config['rotary_base']
        )
        self.norm2 = RMSNorm(d, eps=config['rms_norm_eps'])
        self.ff = FeedForward(d, config['intermediate_size'])

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class QwenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config['hidden_size']
        self.token_emb = nn.Embedding(config['vocab_size'], d)
        self.pos_emb = nn.Embedding(config['max_position_embeddings'], d)
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config['num_hidden_layers'])
        ])
        self.norm = RMSNorm(d, eps=config['rms_norm_eps'])
        self.head = nn.Linear(d, config['vocab_size'], bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        tok_emb = self.token_emb(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits
