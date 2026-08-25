import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return cos, sin

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x shape: [batch, seq_len, n_heads, head_dim]
    b, s, h, d = x.shape
    cos = cos[:s, :].unsqueeze(0).unsqueeze(2).to(x.device)
    sin = sin[:s, :].unsqueeze(0).unsqueeze(2).to(x.device)
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (SiLU(w1(x)) * w3(x)) w2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, d = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.n_heads, self.head_dim)

        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        q = q.transpose(1, 2)  # [b, n_heads, s, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if attn_mask is not None:
            scores = scores + attn_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        context = torch.matmul(attn_weights_dropped, v)
        context = context.transpose(1, 2).contiguous().view(b, s, d)
        out = self.out_proj(context)
        return out, attn_weights

class CrossAttentionMemoryFusion(nn.Module):
    def __init__(self, d_model: int, d_mem: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_mem, d_model, bias=False)
        self.v_proj = nn.Linear(d_mem, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))  # Learnable residual gate initialized to 0
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [b, seq_len, d_model]
        # memory_embeddings: [b, num_mem, d_mem]
        b, s, d = x.shape
        b_m, n_m, d_m = memory_embeddings.shape
        
        if n_m == 0:
            return x, torch.zeros(b, self.n_heads, s, 0, device=x.device)

        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory_embeddings).view(b_m, n_m, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory_embeddings).view(b_m, n_m, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_dropped = self.dropout(attn_weights)

        context = torch.matmul(attn_dropped, v).transpose(1, 2).contiguous().view(b, s, d)
        fused = x + torch.tanh(self.gate) * self.out_proj(context)
        return fused, attn_weights

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, norm_eps: float = 1e-5):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.attn(normed_x, rope_cos=rope_cos, rope_sin=rope_sin, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights
