import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
from fstb.config import ModelConfig
from fstb.models.modules import RMSNorm, TransformerBlock, precompute_rope_freqs

class BaselineTransformer(nn.Module):
    """
    Standard Homogeneous Decoder-Only Transformer Baseline.
    - 24 identical transformer blocks
    - Rotary Positional Embedding (RoPE)
    - RMSNorm
    - SwiGLU Feed-Forward Networks
    - Causal Self-Attention
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                norm_eps=config.norm_eps
            )
            for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights between embedding and lm_head for efficiency
        self.tok_embeddings.weight = self.lm_head.weight
        
        # Precompute RoPE frequencies
        cos, sin = precompute_rope_freqs(
            config.d_model // config.n_heads,
            config.max_seq_len,
            config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        b, s = input_ids.shape
        x = self.tok_embeddings(input_ids)
        x = self.dropout(x)

        all_hidden_states = []
        all_attentions = []

        for block in self.blocks:
            x, attn_weights = block(
                x,
                rope_cos=self.rope_cos,
                rope_sin=self.rope_sin,
                attn_mask=attn_mask
            )
            if return_hidden_states:
                all_hidden_states.append(x)
                all_attentions.append(attn_weights)

        x = self.norm(x)
        logits = self.lm_head(x)

        outputs = {"logits": logits}
        if return_hidden_states:
            outputs["hidden_states"] = all_hidden_states
            outputs["attentions"] = all_attentions

        return outputs
