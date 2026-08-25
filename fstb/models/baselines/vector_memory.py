# fstb/models/baselines/vector_memory.py
"""Vector‑Memory baseline.

A lightweight external‑memory baseline that builds a dense vector bank from the
encoder hidden states and retrieves a memory representation for each token via
nearest‑neighbor similarity (dot‑product). The retrieved vector is projected back
to the model dimension and injected as ``external_memory_bank`` for the decoder.

The implementation re‑uses the existing ``FSTBTransformer`` for both encoder
and decoder to keep the parameter count within the target budget.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from fstb.config import ModelConfig, MemoryConfig
from fstb.models.fstb_transformer import FSTBTransformer

class VectorMemoryBaselineTransformer(nn.Module):
    """Baseline with a simple dense vector memory.

    The memory bank is constructed from the encoder hidden states (stage D
    output). Retrieval is performed by computing dot‑product similarity between a
    query (the current token representation) and all keys, then taking a weighted
    sum. This mimics a basic external‑memory retrieval without adding many
    parameters.
    """

    def __init__(self, model_config: ModelConfig, memory_config: MemoryConfig, embed_dim: int = 128):
        super().__init__()
        self.encoder = FSTBTransformer(model_config, memory_config)
        self.decoder = FSTBTransformer(model_config, memory_config)
        # Projection to a compact retrieval space
        self.key_proj = nn.Linear(model_config.d_model, embed_dim, bias=False)
        self.val_proj = nn.Linear(model_config.d_model, embed_dim, bias=False)
        # Project retrieved vector back to model dimension (tiny bottleneck to keep parity)
        self.retrieval_to_model = nn.Sequential(
            nn.Linear(embed_dim, model_config.d_model),
            nn.ReLU(),
            nn.Linear(model_config.d_model, model_config.d_model)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
        external_memory_bank: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # Encode input sequence
        enc_out = self.encoder(input_ids, return_hidden_states=False)
        encoder_hidden = enc_out.get("stage_d") or enc_out["logits"]  # fallback
        # Build key/value vectors
        keys = self.key_proj(encoder_hidden)   # [b, seq, embed_dim]
        vals = self.val_proj(encoder_hidden)   # [b, seq, embed_dim]
        # Use the last token as query (simple choice)
        query = keys[:, -1:, :]                # [b, 1, embed_dim]
        # Compute similarity (dot product) and softmax
        sim = torch.matmul(query, keys.transpose(-2, -1))  # [b, 1, seq]
        attn = torch.softmax(sim, dim=-1)                 # [b, 1, seq]
        # Retrieve memory representation
        retrieved = torch.matmul(attn, vals)               # [b, 1, embed_dim]
        # Project back to model dimension
        mem_proj = self.retrieval_to_model(retrieved.squeeze(1))  # [b, d_model]
        # Expand to match sequence length
        mem_proj = mem_proj.unsqueeze(1).repeat(1, input_ids.size(1), 1)
        # Decode with retrieved memory injected
        dec_out = self.decoder(
            input_ids,
            return_hidden_states=return_hidden_states,
            external_memory_bank=mem_proj,
        )
        return dec_out
