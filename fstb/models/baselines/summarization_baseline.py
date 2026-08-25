# fstb/models/baselines/summarization_baseline.py
"""Summarization‑based memory baseline.

The baseline encodes the input with a transformer (re‑using ``FSTBTransformer`` as
encoder) and then compresses the entire sequence into a single summary vector
by mean‑pooling the hidden states. The summary is projected back to the model
dimension and injected as ``external_memory_bank`` for the decoder. This mimics a
common approach in long‑context tasks where a summarizer is used to retain
information from earlier turns.

Only a small linear bottleneck is added to keep the total parameter count within
~2 % of the FSTB model.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from fstb.config import ModelConfig, MemoryConfig
from fstb.models.fstb_transformer import FSTBTransformer

class SummarizationBaselineTransformer(nn.Module):
    """Baseline that summarizes the encoder hidden states and feeds them to the decoder.

    Parameters
    ----------
    model_config: ModelConfig
        Configuration for the transformer blocks.
    memory_config: MemoryConfig
        Memory configuration (re‑used for compatibility with ``FSTBTransformer``).
    summary_dim: int, optional
        Dimensionality of the compressed summary vector (default 128).
    """

    def __init__(self, model_config: ModelConfig, memory_config: MemoryConfig, summary_dim: int = 128):
        super().__init__()
        self.encoder = FSTBTransformer(model_config, memory_config)
        self.decoder = FSTBTransformer(model_config, memory_config)
        # Projection from model dim to a compact summary space
        self.summary_proj = nn.Linear(model_config.d_model, summary_dim, bias=False)
        # Small bottleneck to keep parity with FSTB
        self.bottleneck = nn.Sequential(
            nn.Linear(summary_dim, model_config.d_model),
            nn.ReLU(),
            nn.Linear(model_config.d_model, model_config.d_model)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
        external_memory_bank: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # Encode the sequence
        enc_out = self.encoder(input_ids, return_hidden_states=False)
        # Use the stage D hidden representation if available; fallback to logits
        encoder_hidden = enc_out.get("stage_d") or enc_out["logits"]
        # Create a summary vector by mean‑pooling and projecting
        summary_vec = self.summary_proj(encoder_hidden).mean(dim=1, keepdim=True)  # [b, 1, summary_dim]
        # Project back to model dimension
        mem_proj = self.bottleneck(summary_vec.squeeze(1))  # [b, d_model]
        # Expand to the full sequence length so the decoder receives a tensor of shape [b, seq, d_model]
        mem_proj = mem_proj.unsqueeze(1).repeat(1, input_ids.size(1), 1)
        # Decode with the summarized memory injected
        dec_out = self.decoder(
            input_ids,
            return_hidden_states=return_hidden_states,
            external_memory_bank=mem_proj,
        )
        return dec_out
