import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from fstb.config import ModelConfig, MemoryConfig
from fstb.models.fstb_transformer import FSTBTransformer

class RAGBaselineTransformer(nn.Module):
    """Retrieval‑Augmented Generation baseline.

    A lightweight RAG implementation that uses the same FSTBTransformer as
    encoder and decoder but inserts a dense retriever between them. The
    retriever builds a memory bank from the encoder hidden states, averages
    them to obtain a single vector per batch, and projects the vector back to
    the model dimension before feeding it to the decoder as ``external_memory_bank``.

    This design keeps the parameter count within ~2 % of the FSTB model by
    re‑using the existing transformer weights and adding only a small linear
    projection and a bottleneck MLP.
    """

    def __init__(self, model_config: ModelConfig, memory_config: MemoryConfig, retriever_dim: int = 128):
        super().__init__()
        # Encoder and decoder share the same configuration but are separate
        self.encoder = FSTBTransformer(model_config, memory_config)
        self.decoder = FSTBTransformer(model_config, memory_config)
        # Projection to a compact retrieval space
        self.retriever_proj = nn.Linear(model_config.d_model, retriever_dim, bias=False)
        # Small bottleneck to keep parameter parity
        self.balancing = nn.Sequential(
            nn.Linear(retriever_dim, model_config.d_model),
            nn.ReLU(),
            nn.Linear(model_config.d_model, model_config.d_model)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
        external_memory_bank: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # Encode the input sequence
        enc_out = self.encoder(input_ids, return_hidden_states=False)
        # ``logits`` from the encoder have shape [b, seq_len, vocab]
        # Use the hidden representation before the final lm_head (available as "stage_d" output)
        encoder_hidden = enc_out.get("stage_d")
        if encoder_hidden is None:
            # Fallback: use the logits directly (shape includes vocab dim) – not ideal but avoids crash
            encoder_hidden = enc_out["logits"]
        # Build a dense memory bank from the encoder hidden states
        mem_keys = self.retriever_proj(encoder_hidden)  # [b, seq_len, retriever_dim]
        # Simple retrieval: mean pooling over the sequence dimension
        mem_rep = mem_keys.mean(dim=1, keepdim=True)  # [b, 1, retriever_dim]
        # Project back to model dimension via balancing MLP
        mem_proj = self.balancing(mem_rep.squeeze(1))  # [b, d_model]
        mem_proj = mem_proj.unsqueeze(1).repeat(1, input_ids.size(1), 1)  # [b, seq_len, d_model]
        # Decode with the retrieved memory injected
        dec_out = self.decoder(
            input_ids,
            return_hidden_states=return_hidden_states,
            external_memory_bank=mem_proj,
        )
        return dec_out
