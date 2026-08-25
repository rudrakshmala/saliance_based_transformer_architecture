import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from fstb.config import ModelConfig, MemoryConfig
from fstb.models.modules import RMSNorm, TransformerBlock, precompute_rope_freqs
from fstb.models.stage_heads import StageAHeads, StageBHeads, StageCHeads

class AuxBaselineTransformer(nn.Module):
    """
    Model B — AuxBaselineTransformer: 24 homogeneous transformer blocks
    (identical to BaselineTransformer) BUT with auxiliary supervision heads
    bolted on at layers 6, 12, 18.
    """
    def __init__(self, model_config: ModelConfig, memory_config: MemoryConfig, d_bottleneck: int = 0):
        super().__init__()
        self.model_config = model_config
        self.memory_config = memory_config

        self.tok_embeddings = nn.Embedding(model_config.vocab_size, model_config.d_model)
        self.dropout = nn.Dropout(model_config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=model_config.d_model,
                n_heads=model_config.n_heads,
                d_ff=model_config.d_ff,
                dropout=model_config.dropout,
                norm_eps=model_config.norm_eps
            )
            for _ in range(model_config.n_layers)
        ])
        
        self.norm = RMSNorm(model_config.d_model, eps=model_config.norm_eps)
        self.lm_head = nn.Linear(model_config.d_model, model_config.vocab_size, bias=False)
        
        # Tie weights
        self.tok_embeddings.weight = self.lm_head.weight
        
        # Precompute RoPE frequencies
        cos, sin = precompute_rope_freqs(
            model_config.d_model // model_config.n_heads,
            model_config.max_seq_len,
            model_config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Bolted on auxiliary supervision heads
        self.stage_a_heads = StageAHeads(model_config.d_model, memory_config.d_mem, memory_config.num_memory_types)
        self.stage_b_heads = StageBHeads(model_config.d_model, memory_config.d_mem, memory_config.d_sym, memory_config.num_memory_types)
        self.stage_c_heads = StageCHeads(model_config.d_model)

        # Parameter balancing bottleneck MLP (to match FSTB parameter count)
        if d_bottleneck > 0:
            self.balancing_mlp = nn.Sequential(
                nn.Linear(model_config.d_model, d_bottleneck),
                nn.ReLU(),
                nn.Linear(d_bottleneck, model_config.d_model)
            )
        else:
            self.balancing_mlp = None

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
        external_memory_bank: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        b, s = input_ids.shape
        x = self.tok_embeddings(input_ids)
        x = self.dropout(x)

        all_hidden_states = []
        all_attentions = []

        stage_a_preds = None
        stage_b_preds = None
        stage_c_preds = None

        for i, block in enumerate(self.blocks):
            x, attn_weights = block(
                x,
                rope_cos=self.rope_cos,
                rope_sin=self.rope_sin,
                attn_mask=None
            )
            
            if i == 5:
                # Stage A heads (0-indexed layer 5 is the 6th layer)
                stage_a_preds = self.stage_a_heads(x)
            elif i == 11:
                # Stage B heads (0-indexed layer 11 is the 12th layer)
                stage_b_preds = self.stage_b_heads(x)
            elif i == 17:
                # Stage C heads (0-indexed layer 17 is the 18th layer)
                stage_c_preds = self.stage_c_heads(x)

            if return_hidden_states:
                all_hidden_states.append(x)
                all_attentions.append(attn_weights)

        # Optionally use balancing_mlp to ensure it is part of the computation graph if needed,
        # but just applying it or doing nothing is fine. We will apply it and add to x to mimic a residual.
        if self.balancing_mlp is not None:
            x = x + self.balancing_mlp(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        outputs = {
            "logits": logits,
            "stage_a": stage_a_preds,
            "stage_b": stage_b_preds,
            "stage_c": stage_c_preds
        }
        
        if return_hidden_states:
            outputs["hidden_states"] = all_hidden_states
            outputs["attentions"] = all_attentions

        return outputs
