import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from fstb.config import ModelConfig
from fstb.models.baseline import BaselineTransformer
from fstb.models.modules import CrossAttentionMemoryFusion

class VectorMemoryBaselineTransformer(nn.Module):
    """
    Baseline 3: Transformer + Vector Memory.
    Appends a dense vector Key-Value memory buffer directly into attention or via cross-attention.
    """
    def __init__(self, config: ModelConfig, d_mem: int = 128):
        super().__init__()
        self.config = config
        self.d_mem = d_mem
        self.transformer = BaselineTransformer(config)
        self.fusion = CrossAttentionMemoryFusion(config.d_model, d_mem)

    def forward(
        self,
        input_ids: torch.Tensor,
        vector_memory_bank: Optional[torch.Tensor] = None, # [b, num_mem, d_mem]
        attn_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, Any]:
        outputs = self.transformer(input_ids, attn_mask=attn_mask, return_hidden_states=True)
        hidden_states = outputs["hidden_states"][-1]
        
        if vector_memory_bank is not None and vector_memory_bank.size(1) > 0:
            fused_hidden, mem_attn = self.fusion(hidden_states, vector_memory_bank)
            logits = self.transformer.lm_head(self.transformer.norm(fused_hidden))
            outputs["logits"] = logits
            outputs["vector_mem_attn"] = mem_attn
            
        if not return_hidden_states:
            outputs.pop("hidden_states", None)
            outputs.pop("attentions", None)

        return outputs
