import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class MemoryRoutingGate(nn.Module):
    def __init__(self, d_model: int, num_categories: int = 5, temperature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.num_categories = num_categories
        self.temperature = temperature
        
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_categories)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        hard: bool = False,
        temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes routing probabilities over the 5 memory categories:
        0: Discard
        1: Temporary Memory
        2: Episodic Memory
        3: Semantic Memory
        4: Persistent User Memory

        Returns:
            routing_probs: [batch, seq_len, num_categories] (differentiable Gumbel-Softmax or Softmax)
            gate_logits: [batch, seq_len, num_categories]
        """
        temp = temperature if temperature is not None else self.temperature
        logits = self.gate_proj(hidden_states)
        
        if self.training:
            # Gumbel-Softmax for end-to-end gradient flow
            routing_probs = F.gumbel_softmax(logits, tau=temp, hard=hard, dim=-1)
        else:
            if hard:
                idx = logits.argmax(dim=-1)
                routing_probs = F.one_hot(idx, num_classes=self.num_categories).float()
            else:
                routing_probs = F.softmax(logits / temp, dim=-1)

        return routing_probs, logits
