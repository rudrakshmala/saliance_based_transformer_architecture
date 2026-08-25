import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class StageAHeads(nn.Module):
    """Stage A — Memory Selection Output Heads (Blocks 1-6)"""
    def __init__(self, d_model: int, d_mem: int, num_types: int = 5):
        super().__init__()
        self.importance_head = nn.Linear(d_model, 1)
        self.candidate_proj = nn.Linear(d_model, d_mem)
        self.type_head = nn.Linear(d_model, num_types)
        self.temporal_head = nn.Linear(d_model, 1)
        self.emotional_head = nn.Linear(d_model, 1)
        self.confidence_head = nn.Linear(d_model, 1)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "importance_score": torch.sigmoid(self.importance_head(hidden_states)).squeeze(-1),
            "memory_candidate_rep": F.normalize(self.candidate_proj(hidden_states), dim=-1),
            "memory_type_logits": self.type_head(hidden_states),
            "memory_type_probs": F.softmax(self.type_head(hidden_states), dim=-1),
            "temporal_relevance": torch.sigmoid(self.temporal_head(hidden_states)).squeeze(-1),
            "emotional_relevance": torch.sigmoid(self.emotional_head(hidden_states)).squeeze(-1),
            "confidence_score": torch.sigmoid(self.confidence_head(hidden_states)).squeeze(-1)
        }

class StageBHeads(nn.Module):
    """Stage B — Memory Encoding Output Heads (Blocks 7-12)"""
    def __init__(self, d_model: int, d_mem: int, d_sym: int, num_categories: int = 5):
        super().__init__()
        self.content_proj = nn.Linear(d_model, d_mem)
        self.symbolic_proj = nn.Linear(d_model, d_sym)
        self.category_head = nn.Linear(d_model, num_categories)
        self.indexing_key_proj = nn.Linear(d_model, d_sym)
        self.persistence_head = nn.Linear(d_model, 1)
        self.update_strategy_head = nn.Linear(d_model, 3) # 0: overwrite, 1: merge, 2: append

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "content_embedding": F.normalize(self.content_proj(hidden_states), dim=-1),
            "symbolic_summary": F.normalize(self.symbolic_proj(hidden_states), dim=-1),
            "category_logits": self.category_head(hidden_states),
            "indexing_keys": F.normalize(self.indexing_key_proj(hidden_states), dim=-1),
            "persistence_prob": torch.sigmoid(self.persistence_head(hidden_states)).squeeze(-1),
            "update_strategy_logits": self.update_strategy_head(hidden_states)
        }

class StageCHeads(nn.Module):
    """Stage C — Memory Validation Output Heads (Blocks 13-18)"""
    def __init__(self, d_model: int):
        super().__init__()
        self.consistency_head = nn.Linear(d_model, 1)
        self.contradiction_head = nn.Linear(d_model, 1)
        self.reliability_head = nn.Linear(d_model, 1)
        self.temporal_validity_head = nn.Linear(d_model, 1)
        self.retrieval_confidence_head = nn.Linear(d_model, 1)
        self.update_rec_head = nn.Linear(d_model, 3) # 0: keep, 1: invalidate, 2: update

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "consistency_score": torch.sigmoid(self.consistency_head(hidden_states)).squeeze(-1),
            "contradiction_score": torch.sigmoid(self.contradiction_head(hidden_states)).squeeze(-1),
            "source_reliability_score": torch.sigmoid(self.reliability_head(hidden_states)).squeeze(-1),
            "temporal_validity_score": torch.sigmoid(self.temporal_validity_head(hidden_states)).squeeze(-1),
            "retrieval_confidence": torch.sigmoid(self.retrieval_confidence_head(hidden_states)).squeeze(-1),
            "update_rec_logits": self.update_rec_head(hidden_states)
        }
