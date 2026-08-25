import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from fstb.config import LossWeightsConfig

class FSTBLossFunction(nn.Module):
    """
    Multi-Task Auxiliary Loss Calculator for FSTB.
    Combines LM cross-entropy with stage-specific supervision for Stages A, B, C, and D.
    """
    def __init__(self, config: LossWeightsConfig):
        super().__init__()
        self.config = config
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        model_outputs: Dict[str, Any],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # 1. Standard Next-Token Cross-Entropy LM Loss
        logits = model_outputs["logits"]  # [b, s, vocab_size]
        lm_targets = targets["input_ids"]  # [b, s]
        
        # Shift for causal next-token prediction
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shift_targets = lm_targets[:, 1:].contiguous().view(-1)
        loss_lm = self.ce_loss(shift_logits, shift_targets)

        loss_stage_a = torch.tensor(0.0, device=logits.device)
        loss_stage_b = torch.tensor(0.0, device=logits.device)
        loss_stage_c = torch.tensor(0.0, device=logits.device)
        loss_stage_d_aux = torch.tensor(0.0, device=logits.device)

        # 2. Stage A Losses: Importance, Worthiness, Entity, Preference
        if "stage_a" in model_outputs:
            preds_a = model_outputs["stage_a"]
            if "importance_target" in targets:
                loss_stage_a += self.mse_loss(preds_a["importance_score"], targets["importance_target"])
            if "memory_worthiness_target" in targets:
                loss_stage_a += self.bce_loss(preds_a["confidence_score"], targets["memory_worthiness_target"])
            if "memory_type_target" in targets:
                type_logits = preds_a["memory_type_logits"].view(-1, preds_a["memory_type_logits"].size(-1))
                type_targets = targets["memory_type_target"].view(-1)
                loss_stage_a += self.ce_loss(type_logits, type_targets)

        # 3. Stage B Losses: Type classification, Compression, Indexing, Persistence
        if "stage_b" in model_outputs:
            preds_b = model_outputs["stage_b"]
            if "persistence_target" in targets:
                loss_stage_b += self.bce_loss(preds_b["persistence_prob"], targets["persistence_target"])
            if "update_strategy_target" in targets:
                strat_logits = preds_b["update_strategy_logits"].view(-1, 3)
                strat_targets = targets["update_strategy_target"].view(-1)
                loss_stage_b += self.ce_loss(strat_logits, strat_targets)

        # 4. Stage C Losses: Contradiction detection, Temporal consistency, Retrieval validation, Calibration
        if "stage_c" in model_outputs:
            preds_c = model_outputs["stage_c"]
            if "contradiction_target" in targets:
                loss_stage_c += self.bce_loss(preds_c["contradiction_score"], targets["contradiction_target"])
            if "consistency_target" in targets:
                loss_stage_c += self.bce_loss(preds_c["consistency_score"], targets["consistency_target"])

        # 5. Stage D Auxiliary Losses: Factual consistency penalty & Memory utilization reward
        if "mem_fusion_attn" in model_outputs:
            mem_attn = model_outputs["mem_fusion_attn"] # [b, n_heads, s, num_mem]
            # Reward attending to relevant memory slots
            if mem_attn.numel() > 0:
                utilization_reward = -torch.mean(torch.log(mem_attn.mean() + 1e-6))
                loss_stage_d_aux += 0.1 * utilization_reward

        # Total Loss Calculation
        total_loss = (
            self.config.w_lm * loss_lm +
            self.config.w_stage_a * loss_stage_a +
            self.config.w_stage_b * loss_stage_b +
            self.config.w_stage_c * loss_stage_c +
            self.config.w_stage_d_aux * loss_stage_d_aux
        )

        return {
            "loss": total_loss,
            "loss_lm": loss_lm.detach(),
            "loss_stage_a": loss_stage_a.detach(),
            "loss_stage_b": loss_stage_b.detach(),
            "loss_stage_c": loss_stage_c.detach(),
            "loss_stage_d_aux": loss_stage_d_aux.detach()
        }
