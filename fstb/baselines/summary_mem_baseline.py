import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from fstb.config import ModelConfig
from fstb.models.baseline import BaselineTransformer

class SummaryMemoryBaselineTransformer(nn.Module):
    """
    Baseline 4: Transformer + Summarization Memory.
    Maintains a rolling summary text token buffer injected as prefix tokens.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = BaselineTransformer(config)
        self.summary_buffer: Optional[torch.Tensor] = None

    def set_summary(self, summary_tokens: torch.Tensor):
        self.summary_buffer = summary_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, Any]:
        if self.summary_buffer is not None:
            b = input_ids.size(0)
            summary_prefix = self.summary_buffer.unsqueeze(0).repeat(b, 1).to(input_ids.device)
            full_input = torch.cat([summary_prefix, input_ids], dim=1)[:, :self.config.max_seq_len]
        else:
            full_input = input_ids

        return self.transformer(full_input, return_hidden_states=return_hidden_states)

class ReplayMemoryBaselineTransformer(nn.Module):
    """
    Baseline 5: Transformer + Memory Replay.
    Samples prior session conversation turns into current context window.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = BaselineTransformer(config)
        self.replay_buffer: List[torch.Tensor] = []

    def add_to_replay(self, turn_tokens: torch.Tensor):
        self.replay_buffer.append(turn_tokens)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, Any]:
        if self.replay_buffer:
            b = input_ids.size(0)
            # Take last sampled replay chunk
            replay_prefix = self.replay_buffer[-1].unsqueeze(0).repeat(b, 1).to(input_ids.device)
            full_input = torch.cat([replay_prefix, input_ids], dim=1)[:, :self.config.max_seq_len]
        else:
            full_input = input_ids

        return self.transformer(full_input, return_hidden_states=return_hidden_states)
