from typing import Dict
from fstb.config import ModelConfig

class FLOPsCounter:
    @staticmethod
    def estimate_forward_flops(model_cfg: ModelConfig) -> int:
        L = model_cfg.n_layers
        d = model_cfg.d_model
        d_ff = model_cfg.d_ff
        s = model_cfg.max_seq_len
        return int(2 * L * (12 * d**2 * s + 2 * d * d_ff * s))

    @staticmethod
    def estimate_training_flops(model_cfg: ModelConfig, num_steps: int, batch_size: int) -> int:
        forward_flops = FLOPsCounter.estimate_forward_flops(model_cfg)
        return int(6 * forward_flops * num_steps * batch_size)

class TokenBudgetTracker:
    def __init__(self, target_tokens: int):
        self.target_tokens = target_tokens
        self.current_tokens = 0

    def update(self, batch_tokens: int) -> None:
        self.current_tokens += batch_tokens

    def is_exhausted(self) -> bool:
        return self.current_tokens >= self.target_tokens

    def progress(self) -> float:
        if self.target_tokens == 0:
            return 1.0
        return min(1.0, self.current_tokens / self.target_tokens)

    def remaining(self) -> int:
        return max(0, self.target_tokens - self.current_tokens)

class ComputeMatchedConfig:
    def __init__(self, model_cfg: ModelConfig, target_tokens: int, batch_size: int, seq_len: int):
        self.model_cfg = model_cfg
        self.target_tokens = target_tokens
        self.batch_size = batch_size
        self.seq_len = seq_len

    @property
    def num_steps(self) -> int:
        tokens_per_step = self.batch_size * self.seq_len
        return self.target_tokens // max(1, tokens_per_step)

    @property
    def warmup_steps(self) -> int:
        return int(0.1 * self.num_steps)

    def to_dict(self) -> Dict:
        return {
            "model_cfg": self.model_cfg.__dict__,
            "target_tokens": self.target_tokens,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "num_steps": self.num_steps,
            "warmup_steps": self.warmup_steps,
            "estimated_flops": FLOPsCounter.estimate_training_flops(self.model_cfg, self.num_steps, self.batch_size)
        }
