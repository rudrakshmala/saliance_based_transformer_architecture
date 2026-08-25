import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from typing import Callable

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min_ratio = eta_min_ratio

    def get_scheduler(self) -> LambdaLR:
        lr_lambda = self.get_lr_lambda(self.warmup_steps, self.total_steps, self.eta_min_ratio)
        return LambdaLR(self.optimizer, lr_lambda)

    @staticmethod
    def get_lr_lambda(warmup_steps: int, total_steps: int, eta_min_ratio: float) -> Callable[[int], float]:
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (1.0 - eta_min_ratio) * cosine_decay + eta_min_ratio
            
        return lr_lambda

def get_linear_warmup_cosine_decay(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1) -> LambdaLR:
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, min_lr_ratio)
    return scheduler.get_scheduler()
