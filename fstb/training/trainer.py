import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from fstb.models.loss import FSTBLossFunction
from fstb.training.lr_schedule import CosineWarmupScheduler
from fstb.training.compute_budget import TokenBudgetTracker

@dataclass
class TrainingConfig:
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # Schedule
    warmup_steps: int = 100
    total_steps: int = 1000
    # Batch
    batch_size: int = 8
    grad_accumulation_steps: int = 1
    # Training
    max_epochs: int = 10
    eval_every: int = 50
    save_every: int = 100
    early_stopping_patience: int = 5
    # Reproducibility
    seed: int = 42
    # Paths
    checkpoint_dir: str = './checkpoints'
    # Compute
    use_amp: bool = False  # auto-disable on CPU
    target_tokens: int = 100_000  # token budget

class FSTBTrainer:
    def __init__(self, model: nn.Module, train_dataset, val_dataset, config: TrainingConfig, model_name='model'):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not torch.cuda.is_available():
            self.config.use_amp = False
            
        self.model.to(self.device)
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': []
        }

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        decay = set()
        no_decay = set()
    def _create_optimizer(self) -> torch.optim.Optimizer:
        decay_params = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )

    def train(self) -> Dict[str, List[float]]:
        self.set_seed(self.config.seed)
        
        from fstb.data.dataset import fstb_collate_fn
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=fstb_collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, collate_fn=fstb_collate_fn) if self.val_dataset else None
        
        optimizer = self._create_optimizer()
        scheduler = CosineWarmupScheduler(optimizer, self.config.warmup_steps, self.config.total_steps).get_scheduler()
        budget_tracker = TokenBudgetTracker(self.config.target_tokens)
        scaler = torch.amp.GradScaler('cuda') if self.config.use_amp else None
        
        step = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        try:
            from fstb.config import LossWeightsConfig
            fstb_loss_fn = FSTBLossFunction(LossWeightsConfig()).to(self.device)
            use_fstb_loss = True
        except ImportError:
            use_fstb_loss = False

        self.model.train()
        
        while not budget_tracker.is_exhausted() and step < self.config.total_steps:
            for batch in train_loader:
                if budget_tracker.is_exhausted() or step >= self.config.total_steps:
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                
                with torch.amp.autocast('cuda') if self.config.use_amp else torch.autocast(device_type=self.device.type, enabled=False):
                    outputs = self.model(input_ids)
                    
                    if use_fstb_loss and isinstance(outputs, dict) and 'stage_a' in outputs:
                        targets = {k: v.to(self.device) for k, v in batch.items()}
                        loss_dict = fstb_loss_fn(outputs, targets)
                        loss = loss_dict['loss']
                    else:
                        labels = input_ids[:, 1:].contiguous()
                        if isinstance(outputs, dict) and 'logits' in outputs:
                            logits = outputs['logits'][:, :-1, :].contiguous()
                        else:
                            logits = outputs[:, :-1, :].contiguous()
                        vocab_size = logits.size(-1)
                        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=0)

                loss = loss / self.config.grad_accumulation_steps
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.config.grad_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        optimizer.step()
                        
                    scheduler.step()
                    optimizer.zero_grad()
                
                budget_tracker.update(input_ids.numel())
                self.history['train_loss'].append(loss.item() * self.config.grad_accumulation_steps)
                step += 1
                
                if step % self.config.eval_every == 0 and val_loader:
                    val_loss = self.evaluate(val_loader)
                    self.history['val_loss'].append(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.save_checkpoint(step)
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.config.early_stopping_patience:
                        break
                        
                elif step % self.config.save_every == 0:
                    self.save_checkpoint(step)
                    
            if patience_counter >= self.config.early_stopping_patience:
                break
                
        return self.history

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = input_ids[:, 1:].contiguous()
            outputs = self.model(input_ids)
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits'][:, :-1, :].contiguous()
            else:
                logits = outputs[:, :-1, :].contiguous()
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            total_loss += loss.item()
            steps += 1
        return total_loss / max(1, steps)

    def evaluate(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0
        steps = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = input_ids[:, 1:].contiguous()
                outputs = self.model(input_ids)
                if isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits'][:, :-1, :].contiguous()
                else:
                    logits = outputs[:, :-1, :].contiguous()
                vocab_size = logits.size(-1)
                loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=0)
                total_loss += loss.item()
                steps += 1
        self.model.train()
        return total_loss / max(1, steps)

    def save_checkpoint(self, step: int) -> str:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, f"{self.model_name}_step_{step}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'step': step
        }, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
