import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fstb.training.trainer import FSTBTrainer, TrainingConfig
from fstb.training.lr_schedule import get_linear_warmup_cosine_decay
from fstb.training.compute_budget import TokenBudgetTracker, FLOPsCounter, ComputeMatchedConfig
from fstb.config import ModelConfig

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 16)
        self.linear = nn.Linear(16, 10)

    def forward(self, x):
        return {'logits': self.linear(self.embed(x))}

class DummyDataset(Dataset):
    def __len__(self):
        return 10
    
    def __getitem__(self, idx):
        return {'input_ids': torch.randint(0, 10, (10,))}

def test():
    model = DummyModel()
    dataset = DummyDataset()
    config = TrainingConfig(warmup_steps=10, total_steps=100)
    
    # Test scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = get_linear_warmup_cosine_decay(optimizer, 10, 100)
    print("Scheduler instantiated successfully")
    
    # Test compute budget
    model_config = ModelConfig(n_layers=2, d_model=16, d_ff=64, max_seq_len=10)
    flops = FLOPsCounter.estimate_forward_flops(model_config)
    tracker = TokenBudgetTracker(1000)
    matched = ComputeMatchedConfig(model_config, 1000, 8, 10)
    print("Compute budget classes instantiated successfully")
    
    # Test trainer
    trainer = FSTBTrainer(model, dataset, dataset, config)
    print("Trainer instantiated successfully")

if __name__ == "__main__":
    test()
