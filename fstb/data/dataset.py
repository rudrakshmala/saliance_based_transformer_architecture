import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional

class SyntheticTokenEncoder:
    """Simple deterministic character/word token encoder for synthetic text experiments."""
    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size

    def encode(self, text: str, max_len: int = 128) -> torch.Tensor:
        words = text.lower().split()
        token_ids = [(hash(w) % (self.vocab_size - 4)) + 4 for w in words]
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids = token_ids + [0] * (max_len - len(token_ids)) # Pad with 0
        return torch.tensor(token_ids, dtype=torch.long)

class FSTBDataset(Dataset):
    """PyTorch Dataset wrapping synthetic memory benchmark samples."""
    def __init__(self, raw_samples: List[Dict[str, Any]], vocab_size: int = 4096, max_seq_len: int = 128, split: str = 'train', supervision: Optional[Dict[str, Any]] = None, tokenizer: Optional[Any] = None):
        self.raw_samples = raw_samples
        self.max_seq_len = max_seq_len
        self.split = split
        self.supervision = supervision or {}
        
        if tokenizer is not None:
            self.encoder = tokenizer
        else:
            self.encoder = SyntheticTokenEncoder(vocab_size=vocab_size)

    def __len__(self) -> int:
        return len(self.raw_samples)

    @property
    def ground_truth_labels(self) -> Dict[str, Any]:
        return self.supervision

    @classmethod
    def from_benchmark(cls, benchmark_dict: Dict[str, Any], split: str = 'train', tokenizer: Optional[Any] = None, **kwargs):
        samples = benchmark_dict.get(split, [])
        supervision = benchmark_dict.get('supervision', {})
        return cls(raw_samples=samples, split=split, supervision=supervision, tokenizer=tokenizer, **kwargs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.raw_samples[idx]
        
        if "text" in sample:
            text = sample["text"]
        elif "initial_fact" in sample and "contradictory_fact" in sample:
            text = f"{sample['initial_fact']} {sample['contradictory_fact']} {sample['query']}"
        elif "context_t1" in sample:
            text = f"{sample['context_t1']} {sample['context_t2']} {sample['query']}"
        elif "past_statement" in sample:
            text = f"{sample['past_statement']} {sample['recent_statement']} {sample['query']}"
        else:
            text = str(sample)

        input_ids = self.encoder.encode(text, max_len=self.max_seq_len)
        
        # Auxiliary Target Labels
        importance_target = torch.tensor([sample.get("importance", 0.5)], dtype=torch.float32).repeat(self.max_seq_len)
        worthiness_target = torch.tensor([1.0 if sample.get("importance", 0.5) > 0.5 else 0.0], dtype=torch.float32).repeat(self.max_seq_len)
        memory_type_target = torch.tensor([sample.get("memory_type", 2)], dtype=torch.long).repeat(self.max_seq_len)
        persistence_target = torch.tensor([1.0 if sample.get("is_contradiction", False) else 0.0], dtype=torch.float32).repeat(self.max_seq_len)
        contradiction_target = torch.tensor([1.0 if sample.get("is_contradiction", False) else 0.0], dtype=torch.float32).repeat(self.max_seq_len)
        consistency_target = torch.tensor([0.0 if sample.get("is_contradiction", False) else 1.0], dtype=torch.float32).repeat(self.max_seq_len)
        update_strategy_target = torch.tensor([0 if sample.get("is_contradiction", False) else 2], dtype=torch.long).repeat(self.max_seq_len)

        return {
            "input_ids": input_ids,
            "importance_target": importance_target,
            "memory_worthiness_target": worthiness_target,
            "memory_type_target": memory_type_target,
            "persistence_target": persistence_target,
            "contradiction_target": contradiction_target,
            "consistency_target": consistency_target,
            "update_strategy_target": update_strategy_target
        }

def fstb_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    collated = {}
    for key in batch[0].keys():
        collated[key] = torch.stack([b[key] for b in batch])
    return collated
