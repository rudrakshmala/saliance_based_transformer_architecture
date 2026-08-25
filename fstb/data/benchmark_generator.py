from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

from .memory_update_bench import MemoryUpdateBenchmarkGenerator
from .contradiction_bench import ContradictionBenchmarkGenerator
from .long_horizon_bench import LongHorizonBenchmarkGenerator

@dataclass
class BenchmarkConfig:
    num_users: int = 200
    sessions_per_user: int = 20
    mem_update_samples: int = 200
    contradiction_samples: int = 200
    long_horizon_samples: int = 100
    seed: int = 42

class MasterBenchmarkGenerator:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        
    def generate_all(self) -> Dict[str, List[Dict]]:
        mem_gen = MemoryUpdateBenchmarkGenerator(num_samples=self.config.mem_update_samples, seed=self.config.seed)
        contra_gen = ContradictionBenchmarkGenerator(num_samples=self.config.contradiction_samples, seed=self.config.seed)
        long_gen = LongHorizonBenchmarkGenerator(num_samples=self.config.long_horizon_samples, seed=self.config.seed)
        
        return {
            'memory_update': mem_gen.generate(),
            'contradiction': contra_gen.generate(),
            'long_horizon': long_gen.generate()
        }

    def get_train_val_test_split(self, samples: List[Dict], ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Tuple[List, List, List]:
        total = len(samples)
        train_idx = int(total * ratios[0])
        val_idx = train_idx + int(total * ratios[1])
        
        return samples[:train_idx], samples[train_idx:val_idx], samples[val_idx:]
        
    def export_supervision_labels(self, samples: List[Dict]) -> Dict[str, Any]:
        labels = {}
        for s in samples:
            labels[s['sample_id']] = {
                'correct_answer': s.get('correct_answer'),
                'is_contradiction': s.get('is_contradiction', False),
                'is_overwrite': s.get('is_overwrite', False)
            }
        return labels
