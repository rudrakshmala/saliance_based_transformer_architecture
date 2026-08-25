from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 256
    n_layers: int = 24
    n_heads: int = 8
    d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    
    # Functional Stage Block Partitioning (6 blocks per stage = 24 total)
    stage_a_blocks: List[int] = field(default_factory=lambda: list(range(0, 6)))
    stage_b_blocks: List[int] = field(default_factory=lambda: list(range(6, 12)))
    stage_c_blocks: List[int] = field(default_factory=lambda: list(range(12, 18)))
    stage_d_blocks: List[int] = field(default_factory=lambda: list(range(18, 24)))
    
    # Shared vs Independent LayerNorm
    independent_stage_norm: bool = True

@dataclass
class MemoryConfig:
    d_mem: int = 128
    d_sym: int = 64
    max_memory_slots: int = 256
    num_memory_types: int = 5  # Discard, Temp, Episodic, Semantic, Persistent User
    gumbel_temperature: float = 1.0
    gumbel_hard: bool = False
    similarity_threshold: float = 0.5
    decay_rate: float = 0.05
    top_k_retrieval: int = 4
    retrieval_mode: str = "hybrid"  # vector, symbolic, hybrid

@dataclass
class LossWeightsConfig:
    w_lm: float = 1.0
    w_stage_a: float = 0.25
    w_stage_b: float = 0.25
    w_stage_c: float = 0.25
    w_stage_d_aux: float = 0.25

@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    grad_clip: float = 1.0
    mixed_precision: bool = False
    seed: int = 42
    device: str = "cpu"
    checkpoint_dir: str = "./checkpoints"

@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    loss: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "./results"
