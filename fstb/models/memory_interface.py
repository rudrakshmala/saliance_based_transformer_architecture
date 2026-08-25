import enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch

class MemoryType(enum.IntEnum):
    DISCARD = 0
    TEMPORARY = 1
    EPISODIC = 2
    SEMANTIC = 3
    PERSISTENT_USER = 4

@dataclass
class MemoryObject:
    memory_id: str
    content_embedding: torch.Tensor          # [d_mem]
    symbolic_summary: torch.Tensor           # [d_sym]
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: float = 0.5
    confidence: float = 1.0
    timestamp: float = 0.0
    source_id: str = "user_session"
    update_version: int = 1
    retrieval_count: int = 0
    contradiction_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content_embedding": self.content_embedding.detach().cpu().tolist(),
            "symbolic_summary": self.symbolic_summary.detach().cpu().tolist(),
            "memory_type": int(self.memory_type),
            "importance": float(self.importance),
            "confidence": float(self.confidence),
            "timestamp": float(self.timestamp),
            "source_id": self.source_id,
            "update_version": self.update_version,
            "retrieval_count": self.retrieval_count,
            "contradiction_history": self.contradiction_history
        }

@dataclass
class BatchedMemoryState:
    """Vectorized PyTorch tensor state representation of memory objects for parallel GPU/CPU batch computation."""
    memory_ids: List[str]
    content_embeddings: torch.Tensor          # [num_memories, d_mem]
    symbolic_summaries: torch.Tensor          # [num_memories, d_sym]
    memory_types: torch.Tensor                # [num_memories] (int64)
    importance_scores: torch.Tensor           # [num_memories] (float32)
    confidence_scores: torch.Tensor           # [num_memories] (float32)
    timestamps: torch.Tensor                  # [num_memories] (float32)
    update_versions: torch.Tensor             # [num_memories] (int64)
    retrieval_counts: torch.Tensor            # [num_memories] (int64)

    @classmethod
    def empty(cls, d_mem: int, d_sym: int, device: torch.device = torch.device("cpu")) -> "BatchedMemoryState":
        return cls(
            memory_ids=[],
            content_embeddings=torch.zeros(0, d_mem, device=device),
            symbolic_summaries=torch.zeros(0, d_sym, device=device),
            memory_types=torch.zeros(0, dtype=torch.long, device=device),
            importance_scores=torch.zeros(0, dtype=torch.float32, device=device),
            confidence_scores=torch.zeros(0, dtype=torch.float32, device=device),
            timestamps=torch.zeros(0, dtype=torch.float32, device=device),
            update_versions=torch.zeros(0, dtype=torch.long, device=device),
            retrieval_counts=torch.zeros(0, dtype=torch.long, device=device)
        )

    def size(self) -> int:
        return len(self.memory_ids)
