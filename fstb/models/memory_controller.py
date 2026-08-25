import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from fstb.models.memory_interface import MemoryObject, BatchedMemoryState, MemoryType

class DynamicMemoryController(nn.Module):
    def __init__(self, d_mem: int = 128, d_sym: int = 64, max_slots: int = 256, decay_rate: float = 0.05):
        super().__init__()
        self.d_mem = d_mem
        self.d_sym = d_sym
        self.max_slots = max_slots
        self.decay_rate = decay_rate
        
        # Linear projection for hybrid query matching
        self.vector_query_proj = nn.Linear(d_mem, d_mem, bias=False)
        self.symbolic_query_proj = nn.Linear(d_mem, d_sym, bias=False)
        self.hybrid_fusion_weight = nn.Parameter(torch.tensor(0.5))

        # Memory storage container
        self.memories: Dict[str, MemoryObject] = {}

    def store(self, memory_obj: MemoryObject) -> str:
        """Stores or appends a new MemoryObject to internal store."""
        if len(self.memories) >= self.max_slots:
            # Evict least important / oldest memory if capacity reached
            sorted_keys = sorted(
                self.memories.keys(),
                key=lambda k: (self.memories[k].importance, self.memories[k].timestamp)
            )
            if sorted_keys:
                del self.memories[sorted_keys[0]]

        self.memories[memory_obj.memory_id] = memory_obj
        return memory_obj.memory_id

    def invalidate(self, memory_id: str) -> bool:
        """Removes a memory from storage."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False

    def update(self, memory_id: str, new_embedding: Optional[torch.Tensor] = None, new_confidence: Optional[float] = None) -> bool:
        """Updates content embedding or confidence of an existing memory."""
        if memory_id in self.memories:
            mem = self.memories[memory_id]
            if new_embedding is not None:
                mem.content_embedding = new_embedding
            if new_confidence is not None:
                mem.confidence = new_confidence
            mem.update_version += 1
            return True
        return False

    def merge(self, memory_id_a: str, memory_id_b: str, new_id: str) -> Optional[MemoryObject]:
        """Merges memory_a and memory_b into a new unified MemoryObject."""
        if memory_id_a in self.memories and memory_id_b in self.memories:
            mem_a = self.memories[memory_id_a]
            mem_b = self.memories[memory_id_b]
            
            merged_embedding = 0.5 * (mem_a.content_embedding + mem_b.content_embedding)
            merged_symbolic = 0.5 * (mem_a.symbolic_summary + mem_b.symbolic_summary)
            merged_importance = max(mem_a.importance, mem_b.importance)
            merged_confidence = min(mem_a.confidence, mem_b.confidence)
            
            merged_obj = MemoryObject(
                memory_id=new_id,
                content_embedding=merged_embedding,
                symbolic_summary=merged_symbolic,
                memory_type=mem_a.memory_type,
                importance=merged_importance,
                confidence=merged_confidence,
                timestamp=max(mem_a.timestamp, mem_b.timestamp),
                source_id=f"{mem_a.source_id}+{mem_b.source_id}",
                update_version=max(mem_a.update_version, mem_b.update_version) + 1
            )
            
            self.invalidate(memory_id_a)
            self.invalidate(memory_id_b)
            self.store(merged_obj)
            return merged_obj
        return None

    def decay(self, delta_time: float):
        """Applies exponential temporal decay to importance scores."""
        to_delete = []
        for mem_id, mem in self.memories.items():
            if mem.memory_type != MemoryType.PERSISTENT_USER:
                mem.importance *= math.exp(-self.decay_rate * delta_time)
                if mem.importance < 0.05:
                    to_delete.append(mem_id)
        for mem_id in to_delete:
            del self.memories[mem_id]

    def get_batched_state(self, device: torch.device = torch.device("cpu")) -> BatchedMemoryState:
        """Converts stored memory objects into a vectorized PyTorch BatchedMemoryState."""
        if not self.memories:
            return BatchedMemoryState.empty(self.d_mem, self.d_sym, device=device)

        mem_list = list(self.memories.values())
        return BatchedMemoryState(
            memory_ids=[m.memory_id for m in mem_list],
            content_embeddings=torch.stack([m.content_embedding for m in mem_list]).to(device),
            symbolic_summaries=torch.stack([m.symbolic_summary for m in mem_list]).to(device),
            memory_types=torch.tensor([int(m.memory_type) for m in mem_list], dtype=torch.long, device=device),
            importance_scores=torch.tensor([m.importance for m in mem_list], dtype=torch.float32, device=device),
            confidence_scores=torch.tensor([m.confidence for m in mem_list], dtype=torch.float32, device=device),
            timestamps=torch.tensor([m.timestamp for m in mem_list], dtype=torch.float32, device=device),
            update_versions=torch.tensor([m.update_version for m in mem_list], dtype=torch.long, device=device),
            retrieval_counts=torch.tensor([m.retrieval_count for m in mem_list], dtype=torch.long, device=device)
        )

    def retrieve_differentiable(
        self,
        query_states: torch.Tensor,               # [b, seq_len, d_model]
        memory_bank: torch.Tensor,                # [b, num_mem, d_mem]
        mode: str = "hybrid",
        top_k: int = 4,
        min_confidence: float = 0.0,
        temp_window: Optional[Tuple[float, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable soft retrieval mechanism for end-to-end gradient backpropagation.
        Returns:
            retrieved_memories: [b, seq_len, top_k, d_mem] or fused memory reps [b, seq_len, d_mem]
            attn_probs: [b, seq_len, num_mem]
        """
        b, s, d = query_states.shape
        b_m, n_m, d_m = memory_bank.shape
        
        if n_m == 0:
            return torch.zeros(b, s, d_m, device=query_states.device), torch.zeros(b, s, 0, device=query_states.device)

        # Project query to memory embedding space
        v_query = self.vector_query_proj(query_states[:, :, :self.d_mem])  # [b, s, d_mem]
        
        # Cosine similarity matching
        v_query_norm = F.normalize(v_query, dim=-1)
        mem_bank_norm = F.normalize(memory_bank, dim=-1)  # [b, n_m, d_mem]
        
        scores = torch.matmul(v_query_norm, mem_bank_norm.transpose(1, 2)) / math.sqrt(self.d_mem) # [b, s, n_m]
        attn_probs = F.softmax(scores, dim=-1)  # [b, s, n_m]
        
        # Weighted soft memory retrieval
        fused_memory = torch.matmul(attn_probs, memory_bank)  # [b, s, d_mem]
        return fused_memory, attn_probs

    def retrieve(
        self,
        query_embedding: torch.Tensor,            # [d_mem]
        mode: str = "hybrid",
        top_k: int = 4,
        min_confidence: float = 0.0,
        min_timestamp: Optional[float] = None
    ) -> List[Tuple[MemoryObject, float]]:
        """Discrete memory lookup for evaluation & inference."""
        if not self.memories:
            return []

        candidates = []
        for mem in self.memories.values():
            if mem.confidence < min_confidence:
                continue
            if min_timestamp is not None and mem.timestamp < min_timestamp:
                continue
            
            # Vector score
            sim_vec = F.cosine_similarity(query_embedding.unsqueeze(0), mem.content_embedding.unsqueeze(0)).item()
            
            # Symbolic score
            sim_sym = F.cosine_similarity(
                self.symbolic_query_proj(query_embedding.unsqueeze(0)),
                mem.symbolic_summary.unsqueeze(0)
            ).item()
            
            if mode == "vector":
                final_score = sim_vec
            elif mode == "symbolic":
                final_score = sim_sym
            else:  # hybrid
                alpha = torch.sigmoid(self.hybrid_fusion_weight).item()
                final_score = alpha * sim_vec + (1 - alpha) * sim_sym

            candidates.append((mem, final_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_results = candidates[:top_k]
        
        # Increment retrieval count
        for mem, _ in top_results:
            mem.retrieval_count += 1

        return top_results
