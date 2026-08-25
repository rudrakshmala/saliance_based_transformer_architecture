import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from fstb.config import ModelConfig
from fstb.models.baseline import BaselineTransformer

class RAGBaselineTransformer(nn.Module):
    """
    Baseline 2: Transformer + External RAG.
    Dense retriever queries external text document / session chunk buffer
    and prepends retrieved top-k context tokens into the prompt window.
    """
    def __init__(self, config: ModelConfig, top_k: int = 2):
        super().__init__()
        self.config = config
        self.top_k = top_k
        self.transformer = BaselineTransformer(config)
        self.document_store: List[torch.Tensor] = [] # List of token tensors [seq_len]

    def add_documents(self, doc_tokens: torch.Tensor):
        self.document_store.append(doc_tokens)

    def retrieve_context(self, query_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.document_store:
            return None
        # Compute simple TF-IDF / similarity matching between query and doc store
        query_set = set(query_ids.view(-1).tolist())
        scores = []
        for doc in self.document_store:
            doc_set = set(doc.view(-1).tolist())
            overlap = len(query_set.intersection(doc_set)) / (len(query_set) + 1e-5)
            scores.append(overlap)
        
        top_idx = torch.tensor(scores).argsort(descending=True)[:self.top_k]
        retrieved_docs = [self.document_store[i] for i in top_idx]
        return torch.cat(retrieved_docs, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, Any]:
        b, s = input_ids.shape
        retrieved_tokens = self.retrieve_context(input_ids[0])
        
        if retrieved_tokens is not None:
            # Prepend retrieved context to input_ids
            context_batch = retrieved_tokens.unsqueeze(0).repeat(b, 1).to(input_ids.device)
            # Truncate to max sequence length if necessary
            full_input = torch.cat([context_batch, input_ids], dim=1)[:, :self.config.max_seq_len]
        else:
            full_input = input_ids

        return self.transformer(full_input, return_hidden_states=return_hidden_states)
