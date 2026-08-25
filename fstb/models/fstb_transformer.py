import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List, Any
from fstb.config import ModelConfig, MemoryConfig
from fstb.models.modules import RMSNorm, TransformerBlock, CrossAttentionMemoryFusion, precompute_rope_freqs
from fstb.models.memory_controller import DynamicMemoryController
from fstb.models.gating import MemoryRoutingGate
from fstb.models.stage_heads import StageAHeads, StageBHeads, StageCHeads
from fstb.models.memory_interface import MemoryObject, MemoryType

class FSTBTransformer(nn.Module):
    """
    Functionally Specialized Transformer Blocks (FSTB) Model.
    Total 24 transformer blocks explicitly partitioned into 4 functional stages:
    - Stage A (Blocks 1-6): Memory Selection Blocks
    - Stage B (Blocks 7-12): Memory Encoding Blocks
    - Stage C (Blocks 13-18): Memory Validation Blocks
    - Stage D (Blocks 19-24): Response Generation Blocks
    """
    def __init__(self, model_config: ModelConfig, memory_config: MemoryConfig):
        super().__init__()
        self.model_config = model_config
        self.memory_config = memory_config

        self.tok_embeddings = nn.Embedding(model_config.vocab_size, model_config.d_model)
        self.dropout = nn.Dropout(model_config.dropout)

        # Precompute RoPE frequencies
        cos, sin = precompute_rope_freqs(
            model_config.d_model // model_config.n_heads,
            model_config.max_seq_len,
            model_config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Stage A: Memory Selection (Blocks 1-6)
        self.stage_a_blocks = nn.ModuleList([
            TransformerBlock(model_config.d_model, model_config.n_heads, model_config.d_ff, model_config.dropout, model_config.norm_eps)
            for _ in model_config.stage_a_blocks
        ])
        self.stage_a_norm = RMSNorm(model_config.d_model, eps=model_config.norm_eps)
        self.stage_a_heads = StageAHeads(model_config.d_model, memory_config.d_mem, memory_config.num_memory_types)

        # Stage B: Memory Encoding (Blocks 7-12)
        self.stage_b_blocks = nn.ModuleList([
            TransformerBlock(model_config.d_model, model_config.n_heads, model_config.d_ff, model_config.dropout, model_config.norm_eps)
            for _ in model_config.stage_b_blocks
        ])
        self.stage_b_norm = RMSNorm(model_config.d_model, eps=model_config.norm_eps)
        self.stage_b_heads = StageBHeads(model_config.d_model, memory_config.d_mem, memory_config.d_sym, memory_config.num_memory_types)

        # Stage C: Memory Validation (Blocks 13-18)
        self.stage_c_blocks = nn.ModuleList([
            TransformerBlock(model_config.d_model, model_config.n_heads, model_config.d_ff, model_config.dropout, model_config.norm_eps)
            for _ in model_config.stage_c_blocks
        ])
        self.stage_c_norm = RMSNorm(model_config.d_model, eps=model_config.norm_eps)
        self.stage_c_heads = StageCHeads(model_config.d_model)

        # Stage D: Response Generation (Blocks 19-24)
        self.stage_d_blocks = nn.ModuleList([
            TransformerBlock(model_config.d_model, model_config.n_heads, model_config.d_ff, model_config.dropout, model_config.norm_eps)
            for _ in model_config.stage_d_blocks
        ])
        self.memory_fusion = CrossAttentionMemoryFusion(model_config.d_model, memory_config.d_mem, n_heads=4, dropout=model_config.dropout)
        self.stage_d_norm = RMSNorm(model_config.d_model, eps=model_config.norm_eps)
        self.lm_head = nn.Linear(model_config.d_model, model_config.vocab_size, bias=False)

        # Tie weights between embedding and lm_head
        self.tok_embeddings.weight = self.lm_head.weight

        # Routing Gates & External Memory Controller
        self.routing_gate = MemoryRoutingGate(model_config.d_model, memory_config.num_memory_types, memory_config.gumbel_temperature)
        self.memory_controller = DynamicMemoryController(
            d_mem=memory_config.d_mem,
            d_sym=memory_config.d_sym,
            max_slots=memory_config.max_memory_slots,
            decay_rate=memory_config.decay_rate
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        external_memory_bank: Optional[torch.Tensor] = None, # [b, num_mem, d_mem]
        return_hidden_states: bool = False,
        *,
        block_ablation_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        b, s = input_ids.shape
        x = self.tok_embeddings(input_ids)
        x = self.dropout(x)

        all_hidden_states = []
        all_attentions = []

        # --- Stage A: Memory Selection (Blocks 1-6) ---
        for block in self.stage_a_blocks:
            x, attn = block(x, rope_cos=self.rope_cos, rope_sin=self.rope_sin, attn_mask=attn_mask)
            if return_hidden_states:
                all_hidden_states.append(x)
                all_attentions.append(attn)

        stage_a_out = self.stage_a_norm(x)
        
        if block_ablation_mode == 'zero_stage_a':
            stage_a_out = torch.zeros_like(stage_a_out)
        elif block_ablation_mode == 'random_stage_a':
            stage_a_out = torch.randn_like(stage_a_out)

        stage_a_predictions = self.stage_a_heads(stage_a_out)
        routing_probs, routing_logits = self.routing_gate(stage_a_out)

        # --- Stage B: Memory Encoding (Blocks 7-12) ---
        x = stage_a_out
        for block in self.stage_b_blocks:
            x, attn = block(x, rope_cos=self.rope_cos, rope_sin=self.rope_sin, attn_mask=attn_mask)
            if return_hidden_states:
                all_hidden_states.append(x)
                all_attentions.append(attn)

        stage_b_out = self.stage_b_norm(x)

        if block_ablation_mode == 'zero_stage_b':
            stage_b_out = torch.zeros_like(stage_b_out)
        elif block_ablation_mode == 'random_stage_b':
            stage_b_out = torch.randn_like(stage_b_out)

        stage_b_predictions = self.stage_b_heads(stage_b_out)

        # Differentiable Memory Retrieval / Injection from Memory Controller
        if external_memory_bank is None:
            # Use candidate memory embeddings generated by Stage B as local bank
            candidate_bank = stage_b_predictions["content_embedding"] # [b, s, d_mem]
        else:
            candidate_bank = external_memory_bank

        retrieved_mem_rep, retrieval_attn = self.memory_controller.retrieve_differentiable(
            query_states=stage_b_out,
            memory_bank=candidate_bank,
            mode=self.memory_config.retrieval_mode,
            top_k=self.memory_config.top_k_retrieval
        )

        # --- Stage C: Memory Validation (Blocks 13-18) ---
        x = stage_b_out
        for block in self.stage_c_blocks:
            x, attn = block(x, rope_cos=self.rope_cos, rope_sin=self.rope_sin, attn_mask=attn_mask)
            if return_hidden_states:
                all_hidden_states.append(x)
                all_attentions.append(attn)

        stage_c_out = self.stage_c_norm(x)

        if block_ablation_mode == 'zero_stage_c':
            stage_c_out = torch.zeros_like(stage_c_out)
        elif block_ablation_mode == 'random_stage_c':
            stage_c_out = torch.randn_like(stage_c_out)

        stage_c_predictions = self.stage_c_heads(stage_c_out)

        # --- Stage D: Response Generation (Blocks 19-24) ---
        # Inject validated memory via Cross-Attention Fusion
        x, mem_fusion_attn = self.memory_fusion(stage_c_out, candidate_bank)

        for block in self.stage_d_blocks:
            x, attn = block(x, rope_cos=self.rope_cos, rope_sin=self.rope_sin, attn_mask=attn_mask)
            if return_hidden_states:
                all_hidden_states.append(x)
                all_attentions.append(attn)

        stage_d_out = self.stage_d_norm(x)
        logits = self.lm_head(stage_d_out)

        outputs = {
            "logits": logits,
            "stage_a": stage_a_predictions,
            "stage_b": stage_b_predictions,
            "stage_c": stage_c_predictions,
            "routing_probs": routing_probs,
            "routing_logits": routing_logits,
            "retrieved_mem_rep": retrieved_mem_rep,
            "retrieval_attn": retrieval_attn,
            "mem_fusion_attn": mem_fusion_attn
        }

        if return_hidden_states:
            outputs["hidden_states"] = all_hidden_states
            outputs["attentions"] = all_attentions

        return outputs
