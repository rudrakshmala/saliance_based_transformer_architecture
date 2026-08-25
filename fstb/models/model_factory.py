from typing import Tuple, Dict
from fstb.config import ModelConfig, MemoryConfig
from fstb.models.baseline import BaselineTransformer
from fstb.models.aux_baseline import AuxBaselineTransformer
from fstb.models.fstb_transformer import FSTBTransformer
import torch.nn as nn

def get_model_configs(size: str) -> Tuple[ModelConfig, MemoryConfig]:
    size = size.lower()
    
    if size == "tiny":
        model_cfg = ModelConfig(d_model=128, n_heads=4, n_layers=24, d_ff=256, max_seq_len=64)
        mem_cfg = MemoryConfig(d_mem=64, d_sym=32)
    elif size == "small":
        model_cfg = ModelConfig(d_model=256, n_heads=8, n_layers=24, d_ff=512, max_seq_len=128)
        mem_cfg = MemoryConfig(d_mem=128, d_sym=64)
    elif size == "medium":
        model_cfg = ModelConfig(d_model=512, n_heads=8, n_layers=24, d_ff=1024, max_seq_len=256)
        mem_cfg = MemoryConfig(d_mem=256, d_sym=128)
    elif size == "large":
        model_cfg = ModelConfig(d_model=768, n_heads=12, n_layers=24, d_ff=2048, max_seq_len=512)
        mem_cfg = MemoryConfig(d_mem=384, d_sym=192)
    elif size == "xl":
        model_cfg = ModelConfig(d_model=1024, n_heads=16, n_layers=24, d_ff=4096, max_seq_len=512)
        mem_cfg = MemoryConfig(d_mem=512, d_sym=256)
    else:
        raise ValueError(f"Unknown size: {size}")
        
    return model_cfg, mem_cfg

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_parameter_parity(a: nn.Module, b: nn.Module, c: nn.Module, threshold: float = 0.02) -> Dict:
    params_a = count_params(a)
    params_b = count_params(b)
    params_c = count_params(c)
    
    max_params = max(params_a, params_b, params_c)
    min_params = min(params_a, params_b, params_c)
    
    diff = (max_params - min_params) / max_params
    
    return {
        "parity_ok": diff <= threshold,
        "diffs": {
            "baseline": params_a,
            "aux_baseline": params_b,
            "fstb": params_c,
            "max_diff_pct": diff * 100
        }
    }

def build_model_trio(size: str) -> Tuple[BaselineTransformer, AuxBaselineTransformer, FSTBTransformer]:
    model_cfg, mem_cfg = get_model_configs(size)
    
    fstb = FSTBTransformer(model_config=model_cfg, memory_config=mem_cfg)
    params_fstb = count_params(fstb)
    
    # 1. Build Baseline matching FSTB params
    baseline = BaselineTransformer(config=model_cfg)
    params_base = count_params(baseline)
    
    if abs(params_base - params_fstb) / params_fstb > 0.01:
        # Fine-tune d_ff to get baseline as close as possible to FSTB
        # SwiGLU / FeedForward in BaselineTransformer uses 3 * d_model * d_ff per block (w1, w2, w3)
        # Total per layer = 3 * d_model * d_ff
        # 24 layers = 72 * d_model * d_ff
        param_diff = params_fstb - params_base
        d_ff_delta = round(param_diff / (3 * model_cfg.d_model * model_cfg.n_layers))
        new_d_ff = max(1, model_cfg.d_ff + d_ff_delta)
        base_cfg = ModelConfig(**{**model_cfg.__dict__, "d_ff": new_d_ff})
        baseline = BaselineTransformer(config=base_cfg)

    # 2. Build AuxBaseline matching FSTB params
    aux = AuxBaselineTransformer(model_config=model_cfg, memory_config=mem_cfg, d_bottleneck=0)
    params_aux = count_params(aux)
    
    if abs(params_aux - params_fstb) / params_fstb > 0.01:
        # Adjust d_bottleneck in AuxBaseline
        param_diff = params_fstb - params_aux
        if param_diff > 0:
            d_bottleneck = max(1, param_diff // (2 * model_cfg.d_model + 2))
            aux = AuxBaselineTransformer(model_config=model_cfg, memory_config=mem_cfg, d_bottleneck=d_bottleneck)

    return baseline, aux, fstb


# Additional baseline models for external benchmarking
from fstb.models.baselines.rag_baseline import RAGBaselineTransformer
from fstb.models.baselines.vector_memory import VectorMemoryBaselineTransformer
from fstb.models.baselines.summarization_baseline import SummarizationBaselineTransformer
from fstb.models.baselines.aux_only import get_model as get_aux_only_model

def build_all_models(size: str) -> Dict[str, Any]:
    """Build a collection of all baseline and FSTB models for a given size.

    Returns a dictionary with keys:
        'baseline'        : BaselineTransformer
        'aux_baseline'    : AuxBaselineTransformer
        'fstb'            : FSTBTransformer
        'rag'             : RAGBaselineTransformer
        'vector_memory'   : VectorMemoryBaselineTransformer
        'summarization'   : SummarizationBaselineTransformer
    All models share the same ``ModelConfig`` and ``MemoryConfig``.
    """
    model_cfg, mem_cfg = get_model_configs(size)
    # Core trio
    baseline, aux, fstb = build_model_trio(size)
    # External baselines – re‑use the same configs to keep parameter parity
    rag = RAGBaselineTransformer(model_cfg, mem_cfg)
    vector_mem = VectorMemoryBaselineTransformer(model_cfg, mem_cfg)
    summarization = SummarizationBaselineTransformer(model_cfg, mem_cfg)
    # Aux‑only convenience wrapper (same as aux model, kept for API parity)
    aux_only = get_aux_only_model(size)
    return {
        "baseline": baseline,
        "aux_baseline": aux,
        "fstb": fstb,
        "rag": rag,
        "vector_memory": vector_mem,
        "summarization": summarization,
        "aux_only": aux_only,
    }
}
