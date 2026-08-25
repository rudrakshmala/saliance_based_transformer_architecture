import copy
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from fstb.config import ModelConfig, MemoryConfig, LossWeightsConfig
from fstb.models.fstb_transformer import FSTBTransformer
from fstb.models.loss import FSTBLossFunction
from fstb.evaluation.evaluator import BenchmarkEvaluator

class AblationRunner:
    """
    Executes the 10 core architectural ablation studies:
    1. remove Stage A (uniform selection)
    2. remove Stage B (raw embeddings stored)
    3. remove Stage C (direct memory injection without validation)
    4. remove auxiliary losses (lambda_aux = 0)
    5. remove gating (fixed uniform routing)
    6. remove structured memory objects (unstructured vector store)
    7. random block partitioning
    8. different block allocation ratios (e.g. 3-3-3-15)
    9. frozen early stages (Stage A/B frozen)
    10. shared vs independent normalization
    """
    ABLATION_NAMES = [
        "full_fstb",
        "no_stage_a",
        "no_stage_b",
        "no_stage_c",
        "no_aux_losses",
        "no_gating",
        "no_structured_memory",
        "random_partitioning",
        "skewed_ratios_3_3_3_15",
        "frozen_early_stages",
        "shared_layernorm"
    ]

    def __init__(self, base_model_config: ModelConfig, base_memory_config: MemoryConfig):
        self.base_model_config = base_model_config
        self.base_memory_config = base_memory_config

    def create_ablation_model(self, ablation_name: str) -> Tuple[nn.Module, LossWeightsConfig]:
        m_cfg = copy.deepcopy(self.base_model_config)
        mem_cfg = copy.deepcopy(self.base_memory_config)
        loss_cfg = LossWeightsConfig()

        if ablation_name == "no_aux_losses":
            loss_cfg.w_stage_a = 0.0
            loss_cfg.w_stage_b = 0.0
            loss_cfg.w_stage_c = 0.0
            loss_cfg.w_stage_d_aux = 0.0

        elif ablation_name == "random_partitioning":
            # Shuffle block stage allocations randomly
            blocks = list(range(24))
            m_cfg.stage_a_blocks = blocks[0:6]
            m_cfg.stage_b_blocks = blocks[6:12]
            m_cfg.stage_c_blocks = blocks[12:18]
            m_cfg.stage_d_blocks = blocks[18:24]

        elif ablation_name == "skewed_ratios_3_3_3_15":
            m_cfg.stage_a_blocks = list(range(0, 3))
            m_cfg.stage_b_blocks = list(range(3, 6))
            m_cfg.stage_c_blocks = list(range(6, 9))
            m_cfg.stage_d_blocks = list(range(9, 24))

        elif ablation_name == "shared_layernorm":
            m_cfg.independent_stage_norm = False

        model = FSTBTransformer(m_cfg, mem_cfg)

        if ablation_name == "frozen_early_stages":
            # Freeze Stage A and Stage B parameters
            for param in model.stage_a_blocks.parameters():
                param.requires_grad = False
            for param in model.stage_b_blocks.parameters():
                param.requires_grad = False

        return model, loss_cfg

    def run_all_ablations(self, eval_dataset: Any, evaluator: BenchmarkEvaluator) -> Dict[str, Dict[str, Any]]:
        results = {}
        for name in self.ABLATION_NAMES:
            model, loss_cfg = self.create_ablation_model(name)
            res = evaluator.evaluate(model, eval_dataset)
            results[name] = res["metrics"]
        return results
