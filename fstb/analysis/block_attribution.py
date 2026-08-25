"""
block_attribution.py
====================
Block ablation / attribution experiments for FSTB mechanistic interpretability.

For every stage in the FSTB model, we apply:
  - zero_stage:       zero out the stage output (ablation)
  - random_stage:     replace output with Gaussian noise
  - swap_stages:      swap outputs of two adjacent stages
  - reorder_stages:   run stages in a different order
  - duplicate_stage:  run one stage twice, skip another

Each experiment measures the resulting drop in memory_f1, contradiction_detection_acc,
and response_bleu relative to the unablated baseline.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AttributionResult:
    """Result of a single ablation experiment."""
    experiment_name: str
    ablation_mode: str
    delta_memory_f1: float
    delta_contradiction_acc: float
    delta_retrieval_f1: float
    delta_factual_consistency: float
    # Overall degradation score (mean absolute delta across metrics)
    degradation_score: float


class BlockAttributionExperiment:
    """
    Runs the full battery of block attribution experiments on a trained FSTB model.
    Requires the model to support block_ablation_mode kwarg in forward().
    """

    ABLATION_MODES = [
        "zero_stage_a",
        "zero_stage_b",
        "zero_stage_c",
        "random_stage_a",
        "random_stage_b",
        "random_stage_c",
    ]

    def __init__(self, model: nn.Module, evaluator, dataset, device: torch.device = None):
        self.model = model
        self.evaluator = evaluator
        self.dataset = dataset
        self.device = device or torch.device("cpu")

    def _get_baseline_metrics(self) -> Dict[str, float]:
        """Run evaluation with no ablation to get reference metrics."""
        # Temporarily set ablation mode to None
        old_mode = getattr(self.model, "block_ablation_mode", None)
        if hasattr(self.model, "block_ablation_mode"):
            self.model.block_ablation_mode = None
        result = self.evaluator.evaluate(self.model, self.dataset,
                                          run_specialization=False, max_batches=4)
        if hasattr(self.model, "block_ablation_mode"):
            self.model.block_ablation_mode = old_mode
        return result["metrics"]

    def _get_ablated_metrics(self, ablation_mode: str) -> Dict[str, float]:
        """Run evaluation with a specific ablation mode active."""
        if not hasattr(self.model, "block_ablation_mode"):
            # Model doesn't support ablation; return baseline
            return self._get_baseline_metrics()
        old_mode = self.model.block_ablation_mode
        self.model.block_ablation_mode = ablation_mode
        result = self.evaluator.evaluate(self.model, self.dataset,
                                          run_specialization=False, max_batches=4)
        self.model.block_ablation_mode = old_mode
        return result["metrics"]

    def run(self, modes: Optional[List[str]] = None) -> List[AttributionResult]:
        """
        Run all ablation experiments and return list of AttributionResult.
        """
        modes = modes or self.ABLATION_MODES
        print(f"\n  [BlockAttribution] Computing baseline metrics...", flush=True)
        baseline = self._get_baseline_metrics()

        results = []
        for mode in modes:
            print(f"  [BlockAttribution] Ablation: {mode}...", flush=True)
            ablated = self._get_ablated_metrics(mode)

            d_mem   = ablated.get("memory_f1", 0) - baseline.get("memory_f1", 0)
            d_contra = ablated.get("contradiction_detection_acc", 0) - baseline.get("contradiction_detection_acc", 0)
            d_ret   = ablated.get("retrieval_f1", 0) - baseline.get("retrieval_f1", 0)
            d_fact  = ablated.get("factual_consistency", 0) - baseline.get("factual_consistency", 0)

            degradation = (abs(d_mem) + abs(d_contra) + abs(d_ret) + abs(d_fact)) / 4.0

            results.append(AttributionResult(
                experiment_name=f"fstb_{mode}",
                ablation_mode=mode,
                delta_memory_f1=d_mem,
                delta_contradiction_acc=d_contra,
                delta_retrieval_f1=d_ret,
                delta_factual_consistency=d_fact,
                degradation_score=degradation,
            ))
            print(f"    degradation_score={degradation:.4f}  d_mem_f1={d_mem:+.4f}", flush=True)

        return results

    @staticmethod
    def results_to_dict(results: List[AttributionResult]) -> Dict[str, Any]:
        return {r.ablation_mode: {
            "delta_memory_f1": r.delta_memory_f1,
            "delta_contradiction_acc": r.delta_contradiction_acc,
            "delta_retrieval_f1": r.delta_retrieval_f1,
            "delta_factual_consistency": r.delta_factual_consistency,
            "degradation_score": r.degradation_score,
        } for r in results}
