"""
scaling_runner.py
=================
Scaling law runner evaluating Baseline, AuxBaseline, and FSTB across parameter scales:
  - tiny: ~4.5M params
  - small: ~17M params
  - medium: ~66M params
  - large: ~180M params
  - xl: ~300M+ params
"""

import time
import torch
from typing import Dict, Any, List, Optional
from fstb.models.model_factory import build_model_trio, count_params, verify_parameter_parity
from fstb.evaluation.evaluator import BenchmarkEvaluator


class ScalingExperimentRunner:
    """Runs parameter scaling experiments across model trios."""

    SIZES = ["tiny", "small", "medium"]

    def __init__(self, dataset, device: torch.device = None):
        self.dataset = dataset
        self.device = device or torch.device("cpu")
        self.evaluator = BenchmarkEvaluator(device=self.device)

    def run_scaling(self, sizes: Optional[List[str]] = None) -> Dict[str, Any]:
        target_sizes = sizes or self.SIZES
        scaling_results = {}

        print("\n" + "=" * 60, flush=True)
        print("  FSTB Parameter Scaling Experiment", flush=True)
        print("=" * 60, flush=True)

        for size in target_sizes:
            print(f"\n  [Scaling] Building model trio for size='{size}'...", flush=True)
            t0 = time.time()
            base, aux, fstb = build_model_trio(size)
            parity = verify_parameter_parity(base, aux, fstb)

            p_base = count_params(base)
            p_aux  = count_params(aux)
            p_fstb = count_params(fstb)
            print(f"    Params: Base={p_base:,} | Aux={p_aux:,} | FSTB={p_fstb:,} (diff={parity['diffs']['max_diff_pct']:.3f}%)", flush=True)

            print(f"  [Scaling] Evaluating size='{size}'...", flush=True)
            res_base = self.evaluator.evaluate(base, self.dataset, run_specialization=False, max_batches=2)
            res_aux  = self.evaluator.evaluate(aux,  self.dataset, run_specialization=False, max_batches=2)
            res_fstb = self.evaluator.evaluate(fstb, self.dataset, run_specialization=True,  max_batches=2)

            scaling_results[size] = {
                "params": {"baseline": p_base, "aux_baseline": p_aux, "fstb": p_fstb},
                "parity_info": parity,
                "baseline_metrics": res_base["metrics"],
                "aux_baseline_metrics": res_aux["metrics"],
                "fstb_metrics": res_fstb["metrics"],
                "eval_time_sec": round(time.time() - t0, 2)
            }
            print(f"    FSTB Memory F1: {res_fstb['metrics']['memory_f1']:.4f} | Base Memory F1: {res_base['metrics']['memory_f1']:.4f}", flush=True)

        return scaling_results
