"""
seed_manager.py
===============
Manages deterministic seeding and multi-seed result aggregation across runs.
"""

import random
import numpy as np
import torch
from typing import Dict, Any, List, Optional


class SeedManager:
    """Handles multi-seed execution and statistical aggregation."""

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def aggregate_seed_results(seed_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metric dictionaries from N seeds into mean, std, and 95% CIs.
        """
        if not seed_results:
            return {}

        keys = seed_results[0].keys()
        aggregated = {}

        for k in keys:
            vals = [res[k] for res in seed_results if k in res and res[k] is not None]
            if not vals:
                continue
            arr = np.array(vals)
            mean_val = float(np.mean(arr))
            std_val  = float(np.std(arr)) if len(arr) > 1 else 0.0
            ci95     = float(1.96 * std_val / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

            aggregated[k] = {
                "mean": mean_val,
                "std": std_val,
                "ci95": ci95,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "raw_values": vals
            }

        return aggregated
