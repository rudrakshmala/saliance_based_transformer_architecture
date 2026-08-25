# Robustness evaluation suite for Phase 3
# ------------------------------------------------
# This module provides a lightweight wrapper to assess model robustness under
# a variety of perturbations that are relevant for memory‑centric language
# modelling.  The implementation re‑uses the existing BenchmarkEvaluator to
# compute the full metric suite on a perturbed version of the input dataset.
# Each method returns a dictionary of metrics (the same keys produced by
# BenchmarkEvaluator.evaluate).

import copy
from typing import Any, Dict, List

import torch

from fstb.evaluation.evaluator import BenchmarkEvaluator


class RobustnessEvaluator:
    """Utility class to evaluate model robustness.

    Each method creates a perturbed copy of the original ``FSTBDataset`` and
    forwards it to :class:`BenchmarkEvaluator`.  The returned dictionary mirrors
    the ``metrics`` entry of the standard evaluation pipeline.
    """

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.base_evaluator = BenchmarkEvaluator(device=self.device)

    def _perturb_dataset(self, dataset: Any, transform) -> Any:
        """Clone a dataset and apply ``transform`` to its ``input_ids``.

        ``transform`` receives a ``torch.Tensor`` and must return a new tensor of
        identical shape.  The original ``dataset`` is deep‑copied to avoid side‑
        effects.
        """
        perturbed = copy.deepcopy(dataset)
        perturbed.input_ids = transform(perturbed.input_ids)
        return perturbed

    # ---------------------------------------------------------------------
    # 1. Noisy conversations – inject random token ids uniformly sampled from
    #    the model vocabulary.
    # ---------------------------------------------------------------------
    def evaluate_noise(self, model, dataset, noise_ratio: float = 0.1) -> Dict[str, float]:
        vocab_size = getattr(model.config, "vocab_size", 50257)
        def add_noise(tensor: torch.Tensor) -> torch.Tensor:
            mask = torch.rand_like(tensor, dtype=torch.float) < noise_ratio
            random_ids = torch.randint(0, vocab_size, tensor.shape, device=tensor.device)
            return torch.where(mask, random_ids, tensor)
        perturbed = self._perturb_dataset(dataset, add_noise)
        return self.base_evaluator.evaluate(model, perturbed, run_specialization=False, max_batches=5)["metrics"]

    # ---------------------------------------------------------------------
    # 2. Ambiguous memory updates – replace specified slots with an UNK token.
    # ---------------------------------------------------------------------
    def evaluate_ambiguity(self, model, dataset, ambiguous_slots: List[int] | None = None) -> Dict[str, float]:
        ambiguous_slots = ambiguous_slots or []
        def mask_ambiguity(tensor: torch.Tensor) -> torch.Tensor:
            for slot in ambiguous_slots:
                if 0 <= slot < tensor.size(1):
                    tensor[:, slot] = 0  # token id 0 assumed to be UNK
            return tensor
        perturbed = self._perturb_dataset(dataset, mask_ambiguity)
        return self.base_evaluator.evaluate(model, perturbed, run_specialization=False, max_batches=5)["metrics"]

    # ---------------------------------------------------------------------
    # 3. Adversarial injection – prepend a short phrase to each input.
    # ---------------------------------------------------------------------
    def evaluate_adversarial(self, model, dataset, phrase: str) -> Dict[str, float]:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            adv_ids = torch.tensor(tokenizer.encode(phrase), device=dataset.input_ids.device)
        else:
            adv_ids = torch.tensor([9999], device=dataset.input_ids.device)
        def prepend_adv(tensor: torch.Tensor) -> torch.Tensor:
            adv_len = adv_ids.size(0)
            trimmed = tensor[:, :-adv_len]
            return torch.cat([adv_ids.expand(tensor.size(0), -1), trimmed], dim=1)
        perturbed = self._perturb_dataset(dataset, prepend_adv)
        return self.base_evaluator.evaluate(model, perturbed, run_specialization=False, max_batches=5)["metrics"]

    # ---------------------------------------------------------------------
    # 4. Truncation – zero‑pad the tail of the sequence to simulate context loss.
    # ---------------------------------------------------------------------
    def evaluate_truncation(self, model, dataset, trunc_ratio: float = 0.2) -> Dict[str, float]:
        def truncate(tensor: torch.Tensor) -> torch.Tensor:
            seq_len = tensor.size(1)
            keep_len = int(seq_len * (1 - trunc_ratio))
            tensor[:, keep_len:] = 0
            return tensor
        perturbed = self._perturb_dataset(dataset, truncate)
        return self.base_evaluator.evaluate(model, perturbed, run_specialization=False, max_batches=5)["metrics"]

    # ---------------------------------------------------------------------
    # 5. Retrieval noise – replace a fraction of memory slot targets with
    #    random distractor IDs.
    # ---------------------------------------------------------------------
    def evaluate_retrieval_noise(self, model, dataset, distractor_ids: List[int]) -> Dict[str, float]:
        perturbed = copy.deepcopy(dataset)
        if hasattr(perturbed, "memory_type_target"):
            mask = torch.rand_like(perturbed.memory_type_target, dtype=torch.float) < 0.1
            distractors = torch.tensor(distractor_ids, device=perturbed.memory_type_target.device)
            rand_idx = torch.randint(0, len(distractor_ids), perturbed.memory_type_target.shape, device=perturbed.memory_type_target.device)
            perturbed.memory_type_target = torch.where(mask, distractors[rand_idx], perturbed.memory_type_target)
        return self.base_evaluator.evaluate(model, perturbed, run_specialization=False, max_batches=5)["metrics"]

# End of robustness evaluation utilities.
