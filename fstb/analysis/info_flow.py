"""
info_flow.py
============
Tracks hidden-state trajectories through all 24 transformer blocks.

Computes:
  - Layer-wise representation drift (L2 distance from input)
  - Memory object evolution across Stage B blocks
  - Routing decision entropy across Stage A blocks
  - Per-token trajectory visualization data
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional


class InfoFlowAnalyzer:
    """
    Instrument a model's forward pass to capture hidden state trajectories.
    Works with both BaselineTransformer and FSTBTransformer (and AuxBaseline).
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def analyze(
        self,
        input_ids: torch.Tensor,
        max_tokens_to_track: int = 16
    ) -> Dict[str, Any]:
        """
        Run a forward pass and capture:
          - hidden_states: list of (L+1) tensors [B, T, D] (embedding + each block output)
          - layer_drift: L2 distance of each layer output from embedding
          - routing_entropy: per-step entropy of routing probs (FSTB only)
          - retrieval_confidence: retrieval attention entropy (FSTB only)
        """
        self.model.eval()
        self.model.to(self.device)
        ids = input_ids.to(self.device)[:, :max_tokens_to_track]

        outputs = self.model(ids, return_hidden_states=True)

        hidden_states: List[torch.Tensor] = outputs.get("hidden_states", [])

        # ── Layer-wise drift from initial embedding ─────────────────────
        layer_drift = []
        if hidden_states:
            h0 = hidden_states[0].detach().cpu()  # embedding / first block
            for i, h in enumerate(hidden_states):
                h_cpu = h.detach().cpu()
                drift = (h_cpu - h0).norm(dim=-1).mean().item()  # mean over B,T
                layer_drift.append(drift)

        # ── Routing entropy (FSTB-specific) ────────────────────────────
        routing_entropy = []
        if "routing_probs" in outputs:
            rp = outputs["routing_probs"].detach().cpu()  # [B, T, n_routes]
            # Entropy: -sum(p * log(p+eps))
            ent = -(rp * (rp + 1e-8).log()).sum(dim=-1).mean(dim=(0, 1)).item()
            routing_entropy = [ent]

        # ── Retrieval attention entropy (FSTB-specific) ─────────────────
        retrieval_conf = None
        if "retrieval_attn" in outputs:
            ra = outputs["retrieval_attn"]
            if ra is not None and isinstance(ra, torch.Tensor):
                ra_cpu = ra.detach().cpu()
                ent_ra = -(ra_cpu * (ra_cpu + 1e-8).log()).sum(dim=-1).mean().item()
                retrieval_conf = 1.0 - ent_ra  # higher = more confident retrieval

        # ── Build trajectory snapshot for first sample, all tokens ──────
        trajectory = []
        for layer_idx, h in enumerate(hidden_states[:24]):  # cap at 24
            h_np = h[0].detach().cpu().numpy()  # [T, D]
            trajectory.append({
                "layer": layer_idx,
                "mean_norm": float(np.linalg.norm(h_np, axis=-1).mean()),
                "mean_activation": float(h_np.mean()),
                "std_activation": float(h_np.std()),
            })

        return {
            "layer_drift": layer_drift,
            "routing_entropy": routing_entropy,
            "retrieval_confidence": retrieval_conf,
            "trajectory": trajectory,
            "num_layers_captured": len(hidden_states),
        }

    @staticmethod
    def compare_models(
        flow_a: Dict[str, Any],
        flow_b: Dict[str, Any],
        flow_c: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare information flow profiles across 3 models for visualization.
        """
        return {
            "baseline_drift": flow_a.get("layer_drift", []),
            "aux_baseline_drift": flow_b.get("layer_drift", []),
            "fstb_drift": flow_c.get("layer_drift", []),
            "fstb_routing_entropy": flow_c.get("routing_entropy", []),
            "fstb_retrieval_confidence": flow_c.get("retrieval_confidence"),
        }
