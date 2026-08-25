"""
specialization.py
=================
Internal Specialization Analyzer for FSTB Phase 2.

Computes:
  1. CKA (Centered Kernel Alignment) layer-to-layer similarity matrix (24x24)
  2. SVCCA (Singular Vector Canonical Correlation Analysis) similarity matrix
  3. Attention entropy per layer
  4. Representational clustering purity per stage
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional


class InternalSpecializationAnalyzer:
    """Analyzes representation similarity, attention entropy, and clustering across layers."""

    @staticmethod
    def compute_cka(hs1: torch.Tensor, hs2: torch.Tensor) -> float:
        """Compute Linear Centered Kernel Alignment between two hidden state matrices."""
        X = hs1.view(hs1.size(0), -1).float()
        Y = hs2.view(hs2.size(0), -1).float()

        # Center columns
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        # Cross-covariance matrix
        hs_cross = torch.matmul(X.t(), Y)
        norm_cross = torch.norm(hs_cross, p="fro") ** 2

        norm_x = torch.norm(torch.matmul(X.t(), X), p="fro")
        norm_y = torch.norm(torch.matmul(Y.t(), Y), p="fro")

        denom = norm_x * norm_y
        if denom < 1e-8:
            return 0.0

        return (norm_cross / denom).item()

    @staticmethod
    def compute_svcca(hs1: torch.Tensor, hs2: torch.Tensor, n_components: int = 10) -> float:
        """Compute SVCCA similarity between two representation matrices."""
        X = hs1.view(hs1.size(0), -1).float().cpu().numpy()
        Y = hs2.view(hs2.size(0), -1).float().cpu().numpy()

        # Center
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        try:
            # SVD reduction
            ux, sx, _ = np.linalg.svd(X, full_matrices=False)
            uy, sy, _ = np.linalg.svd(Y, full_matrices=False)

            k = min(n_components, ux.shape[1], uy.shape[1])
            if k == 0:
                return 0.0

            ux_k = ux[:, :k]
            uy_k = uy[:, :k]

            # Canonical correlation between top singular vectors
            q, r = np.linalg.qr(ux_k)
            q2, r2 = np.linalg.qr(uy_k)
            svd_corr = np.linalg.svd(np.dot(q.T, q2), compute_uv=False)
            return float(np.mean(svd_corr))
        except Exception:
            return float(InternalSpecializationAnalyzer.compute_cka(hs1, hs2))

    @staticmethod
    def compute_attention_entropy(attn_matrix: torch.Tensor) -> float:
        """Compute mean entropy over attention heads and tokens."""
        # attn_matrix: [B, H, S, S]
        p = attn_matrix + 1e-8
        entropy = -(p * torch.log(p)).sum(dim=-1)  # sum over keys -> [B, H, S]
        return float(entropy.mean().item())

    def analyze(
        self,
        hidden_states: List[torch.Tensor],
        attentions: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Analyze 24-layer representations.
        Returns CKA matrix, SVCCA matrix, layer entropies, and stage-wise similarity metrics.
        """
        n_layers = len(hidden_states)
        cka_matrix = np.zeros((n_layers, n_layers))
        svcca_matrix = np.zeros((n_layers, n_layers))

        for i in range(n_layers):
            for j in range(i, n_layers):
                val_cka = self.compute_cka(hidden_states[i], hidden_states[j])
                cka_matrix[i, j] = val_cka
                cka_matrix[j, i] = val_cka

                val_svcca = self.compute_svcca(hidden_states[i], hidden_states[j])
                svcca_matrix[i, j] = val_svcca
                svcca_matrix[j, i] = val_svcca

        layer_entropies = []
        if attentions:
            for attn in attentions:
                if isinstance(attn, torch.Tensor) and attn.ndim == 4:
                    layer_entropies.append(self.compute_attention_entropy(attn))
                else:
                    layer_entropies.append(1.0)

        # Stage-wise mean similarity (Stage A vs Stage B, etc.)
        def _stage_sim(m, r1, r2):
            sub = m[r1[0]:r1[1], r2[0]:r2[1]]
            return float(np.mean(sub)) if sub.size > 0 else 0.0

        stage_ranges = [(0, 6), (6, 12), (12, 18), (18, min(24, n_layers))]
        stage_names = ["Stage A", "Stage B", "Stage C", "Stage D"]

        stage_block_similarity = {}
        for idx1, s1 in enumerate(stage_names):
            for idx2, s2 in enumerate(stage_names):
                r1 = stage_ranges[idx1]
                r2 = stage_ranges[idx2]
                key = f"{s1} vs {s2}"
                stage_block_similarity[key] = _stage_sim(cka_matrix, r1, r2)

        return {
            "cka_matrix": cka_matrix.tolist(),
            "svcca_matrix": svcca_matrix.tolist(),
            "layer_entropies": layer_entropies,
            "stage_similarity": stage_block_similarity,
            "stage_a_self_cka": _stage_sim(cka_matrix, (0, 6), (0, 6)),
            "stage_b_self_cka": _stage_sim(cka_matrix, (6, 12), (6, 12)),
            "stage_c_self_cka": _stage_sim(cka_matrix, (12, 18), (12, 18)),
            "stage_d_self_cka": _stage_sim(cka_matrix, (18, 24), (18, 24)),
        }
