"""
probe_evaluator.py
==================
Linear probing module for mechanistic interpretability.

Trains linear probes (Logistic Regression / Ridge) on hidden representations
at each of the 24 transformer blocks to answer key specialization questions:
  1. Does Stage A (blocks 1-6) specialize in memory-worthiness selection?
  2. Does Stage B (blocks 7-12) specialize in memory-type classification?
  3. Does Stage C (blocks 13-18) specialize in contradiction detection?
  4. Does Stage D (blocks 19-24) encode temporal validity & answer synthesis?
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple


class LayerProbeEvaluator:
    """
    Trains and evaluates linear probes across all 24 layers of a model.
    Uses numpy / scikit-learn or PyTorch linear classifiers for probe training.
    """

    def __init__(self, n_layers: int = 24):
        self.n_layers = n_layers

    def train_and_eval_probes(
        self,
        hidden_states_by_layer: List[np.ndarray],  # list of 24 arrays [N, D]
        target_labels: np.ndarray,                  # array [N] of int or float
        task_name: str,
        task_type: str = "classification"            # 'classification' or 'regression'
    ) -> Dict[str, Any]:
        """
        Train a linear probe for each layer and return per-layer accuracy/score.
        """
        try:
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            has_sklearn = True
        except ImportError:
            has_sklearn = False

        layer_scores = []
        n_captured = len(hidden_states_by_layer)

        for l_idx in range(min(self.n_layers, n_captured)):
            X = hidden_states_by_layer[l_idx]  # [N, D]
            y = target_labels

            # Ensure valid 2D shapes and minimum sample count
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)

            if len(X) < 10 or len(np.unique(y)) < (2 if task_type == "classification" else 1):
                # Fallback score if data insufficient
                layer_scores.append(0.5)
                continue

            if has_sklearn:
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    if task_type == "classification":
                        clf = LogisticRegression(max_iter=200, solver="lbfgs")
                        clf.fit(X_scaled, y)
                        score = float(clf.score(X_scaled, y))
                    else:
                        reg = Ridge()
                        reg.fit(X_scaled, y)
                        score = float(reg.score(X_scaled, y))
                except Exception:
                    score = 0.5
            else:
                # Simple PyTorch linear probe fallback
                score = self._pytorch_probe_fallback(X, y, task_type)

            layer_scores.append(score)

        # Pad remaining layers if fewer than 24 captured
        while len(layer_scores) < self.n_layers:
            layer_scores.append(0.5)

        # Determine best stage (A: 0-5, B: 6-11, C: 12-17, D: 18-23)
        stage_averages = {
            "Stage A (1-6)": float(np.mean(layer_scores[0:6])),
            "Stage B (7-12)": float(np.mean(layer_scores[6:12])),
            "Stage C (13-18)": float(np.mean(layer_scores[12:18])),
            "Stage D (19-24)": float(np.mean(layer_scores[18:24])),
        }

        best_stage = max(stage_averages, key=stage_averages.get)

        return {
            "task_name": task_name,
            "layer_scores": layer_scores,
            "stage_averages": stage_averages,
            "best_stage": best_stage,
            "peak_layer": int(np.argmax(layer_scores)) + 1,
            "peak_score": float(np.max(layer_scores)),
        }

    def _pytorch_probe_fallback(self, X: np.ndarray, y: np.ndarray, task_type: str) -> float:
        """Simple PyTorch linear probe training fallback."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long if task_type == "classification" else torch.float32)

        N, D = X_t.shape
        n_classes = len(torch.unique(y_t)) if task_type == "classification" else 1

        probe = nn.Linear(D, n_classes)
        optimizer = torch.optim.SGD(probe.parameters(), lr=0.05)
        loss_fn = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()

        for _ in range(50):
            optimizer.zero_grad()
            out = probe(X_t)
            if task_type == "classification":
                loss = loss_fn(out, y_t)
            else:
                loss = loss_fn(out.squeeze(), y_t)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            out = probe(X_t)
            if task_type == "classification":
                preds = out.argmax(dim=-1)
                acc = (preds == y_t).float().mean().item()
                return float(acc)
            else:
                return float(1.0 - (loss.item() / (y_t.var().item() + 1e-6)))
