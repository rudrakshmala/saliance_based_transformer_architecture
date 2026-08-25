# fstb/models/baselines/vanilla.py
"""Vanilla Transformer baseline for external benchmark.

Uses the same model factory as the original baseline (no auxiliary components).
"""

from fstb.models.model_factory import build_baseline_model


def get_model(config_name: str = "small"):
    """Return a vanilla transformer model.

    Parameters
    ----------
    config_name: str
        Size identifier (e.g., "small" for ~30M parameters).
    """
    # The factory builds a baseline model with the requested config.
    model = build_baseline_model(config_name)
    return model
