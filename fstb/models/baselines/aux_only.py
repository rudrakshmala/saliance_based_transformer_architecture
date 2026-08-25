# fstb/models/baselines/aux_only.py
"""Auxiliary‑only baseline wrapper.

Provides a convenient ``get_model`` function that returns the ``AuxBaselineTransformer``
used in the core experiments. This file exists so that external benchmark code can
import a uniform ``baselines`` package without needing to know the internal module
structure.
"""

from fstb.models.aux_baseline import AuxBaselineTransformer
from fstb.config import ModelConfig, MemoryConfig

def get_model(config_name: str = "small") -> AuxBaselineTransformer:
    """Return an AuxBaselineTransformer.

    Parameters
    ----------
    config_name: str
        Size identifier (e.g., "small" for ~30 M parameters).
    """
    # Re‑use the model factory to obtain matching configs
    from fstb.models.model_factory import get_model_configs
    model_cfg, mem_cfg = get_model_configs(config_name)
    return AuxBaselineTransformer(model_cfg, mem_cfg)
