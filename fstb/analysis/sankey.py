"""
sankey.py
=========
Generates routing flow diagrams using Plotly (if available) or text-based fallback.

Produces:
  - Sankey diagram of routing probability mass across Stage A memory categories
  - Attention flow diagram across stages
  - Saves as self-contained HTML
"""

import json
from typing import Dict, Any, List, Optional
import numpy as np


def _make_sankey_html(
    sources: List[int],
    targets: List[int],
    values: List[float],
    labels: List[str],
    title: str,
) -> str:
    """Generate a Plotly Sankey diagram as self-contained HTML string."""
    data = {
        "type": "sankey",
        "orientation": "h",
        "node": {
            "pad": 15,
            "thickness": 20,
            "line": {"color": "black", "width": 0.5},
            "label": labels,
            "color": [
                "#2196F3", "#4CAF50", "#FF9800", "#E91E63",  # source nodes
                "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",  # target nodes
            ][:len(labels)],
        },
        "link": {
            "source": sources,
            "target": targets,
            "value": values,
            "color": "rgba(100,150,255,0.3)",
        },
    }

    layout = {
        "title": {"text": title, "font": {"size": 16}},
        "font": {"size": 12},
        "paper_bgcolor": "#fafafa",
    }

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.18.0.min.js"></script>
</head>
<body>
  <div id="sankey" style="width:900px;height:500px;"></div>
  <script>
    Plotly.newPlot('sankey', [{json.dumps(data)}], {json.dumps(layout)});
  </script>
</body>
</html>"""
    return html


def generate_routing_sankey(
    routing_probs: np.ndarray,
    output_path: str,
    memory_categories: List[str] = None,
) -> str:
    """
    Build a Sankey diagram showing how routing probability mass flows from
    input tokens to the 5 memory category channels.

    routing_probs: shape [T, n_routes] averaged over batch
    """
    memory_categories = memory_categories or [
        "Episodic", "Semantic", "Procedural", "Emotional", "Working"
    ]
    n_routes = routing_probs.shape[-1] if hasattr(routing_probs, "shape") else 5

    # Aggregate flow: input node -> each route node
    input_label = "Token Representations"
    labels = [input_label] + memory_categories[:n_routes]

    # Mean routing probability per category
    if hasattr(routing_probs, "mean"):
        mean_probs = routing_probs.mean(axis=0).tolist()
    else:
        mean_probs = [1.0 / n_routes] * n_routes

    sources = [0] * n_routes
    targets = list(range(1, n_routes + 1))
    values  = [float(p) * 100 for p in mean_probs]

    html = _make_sankey_html(sources, targets, values, labels,
                              "FSTB Memory Routing Flow (Stage A)")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def generate_stage_flow_sankey(
    stage_metrics: Dict[str, float],
    output_path: str,
) -> str:
    """
    Build a stage-to-stage information flow Sankey diagram.
    Node labels: Input → Stage A → Stage B → Stage C → Stage D → Output
    Edge values: relative metric contribution of each stage.
    """
    labels = ["Input", "Stage A\n(Selection)", "Stage B\n(Encoding)",
              "Stage C\n(Validation)", "Stage D\n(Generation)", "Output"]

    # Default uniform flow; replace with real values if provided
    a_imp = stage_metrics.get("stage_a_importance", 1.0)
    b_imp = stage_metrics.get("stage_b_importance", 1.0)
    c_imp = stage_metrics.get("stage_c_importance", 1.0)
    d_imp = stage_metrics.get("stage_d_importance", 1.0)

    total = a_imp + b_imp + c_imp + d_imp
    a_val = (a_imp / total) * 100
    b_val = (b_imp / total) * 100
    c_val = (c_imp / total) * 100
    d_val = (d_imp / total) * 100

    sources = [0, 1, 2, 3, 4]
    targets = [1, 2, 3, 4, 5]
    values  = [a_val + b_val + c_val + d_val, b_val + c_val + d_val,
               c_val + d_val, d_val, d_val]

    html = _make_sankey_html(sources, targets, values, labels,
                              "FSTB Information Flow Across Functional Stages")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
