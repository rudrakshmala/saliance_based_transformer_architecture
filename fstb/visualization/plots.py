import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List

def plot_cka_heatmap(cka_matrix: List[List[float]], save_path: str):
    """Plots 24x24 Layer-wise CKA Representation Similarity Heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        np.array(cka_matrix),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        xticklabels=range(1, 25),
        yticklabels=range(1, 25),
        cbar_kws={'label': 'CKA Similarity'}
    )
    plt.title("FSTB Layer-wise Representation Similarity (CKA)")
    plt.xlabel("Transformer Block Layer (1-24)")
    plt.ylabel("Transformer Block Layer (1-24)")
    
    # Draw red lines demarcating the 4 functional stages
    plt.axvline(x=6, color='red', linestyle='--', linewidth=1.5)
    plt.axvline(x=12, color='red', linestyle='--', linewidth=1.5)
    plt.axvline(x=18, color='red', linestyle='--', linewidth=1.5)
    plt.axhline(y=6, color='red', linestyle='--', linewidth=1.5)
    plt.axhline(y=12, color='red', linestyle='--', linewidth=1.5)
    plt.axhline(y=18, color='red', linestyle='--', linewidth=1.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def plot_attention_entropy(entropies: List[float], save_path: str):
    """Plots Attention Entropy per Transformer Block."""
    plt.figure(figsize=(9, 5))
    layers = list(range(1, len(entropies) + 1))
    plt.plot(layers, entropies, marker='o', color='#2b5c8f', linewidth=2)
    plt.axvspan(1, 6, color='#ffcccc', alpha=0.3, label='Stage A: Selection')
    plt.axvspan(6, 12, color='#ccffcc', alpha=0.3, label='Stage B: Encoding')
    plt.axvspan(12, 18, color='#ffffcc', alpha=0.3, label='Stage C: Validation')
    plt.axvspan(18, 24, color='#e6ccff', alpha=0.3, label='Stage D: Generation')

    plt.xlabel("Transformer Block Layer (1-24)")
    plt.ylabel("Attention Shannon Entropy (nats)")
    plt.title("Internal Block Specialization: Attention Entropy Across Stages")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def plot_ablation_bar_chart(ablation_results: Dict[str, Dict[str, float]], save_path: str):
    """Plots Ablation Comparison Bar Chart with Baseline & FSTB models."""
    plt.figure(figsize=(12, 6))
    names = list(ablation_results.keys())
    f1_scores = [ablation_results[k].get("memory_f1", 0.0) for k in names]

    bars = plt.bar(names, f1_scores, color='#34495e')
    if len(bars) > 0:
        bars[0].set_color('#27ae60') # Highlight Full FSTB in green

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Memory Retention F1 Score")
    plt.title("Ablation Study Matrix: Impact of Removing FSTB Architectural Components")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
