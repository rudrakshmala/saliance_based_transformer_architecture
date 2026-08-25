"""
latex_tables.py
===============
Generates publication-quality LaTeX booktabs tables for arXiv paper submission.

Outputs:
  1. Primary Model Comparison Table (Baseline vs AuxBaseline vs FSTB)
  2. Statistical Significance Table (p-values, Cohen's d, CIs)
  3. Ablation Matrix Table (Full FSTB vs 10 Ablation Conditions)
  4. Layer Probing Accuracy Table (Peak layer & accuracy per stage)
  5. Parameter Parity & Compute Budget Table
"""

from typing import Dict, Any, List, Optional


def generate_primary_comparison_table(
    baseline_metrics: Dict[str, float],
    aux_metrics: Dict[str, float],
    fstb_metrics: Dict[str, float]
) -> str:
    """Generate LaTeX table comparing Baseline, AuxBaseline, and FSTB."""
    metrics_to_show = [
        ("Memory Precision", "memory_precision"),
        ("Memory Recall", "memory_recall"),
        ("Memory F1", "memory_f1"),
        ("Mean Reciprocal Rank (MRR)", "mrr"),
        ("Recall@1", "recall_at_1"),
        ("Recall@3", "recall_at_3"),
        ("Overwrite Accuracy", "overwrite_accuracy"),
        ("Stale Memory Suppression", "stale_memory_suppression"),
        ("Contradiction Detection Acc.", "contradiction_detection_acc"),
        ("Contradiction Resolution Acc.", "contradiction_resolution_acc"),
        ("Factual Consistency", "factual_consistency"),
        ("Response BLEU", "response_bleu"),
    ]

    latex = r"""\begin{table}[htbp]
\centering
\caption{\textbf{Primary Performance Comparison across Controlled Models.} Comparison of standard 24-layer baseline (Model A), auxiliary-supervised baseline (Model B), and FSTB (Model C) under equal parameter count and compute budget.}
\label{tab:primary_comparison}
\begin{tabular}{l c c c c}
\toprule
\textbf{Metric} & \textbf{Model A (Base)} & \textbf{Model B (AuxBase)} & \textbf{Model C (FSTB)} & \textbf{$\Delta$ (C vs B)} \\
\midrule
"""
    for name, key in metrics_to_show:
        av = baseline_metrics.get(key, 0.0)
        bv = aux_metrics.get(key, 0.0)
        cv = fstb_metrics.get(key, 0.0)
        delta = cv - bv
        sign = "+" if delta >= 0 else ""

        # Highlight best in bold
        best_val = max(av, bv, cv)
        av_str = f"\\textbf{{{av:.4f}}}" if abs(av - best_val) < 1e-5 else f"{av:.4f}"
        bv_str = f"\\textbf{{{bv:.4f}}}" if abs(bv - best_val) < 1e-5 else f"{bv:.4f}"
        cv_str = f"\\textbf{{{cv:.4f}}}" if abs(cv - best_val) < 1e-5 else f"{cv:.4f}"

        latex += f"{name:<30} & {av_str} & {bv_str} & {cv_str} & {sign}{delta:.4f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_stat_significance_table(stat_res: Dict[str, Any]) -> str:
    """Generate LaTeX table for statistical significance tests."""
    pv_t = stat_res.get("p_value_ttest")
    pv_w = stat_res.get("p_value_wilcoxon", 1.0)
    d = stat_res.get("cohens_d", 0.0)
    sig = stat_res.get("statistically_significant", False)

    pv_t_str = f"{pv_t:.6f}" if pv_t is not None else "N/A"

    latex = r"""\begin{table}[htbp]
\centering
\caption{\textbf{Statistical Significance and Effect Size Analysis.} Results of paired t-tests, Wilcoxon signed-rank tests, and Cohen's d effect sizes across multi-seed evaluations.}
\label{tab:statistical_significance}
\begin{tabular}{l c c c c}
\toprule
\textbf{Comparison} & \textbf{Paired t-test ($p$)} & \textbf{Wilcoxon ($p$)} & \textbf{Cohen's d} & \textbf{Significant ($p<0.05$)} \\
\midrule
"""
    latex += f"FSTB vs. Baseline & {pv_t_str} & {pv_w:.6f} & {d:.4f} & {'Yes' if sig else 'No'} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_ablation_table(ablation_summary: Dict[str, Dict[str, float]]) -> str:
    """Generate LaTeX table for the 10-condition ablation matrix."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{\textbf{Ablation Matrix Results.} Functional impact of removing or modifying specialized stage components on key benchmark tasks.}
\label{tab:ablation_matrix}
\begin{tabular}{l c c c c}
\toprule
\textbf{Ablation Condition} & \textbf{Memory F1} & \textbf{Contradiction Acc.} & \textbf{Retrieval F1} & \textbf{Factual Const.} \\
\midrule
"""
    for name, metrics in ablation_summary.items():
        clean_name = name.replace("_", " ").title()
        mf1 = metrics.get("memory_f1", 0.0)
        cacc = metrics.get("contradiction_detection_acc", 0.0)
        rf1 = metrics.get("retrieval_f1", 0.0)
        fc = metrics.get("factual_consistency", 0.0)

        latex += f"{clean_name:<30} & {mf1:.4f} & {cacc:.4f} & {rf1:.4f} & {fc:.4f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_parameter_parity_table(parity_info: Dict[str, Any]) -> str:
    """Generate LaTeX table documenting parameter budget parity."""
    diffs = parity_info.get("diffs", {})

    latex = r"""\begin{table}[htbp]
\centering
\caption{\textbf{Parameter Parity Verification.} Exact parameter counts and relative percentage differences across controlled model variants.}
\label{tab:parameter_parity}
\begin{tabular}{l c c}
\toprule
\textbf{Model Architecture} & \textbf{Parameter Count} & \textbf{Relative Parity Difference (\%)} \\
\midrule
"""
    p_base = diffs.get("baseline", 0)
    p_aux = diffs.get("aux_baseline", 0)
    p_fstb = diffs.get("fstb", 0)

    diff_pct = diffs.get("max_diff_pct", 0.0)

    latex += f"Model A (Baseline) & {p_base:,} & --- \\\\\n"
    latex += f"Model B (AuxBaseline) & {p_aux:,} & {abs(p_aux - p_fstb)/max(1, p_fstb)*100:.3f}\\% \\\\\n"
    latex += f"Model C (FSTB) & {p_fstb:,} & {abs(p_base - p_fstb)/max(1, p_fstb)*100:.3f}\\% \\\\\n"
    latex += r"""\midrule
"""
    latex += f"\\textbf{{Maximum Difference}} & \\textbf{{---}} & \\textbf{{{diff_pct:.3f}\\% (Threshold: $<$2.0\\%)}} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex
