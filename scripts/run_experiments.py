"""
Streamlined FSTB experiment runner.
Avoids slow matplotlib/seaborn/scipy imports at startup.
All heavy imports deferred to when actually needed.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts numpy/Python numeric types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle Python builtin bool (subclasses int, must check first)
        if isinstance(obj, bool):
            return bool(obj)
        return super().default(obj)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_conv_sessions(raw):
    flat = []
    for user in raw:
        for sess in user.get("sessions", []):
            flat.append(sess)
    return flat


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    print("=" * 55, flush=True)
    print("  FSTB Research Framework", flush=True)
    print("=" * 55, flush=True)

    # ── Imports (kept local to avoid startup hang) ─────────────────
    print("[0/5] Loading modules...", flush=True)
    import torch
    from fstb.config import ModelConfig, MemoryConfig, LossWeightsConfig
    from fstb.models.baseline import BaselineTransformer
    from fstb.models.fstb_transformer import FSTBTransformer
    from fstb.data.long_term_conv import LongTermConversationGenerator
    from fstb.data.contradiction import ContradictionDatasetGenerator
    from fstb.data.temporal import TemporalReasoningGenerator, PreferenceEvolutionGenerator
    from fstb.data.dataset import FSTBDataset
    from fstb.evaluation.evaluator import BenchmarkEvaluator
    from fstb.ablations.ablation_runner import AblationRunner
    print("    Modules loaded.", flush=True)

    # ── Config ─────────────────────────────────────────────────────
    d = args.d_model
    model_cfg = ModelConfig(
        d_model=d, n_layers=24, n_heads=4 if d <= 128 else 8,
        d_ff=d * 4, max_seq_len=64, vocab_size=2048
    )
    mem_cfg = MemoryConfig(d_mem=d // 2, d_sym=d // 4, max_memory_slots=64)
    device  = torch.device("cpu")
    print(f"  d_model={d}  max_seq_len={model_cfg.max_seq_len}  vocab={model_cfg.vocab_size}", flush=True)

    # ── Models ─────────────────────────────────────────────────────
    print("\n[1/5] Building models...", flush=True)
    t0 = time.time()
    baseline_model = BaselineTransformer(model_cfg)
    fstb_model     = FSTBTransformer(model_cfg, mem_cfg)
    print(f"    Baseline params : {count_params(baseline_model):,}", flush=True)
    print(f"    FSTB params     : {count_params(fstb_model):,}", flush=True)
    print(f"    Build time      : {time.time()-t0:.1f}s", flush=True)

    # ── Datasets ───────────────────────────────────────────────────
    print("\n[2/5] Generating datasets...", flush=True)
    nu   = 2  if args.quick else 5
    ns   = 3  if args.quick else 20
    nsmp = 10 if args.quick else 60

    conv_flat   = flatten_conv_sessions(
        LongTermConversationGenerator(num_users=nu, num_sessions_per_user=ns, seed=42).generate()
    )
    contra_data  = ContradictionDatasetGenerator(num_samples=nsmp, seed=42).generate()
    temporal_data= TemporalReasoningGenerator(num_samples=nsmp,    seed=42).generate()
    pref_data    = PreferenceEvolutionGenerator(num_samples=nsmp,  seed=42).generate()

    all_samples = conv_flat + contra_data + temporal_data + pref_data
    dataset = FSTBDataset(all_samples, vocab_size=model_cfg.vocab_size,
                          max_seq_len=model_cfg.max_seq_len)
    print(f"    Total samples : {len(dataset)}", flush=True)

    # ── Benchmark Evaluation ───────────────────────────────────────
    print("\n[3/5] Running evaluations...", flush=True)
    evaluator = BenchmarkEvaluator(device=device)
    max_b = 3 if args.quick else 8

    t0 = time.time()
    print("  Evaluating baseline...", flush=True)
    baseline_res = evaluator.evaluate(baseline_model, dataset,
                                      run_specialization=False, max_batches=max_b)
    print(f"    Done ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    print("  Evaluating FSTB...", flush=True)
    fstb_res = evaluator.evaluate(fstb_model, dataset,
                                  run_specialization=True, max_batches=max_b)
    print(f"    Done ({time.time()-t0:.1f}s)", flush=True)

    # ── Print Results Table ────────────────────────────────────────
    print("\n  Results:", flush=True)
    keys = [
        "memory_f1", "memory_precision", "memory_recall",
        "retrieval_f1", "contradiction_detection_acc",
        "overwrite_accuracy", "stale_memory_suppression",
        "factual_consistency", "response_bleu", "response_rouge_l"
    ]
    print(f"\n  {'Metric':<38} {'Baseline':>10} {'FSTB':>10} {'Delta':>10}", flush=True)
    print("  " + "-" * 70, flush=True)
    for k in keys:
        bv = baseline_res["metrics"].get(k, 0.0)
        fv = fstb_res["metrics"].get(k, 0.0)
        sign = "+" if fv >= bv else ""
        print(f"  {k:<38} {bv:>10.4f} {fv:>10.4f} {sign}{fv-bv:>9.4f}", flush=True)

    # ── Statistical Significance ───────────────────────────────────
    print("\n[4/5] Statistical significance...", flush=True)
    # Deferred import of scipy
    from fstb.evaluation.stat_tests import StatisticalSignificanceAnalyzer
    fstb_scores     = [fstb_res["metrics"][k]     for k in ["memory_f1", "retrieval_f1",
                                                              "contradiction_detection_acc", "factual_consistency"]]
    baseline_scores = [baseline_res["metrics"][k] for k in ["memory_f1", "retrieval_f1",
                                                              "contradiction_detection_acc", "factual_consistency"]]
    stat_res = StatisticalSignificanceAnalyzer.compare_models(fstb_scores, baseline_scores)
    pv_t = stat_res['p_value_ttest']
    pv_w = stat_res['p_value_wilcoxon']
    pv_t_str = f"{pv_t:.6f}" if pv_t is not None else "NaN (identical scores — train models first)"
    print(f"  p-value (paired t-test) : {pv_t_str}", flush=True)
    print(f"  p-value (Wilcoxon)      : {pv_w:.6f}", flush=True)
    print(f"  Cohen's d               : {stat_res['cohens_d']:.4f}", flush=True)
    print(f"  Significant             : {stat_res['statistically_significant']}", flush=True)

    # ── Ablation Studies ───────────────────────────────────────────
    print("\n[5/5] Ablation studies...", flush=True)
    ablation_runner = AblationRunner(model_cfg, mem_cfg)
    abl_names = ablation_runner.ABLATION_NAMES[:4] if args.quick else ablation_runner.ABLATION_NAMES
    ablation_summary = {}
    for name in abl_names:
        t0 = time.time()
        model_abl, _ = ablation_runner.create_ablation_model(name)
        res_abl = evaluator.evaluate(model_abl, dataset,
                                     run_specialization=False, max_batches=2)
        mem_f1 = res_abl["metrics"]["memory_f1"]
        ablation_summary[name] = res_abl["metrics"]
        print(f"  [{name:<30}]  memory_f1={mem_f1:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    # ── Save Results ───────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    results_json = {
        "config": {"d_model": d, "n_layers": 24},
        "baseline": baseline_res["metrics"],
        "fstb": fstb_res["metrics"],
        "stat_tests": stat_res,
        "ablations": ablation_summary,
    }
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved: {json_path}", flush=True)

    # ── Figures (deferred matplotlib import) ──────────────────────
    print("  Generating figures...", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — no GUI needed
        from fstb.visualization.plots import (
            plot_cka_heatmap, plot_attention_entropy, plot_ablation_bar_chart
        )
        figures_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        spec = fstb_res.get("specialization", {})
        if "cka_matrix" in spec:
            plot_cka_heatmap(spec["cka_matrix"],
                             os.path.join(figures_dir, "cka_similarity_heatmap.png"))
            print("    Saved: cka_similarity_heatmap.png", flush=True)
        if "layer_entropies" in spec:
            plot_attention_entropy(spec["layer_entropies"],
                                   os.path.join(figures_dir, "attention_entropy_by_layer.png"))
            print("    Saved: attention_entropy_by_layer.png", flush=True)
        plot_ablation_bar_chart(ablation_summary,
                                os.path.join(figures_dir, "ablation_matrix_comparison.png"))
        print("    Saved: ablation_matrix_comparison.png", flush=True)

        # ── HTML Dashboard ─────────────────────────────────────────
        from fstb.visualization.dashboard import FSTBDashboardGenerator
        dashboard_path = os.path.join(args.output_dir, "dashboard.html")
        FSTBDashboardGenerator.generate_html_report(
            results_summary={"fstb": fstb_res["metrics"], "baseline": baseline_res["metrics"]},
            ablation_summary=ablation_summary,
            stat_summary=stat_res,
            output_html_path=dashboard_path
        )
        print(f"    Saved: dashboard.html", flush=True)
    except Exception as e:
        print(f"  [Warning] Figure generation skipped: {e}", flush=True)

    print("\n" + "=" * 55, flush=True)
    print(f"  COMPLETE — results in: {args.output_dir}", flush=True)
    print("=" * 55, flush=True)


if __name__ == "__main__":
    main()
