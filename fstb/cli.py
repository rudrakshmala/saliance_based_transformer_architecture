import argparse
import os
import json
import torch
from fstb.config import ModelConfig, MemoryConfig, LossWeightsConfig, TrainingConfig
from fstb.models.baseline import BaselineTransformer
from fstb.models.fstb_transformer import FSTBTransformer
from fstb.data.long_term_conv import LongTermConversationGenerator
from fstb.data.contradiction import ContradictionDatasetGenerator
from fstb.data.temporal import TemporalReasoningGenerator, PreferenceEvolutionGenerator
from fstb.data.dataset import FSTBDataset
from fstb.evaluation.evaluator import BenchmarkEvaluator
from fstb.evaluation.stat_tests import StatisticalSignificanceAnalyzer
from fstb.ablations.ablation_runner import AblationRunner
from fstb.visualization.plots import plot_cka_heatmap, plot_attention_entropy, plot_ablation_bar_chart
from fstb.visualization.dashboard import FSTBDashboardGenerator


def flatten_conv_sessions(raw: list) -> list:
    """Flatten nested long-term conversation data: {user, sessions} -> flat session list."""
    flat = []
    for user in raw:
        for sess in user.get("sessions", []):
            flat.append(sess)
    return flat


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="FSTB Experimental Framework CLI")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["train", "eval", "ablate", "all"],
                        help="Which phase to run")
    parser.add_argument("--d-model", type=int, default=256,
                        help="Hidden dimension size")
    parser.add_argument("--n-layers", type=int, default=24,
                        help="Total transformer layers")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory for figures and dashboard")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: fewer samples, fewer ablations")
    args = parser.parse_args()

    device = torch.device("cpu")
    print("=" * 55)
    print("  FSTB Research Framework — Experiment Pipeline")
    print("=" * 55)
    print(f"  Device  : {device}")
    print(f"  d_model : {args.d_model}   n_layers: {args.n_layers}")
    print(f"  Mode    : {args.mode}  Quick: {args.quick}")
    print("=" * 55)

    model_cfg = ModelConfig(d_model=args.d_model, n_layers=args.n_layers)
    mem_cfg   = MemoryConfig(d_mem=args.d_model // 2, d_sym=args.d_model // 4)

    # ── Model instantiation ────────────────────────────────────────────
    baseline_model = BaselineTransformer(model_cfg)
    fstb_model     = FSTBTransformer(model_cfg, mem_cfg)

    baseline_params = count_params(baseline_model)
    fstb_params     = count_params(fstb_model)
    print(f"\n  Baseline parameters : {baseline_params:,}")
    print(f"  FSTB parameters     : {fstb_params:,}")

    # ── Dataset generation ─────────────────────────────────────────────
    num_users   = 2 if args.quick else 5
    num_sess    = 3 if args.quick else 20
    num_samples = 10 if args.quick else 80

    print(f"\n[1/5] Generating synthetic benchmark datasets ...")
    conv_nested  = LongTermConversationGenerator(num_users=num_users,
                                                  num_sessions_per_user=num_sess,
                                                  seed=42).generate()
    conv_flat    = flatten_conv_sessions(conv_nested)
    contra_data  = ContradictionDatasetGenerator(num_samples=num_samples, seed=42).generate()
    temporal_data = TemporalReasoningGenerator(num_samples=num_samples, seed=42).generate()
    pref_data    = PreferenceEvolutionGenerator(num_samples=num_samples, seed=42).generate()

    all_samples = conv_flat + contra_data + temporal_data + pref_data
    dataset = FSTBDataset(all_samples, vocab_size=model_cfg.vocab_size,
                          max_seq_len=model_cfg.max_seq_len)
    print(f"    Total samples  : {len(dataset)}")

    # ── Benchmark evaluation ───────────────────────────────────────────
    print(f"\n[2/5] Running benchmark evaluations ...")
    evaluator = BenchmarkEvaluator(device=device)

    baseline_res = evaluator.evaluate(baseline_model, dataset,
                                      run_specialization=False, max_batches=4)
    fstb_res     = evaluator.evaluate(fstb_model, dataset,
                                      run_specialization=True,  max_batches=4)

    print(f"\n  {'Metric':<35} {'Baseline':>10} {'FSTB':>10}")
    print("  " + "-" * 57)
    metrics_to_print = [
        "memory_f1", "memory_precision", "memory_recall",
        "retrieval_f1", "contradiction_detection_acc",
        "overwrite_accuracy", "stale_memory_suppression",
        "factual_consistency", "response_bleu", "response_rouge_l"
    ]
    for m in metrics_to_print:
        bv = baseline_res["metrics"].get(m, 0.0)
        fv = fstb_res["metrics"].get(m, 0.0)
        delta_str = f"(+{fv-bv:.4f})" if fv >= bv else f"({fv-bv:.4f})"
        print(f"  {m:<35} {bv:>10.4f} {fv:>10.4f}  {delta_str}")

    # ── Statistical significance ───────────────────────────────────────
    print(f"\n[3/5] Computing statistical significance ...")
    fstb_scores     = [fstb_res["metrics"]["memory_f1"],
                       fstb_res["metrics"]["contradiction_detection_acc"],
                       fstb_res["metrics"]["factual_consistency"],
                       fstb_res["metrics"]["response_bleu"]]
    baseline_scores = [baseline_res["metrics"]["memory_f1"],
                       baseline_res["metrics"]["contradiction_detection_acc"],
                       baseline_res["metrics"]["factual_consistency"],
                       baseline_res["metrics"]["response_bleu"]]

    stat_res = StatisticalSignificanceAnalyzer.compare_models(fstb_scores, baseline_scores)
    print(f"  p-value (paired t-test)   : {stat_res['p_value_ttest']:.6f}")
    print(f"  p-value (Wilcoxon)        : {stat_res['p_value_wilcoxon']:.6f}")
    print(f"  Cohen's d effect size     : {stat_res['cohens_d']:.4f}")
    print(f"  Statistically significant : {stat_res['statistically_significant']}")

    # ── Ablation studies ───────────────────────────────────────────────
    print(f"\n[4/5] Running ablation matrix ...")
    ablation_runner = AblationRunner(model_cfg, mem_cfg)
    # Limit ablations in quick mode, run all 11 in full mode
    ablation_names = (ablation_runner.ABLATION_NAMES[:4]
                      if args.quick else ablation_runner.ABLATION_NAMES)
    ablation_summary = {}
    for name in ablation_names:
        model_abl, _ = ablation_runner.create_ablation_model(name)
        res_abl = evaluator.evaluate(model_abl, dataset,
                                     run_specialization=False, max_batches=2)
        ablation_summary[name] = res_abl["metrics"]
        print(f"  [{name}]  memory_f1={res_abl['metrics']['memory_f1']:.4f}")

    # ── Figures & Dashboard ────────────────────────────────────────────
    print(f"\n[5/5] Generating figures and HTML dashboard ...")
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    spec = fstb_res.get("specialization", {})
    if "cka_matrix" in spec:
        cka_path = os.path.join(figures_dir, "cka_similarity_heatmap.png")
        plot_cka_heatmap(spec["cka_matrix"], cka_path)
        print(f"  Saved: {cka_path}")

    if "layer_entropies" in spec:
        entr_path = os.path.join(figures_dir, "attention_entropy_by_layer.png")
        plot_attention_entropy(spec["layer_entropies"], entr_path)
        print(f"  Saved: {entr_path}")

    abl_path = os.path.join(figures_dir, "ablation_matrix_comparison.png")
    plot_ablation_bar_chart(ablation_summary, abl_path)
    print(f"  Saved: {abl_path}")

    dashboard_path = os.path.join(args.output_dir, "dashboard.html")
    FSTBDashboardGenerator.generate_html_report(
        results_summary={"fstb": fstb_res["metrics"],
                         "baseline": baseline_res["metrics"]},
        ablation_summary=ablation_summary,
        stat_summary=stat_res,
        output_html_path=dashboard_path
    )
    print(f"  Saved: {dashboard_path}")

    # ── JSON results dump ──────────────────────────────────────────────
    results_json = {
        "baseline": baseline_res["metrics"],
        "fstb": fstb_res["metrics"],
        "stat_tests": stat_res,
        "ablations": ablation_summary,
        "specialization": spec
    }
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\n" + "=" * 55)
    print(f"  Experiment complete! Results in: {args.output_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
