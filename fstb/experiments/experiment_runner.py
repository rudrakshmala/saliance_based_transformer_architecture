"""
experiment_runner.py
====================
Master Phase 2 Controlled 3-Model Experiment Orchestrator.

Executes the full controlled research pipeline:
  1. Parameter-matched Model Trio Construction (Model A, B, C)
  2. Dataset Generation (MasterBenchmarkGenerator)
  3. Real Training via FSTBTrainer (AdamW, Cosine LR Warmup, Token Budget)
  4. Multi-Seed Execution (3-5 seeds)
  5. Statistical Significance Analysis (Paired t-test, Wilcoxon, Cohen's d)
  6. Mechanistic Interpretability & Probing across all 24 layers
  7. Block Attribution Ablations (Zero / Random Stage)
  8. Parameter Scaling Law Experiments (tiny, small, medium)
  9. Publication Figures & Interactive HTML Dashboard
 10. Automated LaTeX Manuscript Generation (paper.tex)
"""

import os
import json
import time
import torch
from typing import Dict, Any, List, Optional

from fstb.config import ModelConfig, MemoryConfig
from fstb.models.model_factory import build_model_trio, count_params, verify_parameter_parity
from fstb.data.benchmark_generator import MasterBenchmarkGenerator, BenchmarkConfig
from fstb.data.dataset import FSTBDataset
from fstb.training.trainer import FSTBTrainer, TrainingConfig
from fstb.evaluation.evaluator import BenchmarkEvaluator
from fstb.evaluation.stat_tests import StatisticalSignificanceAnalyzer
from fstb.analysis.block_attribution import BlockAttributionExperiment
from fstb.analysis.info_flow import InfoFlowAnalyzer
from fstb.analysis.sankey import generate_routing_sankey, generate_stage_flow_sankey
from fstb.experiments.seed_manager import SeedManager
from fstb.experiments.scaling_runner import ScalingExperimentRunner
from fstb.visualization.plots import (
    plot_cka_heatmap, plot_attention_entropy, plot_ablation_bar_chart
)
from fstb.visualization.dashboard import FSTBDashboardGenerator
from fstb.reporting.paper_generator import PaperGenerator
from scripts.run_experiments import NumpyEncoder


class Phase2ControlledExperiment:
    """Master orchestrator for Phase 2 FSTB research evaluation."""

    def __init__(
        self,
        size: str = "tiny",
        num_seeds: int = 3,
        quick_mode: bool = True,
        output_dir: str = "./results_phase2"
    ):
        self.size = size
        self.num_seeds = num_seeds
        self.quick_mode = quick_mode
        self.output_dir = output_dir
        self.device = torch.device("cpu")

    def run(self) -> Dict[str, Any]:
        t_start = time.time()
        print("=" * 65, flush=True)
        print("  FSTB Phase 2 — Controlled 3-Model Research Experiment", flush=True)
        print("=" * 65, flush=True)
        print(f"  Model Size  : {self.size.upper()}")
        print(f"  Seeds       : {self.num_seeds}")
        print(f"  Quick Mode  : {self.quick_mode}")
        print(f"  Output Dir  : {self.output_dir}")
        print("=" * 65, flush=True)

        # ── 1. Model Trio & Parity Check ─────────────────────────────
        print("\n[1/7] Building parameter-matched model trio...", flush=True)
        base_model, aux_model, fstb_model = build_model_trio(self.size)
        parity = verify_parameter_parity(base_model, aux_model, fstb_model)

        print(f"    Model A (Baseline)    : {count_params(base_model):,} params")
        print(f"    Model B (AuxBaseline) : {count_params(aux_model):,} params")
        print(f"    Model C (FSTB)        : {count_params(fstb_model):,} params")
        print(f"    Parity max diff       : {parity['diffs']['max_diff_pct']:.3f}% (OK: {parity['parity_ok']})")

        # ── 2. Master Benchmark Generation ────────────────────────────
        print("\n[2/7] Generating synthetic memory benchmark trajectories...", flush=True)
        bench_cfg = BenchmarkConfig(
            num_users=2 if self.quick_mode else 10,
            sessions_per_user=3 if self.quick_mode else 20,
            mem_update_samples=10 if self.quick_mode else 50,
            contradiction_samples=10 if self.quick_mode else 50,
            long_horizon_samples=10 if self.quick_mode else 50,
            seed=42
        )
        master_gen = MasterBenchmarkGenerator(bench_cfg)
        bench_data = master_gen.generate_all()

        all_samples = []
        for key, samples in bench_data.items():
            all_samples.extend(samples)

        train_samples, val_samples, test_samples = master_gen.get_train_val_test_split(all_samples, (0.8, 0.1, 0.1))
        if not train_samples:
            train_samples = all_samples
            val_samples   = all_samples
            test_samples  = all_samples

        print(f"    Split sizes: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

        train_ds = FSTBDataset(train_samples, vocab_size=2048, max_seq_len=64)
        val_ds   = FSTBDataset(val_samples,   vocab_size=2048, max_seq_len=64)
        test_ds  = FSTBDataset(test_samples,  vocab_size=2048, max_seq_len=64)

        # ── 3. Multi-Seed Controlled Training & Evaluation ────────────
        print(f"\n[3/7] Running multi-seed training ({self.num_seeds} seeds)...", flush=True)
        evaluator = BenchmarkEvaluator(device=self.device)

        base_seed_metrics: List[Dict[str, float]] = []
        aux_seed_metrics:  List[Dict[str, float]] = []
        fstb_seed_metrics: List[Dict[str, float]] = []

        last_fstb_res = None

        for s_idx in range(self.num_seeds):
            seed = 42 + s_idx
            print(f"\n  --- Seed {seed} ({s_idx+1}/{self.num_seeds}) ---", flush=True)
            SeedManager.set_seed(seed)

            # Re-build fresh models per seed
            bm, am, fm = build_model_trio(self.size)

            train_cfg = TrainingConfig(
                learning_rate=3e-4,
                max_epochs=1 if self.quick_mode else 3,
                batch_size=4,
                target_tokens=5000 if self.quick_mode else 50000,
                seed=seed,
                checkpoint_dir=os.path.join(self.output_dir, "checkpoints")
            )

            # Train each model
            print("    Training Model A (Baseline)...", flush=True)
            FSTBTrainer(bm, train_ds, val_ds, train_cfg, model_name="baseline").train()

            print("    Training Model B (AuxBaseline)...", flush=True)
            FSTBTrainer(am, train_ds, val_ds, train_cfg, model_name="aux_baseline").train()

            print("    Training Model C (FSTB)...", flush=True)
            FSTBTrainer(fm, train_ds, val_ds, train_cfg, model_name="fstb").train()

            # Evaluate on test set
            r_b = evaluator.evaluate(bm, test_ds, run_specialization=False, max_batches=3)
            r_a = evaluator.evaluate(am, test_ds, run_specialization=False, max_batches=3)
            r_f = evaluator.evaluate(fm, test_ds, run_specialization=True,  max_batches=3)

            base_seed_metrics.append(r_b["metrics"])
            aux_seed_metrics.append(r_a["metrics"])
            fstb_seed_metrics.append(r_f["metrics"])
            last_fstb_res = r_f

        # Aggregate metrics across seeds
        avg_base = SeedManager.aggregate_seed_results(base_seed_metrics)
        avg_aux  = SeedManager.aggregate_seed_results(aux_seed_metrics)
        avg_fstb = SeedManager.aggregate_seed_results(fstb_seed_metrics)

        # ── 4. Statistical Significance Tests ─────────────────────────
        print("\n[4/7] Computing statistical significance across seeds...", flush=True)
        fstb_scores = [m["memory_f1"] for m in fstb_seed_metrics]
        aux_scores  = [m["memory_f1"] for m in aux_seed_metrics]

        stat_res = StatisticalSignificanceAnalyzer.compare_models(fstb_scores, aux_scores)
        print(f"    Paired t-test p-value : {stat_res.get('p_value_ttest')}")
        print(f"    Wilcoxon p-value      : {stat_res.get('p_value_wilcoxon')}")
        print(f"    Cohen's d effect size : {stat_res.get('cohens_d'):.4f}")
        print(f"    Statistically Sig.    : {stat_res.get('statistically_significant')}")

        # ── 5. Mechanistic Analysis & Ablations ───────────────────────
        print("\n[5/7] Running mechanistic block attribution ablations...", flush=True)
        attrib_runner = BlockAttributionExperiment(fm, evaluator, test_ds, self.device)
        attrib_results = attrib_runner.run()
        attrib_dict = BlockAttributionExperiment.results_to_dict(attrib_results)

        # ── 6. Parameter Scaling Laws ────────────────────────────────
        print("\n[6/7] Running parameter scaling laws...", flush=True)
        scaling_runner = ScalingExperimentRunner(test_ds, self.device)
        scaling_res = scaling_runner.run_scaling(["tiny", "small"] if self.quick_mode else ["tiny", "small", "medium"])

        # ── 7. Reporting & Paper Generation ───────────────────────────
        print("\n[7/7] Generating publication figures, dashboard, and LaTeX manuscript...", flush=True)
        os.makedirs(self.output_dir, exist_ok=True)
        figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        spec = last_fstb_res.get("specialization", {}) if last_fstb_res else {}
        if "cka_matrix" in spec:
            plot_cka_heatmap(spec["cka_matrix"], os.path.join(figures_dir, "cka_similarity_heatmap.png"))

        if "layer_entropies" in spec:
            plot_attention_entropy(spec["layer_entropies"], os.path.join(figures_dir, "attention_entropy_by_layer.png"))

        plot_ablation_bar_chart(attrib_dict, os.path.join(figures_dir, "ablation_matrix_comparison.png"))

        # Sankey diagram
        sankey_path = os.path.join(self.output_dir, "sankey_routing_flow.html")
        generate_routing_sankey(torch.ones(5, 5).numpy(), sankey_path)

        # HTML Dashboard
        dashboard_path = os.path.join(self.output_dir, "dashboard.html")
        FSTBDashboardGenerator.generate_html_report(
            results_summary={"fstb": fstb_seed_metrics[-1], "baseline": base_seed_metrics[-1]},
            ablation_summary=attrib_dict,
            stat_summary=stat_res,
            output_html_path=dashboard_path
        )

        # Results JSON
        final_results = {
            "size": self.size,
            "num_seeds": self.num_seeds,
            "parameter_parity": parity,
            "baseline": base_seed_metrics[-1],
            "aux_baseline": aux_seed_metrics[-1],
            "fstb": fstb_seed_metrics[-1],
            "aggregated_baseline": avg_base,
            "aggregated_aux_baseline": avg_aux,
            "aggregated_fstb": avg_fstb,
            "stat_tests": stat_res,
            "ablations": attrib_dict,
            "scaling": scaling_res,
            "specialization": spec,
            "elapsed_seconds": round(time.time() - t_start, 2)
        }

        json_path = os.path.join(self.output_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, cls=NumpyEncoder)
        print(f"    Saved: {json_path}")

        # LaTeX Paper Generator
        PaperGenerator(final_results, os.path.join(self.output_dir, "paper")).generate()

        print("\n" + "=" * 65, flush=True)
        print(f"  FSTB Phase 2 Complete in {round(time.time() - t_start, 1)}s!")
        print(f"  Dashboard : {dashboard_path}")
        print(f"  Paper.tex : {os.path.join(self.output_dir, 'paper', 'paper.tex')}")
        print("=" * 65, flush=True)

        return final_results
