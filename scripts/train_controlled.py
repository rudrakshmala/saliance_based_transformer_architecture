"""
train_controlled.py
===================
CLI script for training controlled model trios (Baseline, AuxBaseline, FSTB).
"""

import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fstb.models.model_factory import build_model_trio, count_params, verify_parameter_parity
from fstb.data.benchmark_generator import MasterBenchmarkGenerator, BenchmarkConfig
from fstb.data.dataset import FSTBDataset
from fstb.training.trainer import FSTBTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train controlled 3-model trio")
    parser.add_argument("--size", type=str, default="tiny", choices=["tiny", "small", "medium", "large", "xl"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  FSTB Controlled Training — Size: {args.size.upper()} | Seed: {args.seed}")
    print("=" * 60)

    base, aux, fstb = build_model_trio(args.size)
    parity = verify_parameter_parity(base, aux, fstb)

    print(f"  Base params : {count_params(base):,}")
    print(f"  Aux params  : {count_params(aux):,}")
    print(f"  FSTB params : {count_params(fstb):,}")
    print(f"  Parity OK   : {parity['parity_ok']} ({parity['diffs']['max_diff_pct']:.3f}% max diff)")

    # Data
    bench_cfg = BenchmarkConfig(num_users=2, sessions_per_user=3, mem_update_samples=10, contradiction_samples=10)
    data = MasterBenchmarkGenerator(bench_cfg).generate_all()

    train_ds = FSTBDataset(data["train"], vocab_size=2048, max_seq_len=64)
    val_ds   = FSTBDataset(data["val"],   vocab_size=2048, max_seq_len=64)

    cfg = TrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_dir=args.output_dir
    )

    print("\nTraining Model A (Baseline)...")
    FSTBTrainer(base, train_ds, val_ds, cfg, model_name="baseline").train()

    print("\nTraining Model B (AuxBaseline)...")
    FSTBTrainer(aux, train_ds, val_ds, cfg, model_name="aux_baseline").train()

    print("\nTraining Model C (FSTB)...")
    FSTBTrainer(fstb, train_ds, val_ds, cfg, model_name="fstb").train()

    print("\nTraining Complete! Checkpoints saved in:", args.output_dir)


if __name__ == "__main__":
    main()
