# scripts/run_pilot.py
"""Run pilot experiment for all models and seeds.

Usage:
    python -m scripts.run_pilot --output_dir ../pilot_results
"""
import os
import argparse
import random
import numpy as np
import torch

# Add project root to PYTHONPATH
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fstb.models.model_factory import build_all_models
from fstb.training.trainer import FSTBTrainer, TrainingConfig
from fstb.data.pilot_dataset import get_pilot_datasets

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Pilot experiment runner")
    parser.add_argument('--output_dir', type=str, default='pilot_results', help='Root directory for checkpoints and logs')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 2026], help='Random seeds')
    parser.add_argument('--model_types', nargs='+', type=str,
                        default=['baseline', 'aux_baseline', 'fstb', 'rag', 'vector_memory', 'summarization'],
                        help='Models to train')
    parser.add_argument('--max_steps', type=int, default=2000, help='Maximum training steps per model')
    parser.add_argument('--token_budget', type=int, default=500_000, help='Token budget for pilot')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets (PersonaChat + MultiWOZ) – returns train/val splits
    train_dataset, val_dataset = get_pilot_datasets()

    for model_type in args.model_types:
        for seed in args.seeds:
            print(f"\n=== Training {model_type} | seed {seed} ===")
            set_global_seed(seed)
            models = build_all_models('small')  # ~30M params configuration
            model = models[model_type]

            # Build a training config – mirror default but override steps & budget
            config = TrainingConfig(
                warmup_steps=100,
                total_steps=args.max_steps,
                batch_size=8,
                grad_accumulation_steps=1,
                eval_every=100,
                save_every=200,
                early_stopping_patience=5,
                target_tokens=args.token_budget,
                seed=seed,
                checkpoint_dir=os.path.join(args.output_dir, model_type, f"seed_{seed}"),
                use_amp=torch.cuda.is_available()
            )

            trainer = FSTBTrainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, config=config,
                                 model_name=f"{model_type}_seed{seed}")
            history = trainer.train()
            # Save training history
            hist_path = os.path.join(config.checkpoint_dir, 'training_history.json')
            with open(hist_path, 'w') as f:
                import json
                json.dump(history, f, indent=2)
            print(f"Training completed for {model_type} seed {seed}. Checkpoints in {config.checkpoint_dir}")

if __name__ == '__main__':
    main()
