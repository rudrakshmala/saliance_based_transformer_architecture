# run_phase3.py
# ----------------
# CLI entry point for Phase 3 full experiment.
# Mirrors ``run_phase2.py`` but delegates to the Phase‑3 experiment runner.

import argparse
import os
import sys

# Ensure the repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fstb.experiments.experiment_runner_phase3 import main as phase3_main


def main():
    parser = argparse.ArgumentParser(description="Run FSTB Phase 3 full experiment pipeline")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds to run")
    parser.add_argument("--size", type=str, default="tiny", choices=["tiny", "small", "medium", "large", "xl"], help="Model size class")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"], help="Execution mode – quick for fast sanity checks, full for full training")
    parser.add_argument("--output-dir", type=str, default="./results_phase3", help="Directory to store all results and artefacts")
    args = parser.parse_args()

    # The Phase‑3 runner parses its own arguments, but we expose a thin wrapper
    # for convenience.  We forward the arguments via environment variables to
    # keep the runner's signature unchanged.
    os.environ["FSTB_PHASE3_SEEDS"] = str(args.seeds)
    os.environ["FSTB_PHASE3_SIZE"] = args.size
    os.environ["FSTB_PHASE3_MODE"] = args.mode
    os.environ["FSTB_PHASE3_OUTPUT_DIR"] = args.output_dir

    # Invoke the main function from the experiment module.
    phase3_main()


if __name__ == "__main__":
    main()
