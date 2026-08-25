"""
run_phase2.py
=============
Primary CLI entry point for Phase 2 FSTB Controlled Experiments.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fstb.experiments.experiment_runner import Phase2ControlledExperiment


def main():
    parser = argparse.ArgumentParser(description="Run FSTB Phase 2 Controlled Research Experiment")
    parser.add_argument("--size", type=str, default="tiny", choices=["tiny", "small", "medium", "large", "xl"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced samples for fast execution")
    parser.add_argument("--output-dir", type=str, default="./results_phase2")
    args = parser.parse_args()

    exp = Phase2ControlledExperiment(
        size=args.size,
        num_seeds=args.seeds,
        quick_mode=args.quick,
        output_dir=args.output_dir
    )
    exp.run()


if __name__ == "__main__":
    main()
