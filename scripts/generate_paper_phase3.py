# generate_paper_phase3.py
# ---------------------------------
# Thin wrapper that invokes the Phase‑3 paper generation.
# It mirrors ``generate_paper.py`` but points to the Phase‑3 results
# directory.  This script can be called directly from the command line.

import argparse
import os
import sys

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fstb.reporting.paper_generator import PaperGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 3 research paper from aggregated results")
    parser.add_argument("--results-dir", type=str, default="./results_phase3", help="Directory containing aggregated_results.json produced by run_phase3")
    args = parser.parse_args()

    agg_path = os.path.join(args.results_dir, "aggregated_results.json")
    if not os.path.exists(agg_path):
        raise FileNotFoundError(f"Aggregated results not found at {agg_path}")

    # Load results (simple json load – the PaperGenerator expects a dict)
    import json
    with open(agg_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    paper_gen = PaperGenerator(results_dict=results, output_dir=os.path.join(args.results_dir, "paper"))
    paper_path = paper_gen.generate()
    print(f"Paper generated at {paper_path}")


if __name__ == "__main__":
    main()
