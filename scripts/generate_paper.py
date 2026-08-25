"""
generate_paper.py
=================
CLI script for auto-generating arXiv paper LaTeX manuscript from results.json.
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fstb.reporting.paper_generator import PaperGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX paper from results.json")
    parser.add_argument("--results", type=str, default="./results_phase2/results.json")
    parser.add_argument("--output-dir", type=str, default="./results_phase2/paper")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)

    with open(args.results, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    paper_path = PaperGenerator(results_data, args.output_dir).generate()
    print(f"Paper manuscript successfully generated at: {paper_path}")


if __name__ == "__main__":
    main()
