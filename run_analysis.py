#!/usr/bin/env python3
"""
Run Analysis
============

Entry point for the mediation–moderation analysis.

Usage
-----
    # With real survey data in data/ directory:
    python run_analysis.py

    # With an explicit file path:
    python run_analysis.py --file path/to/data.xlsx

    # With synthetic sample data (for demonstration/testing):
    python run_analysis.py --sample
"""

import argparse
import sys

from analysis.data_loader import generate_sample_data, load_survey_data
from analysis.mediation_moderation_analysis import run_full_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Run mediation–moderation analysis on survey data."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to the survey data Excel file (.xlsx).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use synthetic sample data instead of a real file.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=300,
        help="Number of synthetic respondents (default: 300).",
    )
    args = parser.parse_args()

    if args.sample:
        print("Generating synthetic sample data …")
        df = generate_sample_data(n=args.sample_n)
    else:
        df = load_survey_data(filepath=args.file)

    results = run_full_analysis(df)
    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
