"""
CLI entry point for post-hoc surrogate trajectory computation.

Usage:
    python -m src.experiments.benchmark_experiments.post_hoc_surrogate_experiment \
        --results_dir /path/to/results \
        --n_jobs 16
"""

import argparse
from pathlib import Path

from src.visualizations.preprocessing.post_hoc_surrogate_preprocessing import (
    preprocess_post_hoc_surrogate_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute post-hoc surrogate trajectories for benchmark experiments"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to experiment results directory containing task folders",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress information",
    )

    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    trajectories_df = preprocess_post_hoc_surrogate_experiment(
        results_dir=args.results_dir,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )

    if trajectories_df is None:
        print("Failed to compute trajectories")
        return 1

    print("\nSuccessfully computed post-hoc surrogate trajectories")
    print(f"Output: {results_path / 'trajectories_post_hoc_surrogate.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
