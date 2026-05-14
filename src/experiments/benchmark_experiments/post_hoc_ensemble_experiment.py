"""
CLI entry point for post-hoc ensemble trajectory computation.

Usage:
    python -m src.experiments.benchmark_experiments.post_hoc_ensemble_experiment \
        --results_dir /path/to/results \
    --ensemble_size 40 \
        --n_jobs 16
"""

import argparse
from pathlib import Path
from src.visualizations.preprocessing.post_hoc_ensemble_preprocessing import (
    preprocess_post_hoc_ensemble_experiment,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(
        description="Compute post-hoc Caruana ensemble trajectories for benchmark experiments"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to experiment results directory containing task folders"
    )
    
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=40,
        help="Number of greedy ensemble selection steps (default: 40)"
    )
    
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress information"
    )
    
    args = parser.parse_args()
    
    # Validate results directory
    results_path = Path(args.results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")
    
    # Run preprocessing
    trajectories_df = preprocess_post_hoc_ensemble_experiment(
        results_dir=args.results_dir,
        ensemble_size=args.ensemble_size,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
    
    if trajectories_df is not None:
        print(f"\nSuccessfully computed post-hoc ensemble trajectories")
        print(f"Output: {results_path / 'trajectories_post_hoc_ensemble.csv'}")
        return 0
    else:
        print("Failed to compute trajectories")
        return 1


if __name__ == "__main__":
    exit(main())
