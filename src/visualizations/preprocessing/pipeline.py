import argparse
from pathlib import Path

from src.visualizations.preprocessing.preprocessing import preprocess_multiple_experiments
from src.visualizations.preprocessing.shared_utils import (
    RESULTS_FILE,
    TRAJECTORIES_FILE,
    get_experiment_dataset_id,
    is_small_dataset,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def include_experiment_dir(experiment_dir_name: str) -> bool:
    dataset_id = get_experiment_dataset_id(experiment_dir_name)

    if experiment_dir_name.startswith("RealMLP") and dataset_id is not None:
        return is_small_dataset(dataset_id)

    return True


def discover_experiment_dirs(results_dir: Path) -> list[Path]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Full results directory not found: {results_dir}")

    experiment_dirs = [
        path
        for path in sorted(results_dir.iterdir())
        if path.is_dir() and include_experiment_dir(path.name)
    ]
    if not experiment_dirs:
        raise ValueError(f"No experiment directories found in: {results_dir}")
    return experiment_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run regular preprocessing and combine experiment outputs."
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        help=(
            "Recalculate per-experiment preprocessing outputs from raw history.csv and artifacts files "
            "even if preprocessed files already exist."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Parent directory containing full experiment result directories (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/visualizations/data"),
        help="Directory for the combined gzip CSV outputs (default: src/visualizations/data)",
    )
    args = parser.parse_args()

    try:
        experiment_dirs = discover_experiment_dirs(args.results_dir)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    preprocess_multiple_experiments(
        experiment_dirs=[str(path) for path in experiment_dirs],
        output_path=str(args.output_dir),
        recalculate=args.recalculate,
    )

    print(
        f"Saved combined outputs to {args.output_dir / RESULTS_FILE} "
        f"and {args.output_dir / TRAJECTORIES_FILE}"
    )