import os
import argparse
from pathlib import Path

from src.visualizations.preprocessing.preprocessing import preprocess_multiple_experiments
from src.visualizations.preprocessing.shared_utils import (
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


def count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8") as handle:
        # Subtract header; clamp at 0 for empty files.
        return max(0, sum(1 for _ in handle) - 1)

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
    args = parser.parse_args()

    experiment_dirs = [
        f"results_prd/{experiment_dir_name}"
        for experiment_dir_name in sorted(os.listdir("results_prd"))
        if include_experiment_dir(experiment_dir_name)
    ]

    preprocess_multiple_experiments(
        experiment_dirs=experiment_dirs,
        output_path="src/visualizations/data",
        recalculate=args.recalculate,
        return_combined_dataframes=False,
    )

    output_dir = Path("src/visualizations/data")
    results_shape = (count_csv_rows(output_dir / "preprocessed_results.csv"), "?")
    trajectories_shape = (count_csv_rows(output_dir / "trajectories.csv"), "?")

    print(f"Results rows: {results_shape[0]}")
    print(f"Trajectories rows: {trajectories_shape[0]}")
    print(
        "Saved combined outputs to src/visualizations/data/preprocessed_results.csv, "
        "src/visualizations/data/trajectories.csv, "
        "src/visualizations/data/trajectories_surrogate_selection.csv, "
        "src/visualizations/data/trajectories_one_se_selection.csv, "
        "src/visualizations/data/trajectories_makarova.csv, "
        "src/visualizations/data/trajectories_post_hoc_ensemble.csv (if present), and "
        "src/visualizations/data/trajectories_post_hoc_surrogate.csv (if present)"
    )