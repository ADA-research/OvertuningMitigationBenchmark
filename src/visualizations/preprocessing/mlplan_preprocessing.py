"""MLPlan final-result preprocessing.

For each MLPlan run directory (identified by the presence of a ``phase2/``
subfolder), this module selects the single best configuration according to
``mlplan_score``, computes the cv_ensembled_test_score by ensembling the
10 phase-2 fold predictions, and writes a single-row "trajectory" file.

The output is schema-compatible with all other trajectory files so it can be
loaded by the visualization pipeline for the critical difference diagram.
MLPlan has no multi-point trajectory, so all overtuning/regret fields are NaN.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.visualizations.preprocessing.preprocessing import (
    convert_negative_metric_to_loss,
    extract_metadata_from_config,
    get_ensembled_test_scores_per_iteration,
)


# Must match REQUIRED_TRAJECTORY_COLUMNS in post_hoc_surrogate_preprocessing.py
# so that append_csv_file schema alignment works correctly.
REQUIRED_TRAJECTORY_COLUMNS = [
    "iteration",
    "experiment_id",
    "experiment_source",
    "raw_experiment_id",
    "dataset_id",
    "model",
    "optimizer",
    "metric",
    "problem_type",
    "mitigation",
    "reshuffling",
    "inner_split",
    "selection_set_size",
    "repetition",
    "outer_fold",
    "random_state",
    "outer_n_folds",
    "outer_n_repeats",
    "val_performance",
    "retrain_test_loss",
    "ensembled_test_loss",
    "retrain_meta_overfitting",
    "ensemble_meta_overfitting",
    "retrain_regret",
    "ensemble_regret",
    "retrain_overtuning",
    "ensemble_overtuning",
    "relative_retrain_overtuning",
    "relative_ensemble_overtuning",
]


def _is_mlplan_run_dir(run_dir: Path) -> bool:
    """Return True if the run directory contains a phase2 artifact subfolder."""
    return (run_dir / "phase2").is_dir() or (run_dir / "Phase2").is_dir()


def _resolve_phase2_artifact_path(run_dir: Path) -> Path:
    for candidate in [
        run_dir / "phase2" / "artifacts.npz",
        run_dir / "Phase2" / "artifacts.npz",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No phase2/artifacts.npz found in {run_dir}")


def _process_mlplan_run(task_dir: Path, experiment_name: str) -> dict:
    """Process one MLPlan run directory and return a result dict.

    This is a top-level function so joblib can pickle it for parallel execution.
    """
    history_path = task_dir / "history.csv"
    config_path = task_dir / "task_config.yaml"

    if not history_path.exists() or not config_path.exists():
        return {
            "status": "SKIPPED",
            "task_dir": str(task_dir),
            "reason": "Missing history.csv or task_config.yaml",
        }

    if not _is_mlplan_run_dir(task_dir):
        return {
            "status": "SKIPPED",
            "task_dir": str(task_dir),
            "reason": "Not an MLPlan run (no phase2 directory)",
        }

    try:
        phase2_artifact_path = _resolve_phase2_artifact_path(task_dir)
    except FileNotFoundError as exc:
        return {"status": "FAILED", "task_dir": str(task_dir), "error": str(exc)}

    try:
        history_df = pd.read_csv(history_path)
    except Exception as exc:
        return {
            "status": "FAILED",
            "task_dir": str(task_dir),
            "error": f"Failed to read history.csv: {exc}",
        }

    if "mlplan_score" not in history_df.columns:
        return {
            "status": "SKIPPED",
            "task_dir": str(task_dir),
            "reason": "No mlplan_score column in history.csv",
        }

    phase2_df = history_df.dropna(subset=["mlplan_score"]).copy()
    if phase2_df.empty:
        return {
            "status": "SKIPPED",
            "task_dir": str(task_dir),
            "reason": "No phase 2 rows (all mlplan_score values are NaN)",
        }

    # Select the globally best row from the original phase-2 history,
    # then use its iteration as the selected ML-Plan incumbent iteration.
    # This guarantees we truly pick the minimum mlplan_score observed.
    best_row_idx = int(phase2_df["mlplan_score"].astype(float).idxmin())
    best_row = phase2_df.loc[best_row_idx]
    best_iteration = int(best_row["iteration"])

    best_iter_df = phase2_df[phase2_df["iteration"] == best_iteration].copy()
    if best_iter_df.empty:
        return {
            "status": "FAILED",
            "task_dir": str(task_dir),
            "error": f"No rows found for selected iteration {best_iteration}",
        }

    # Use the selected iteration's row values for reported test/validation losses.
    avg_val_score = float(best_row["avg_val_score"])
    retrained_test_score = float(best_row["avg_test_score"])

    try:
        with open(config_path, "r") as fh:
            task_config = yaml.safe_load(fh)
    except Exception as exc:
        return {
            "status": "FAILED",
            "task_dir": str(task_dir),
            "error": f"Failed to read task_config.yaml: {exc}",
        }

    problem_type = task_config.get("problem_type", "binary")
    metric_name = task_config.get("metric", "roc_auc")

    # Ensemble the phase-2 fold predictions for the selected iteration.
    try:
        with np.load(phase2_artifact_path) as artifact:
            cv_ensemble_scores = get_ensembled_test_scores_per_iteration(
                df=best_iter_df,
                artifacts=artifact,
                problem_type=problem_type,
                metric_name=metric_name,
            )
    except Exception as exc:
        return {
            "status": "FAILED",
            "task_dir": str(task_dir),
            "error": f"Failed to compute ensemble score: {exc}",
        }

    if best_iteration not in cv_ensemble_scores:
        return {
            "status": "FAILED",
            "task_dir": str(task_dir),
            "error": f"Ensemble score missing for iteration {best_iteration}",
        }

    cv_ensembled_test_score = float(cv_ensemble_scores[best_iteration])

    return {
        "status": "SUCCESS",
        "task_dir": str(task_dir),
        "best_iteration": best_iteration,
        "avg_val_score": avg_val_score,
        "retrained_test_score": retrained_test_score,
        "cv_ensembled_test_score": cv_ensembled_test_score,
        "task_config": task_config,
    }


def preprocess_mlplan_experiment(
    results_dir: str,
    n_jobs: int = 1,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """Process an MLPlan experiment directory and write ``trajectories_mlplan.csv``.

    Scans all run subdirectories of ``results_dir`` for MLPlan runs (those
    with a ``phase2/`` subfolder).  For each valid run:

    1. Reads ``history.csv`` and filters to phase-2 rows
       (where ``mlplan_score`` is not NaN).
    2. Selects the iteration with the minimum ``mlplan_score``.
    3. Computes the cv_ensembled_test_score by averaging the phase-2 fold
       predictions for the selected iteration.
    4. Applies metric-sign conversion (e.g. ROC-AUC negation → loss scale).
    5. Writes a single-row trajectory entry (``iteration=0``) with all
       overtuning / regret fields set to NaN.

    The resulting ``trajectories_mlplan.csv`` is written at the experiment-dir
    level (``results_dir/trajectories_mlplan.csv``) and is compatible with
    ``preprocess_multiple_experiments`` for aggregation and normalization.

    Args:
        results_dir: Path to the experiment directory containing run subdirectories.
        n_jobs: Number of parallel workers for joblib.
        verbose: Whether to print progress information.

    Returns:
        DataFrame of trajectory rows, or None if no valid MLPlan runs were found.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory does not exist: {results_dir}")
        return None

    task_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    if not task_dirs:
        print(f"No task directories found in {results_dir}")
        return None

    if verbose:
        print(f"Found {len(task_dirs)} task directories; scanning for MLPlan runs")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_mlplan_run)(
            task_dir=task_dir,
            experiment_name=results_path.name,
        )
        for task_dir in tqdm(task_dirs, disable=not verbose, desc="Scanning MLPlan runs")
    )

    success_count = 0
    skip_count = 0
    fail_count = 0
    raw_rows: list[dict] = []

    for result in results:
        if result["status"] == "SKIPPED":
            skip_count += 1
            continue
        if result["status"] == "FAILED":
            fail_count += 1
            if verbose:
                print(f"Failed {result['task_dir']}: {result['error']}")
            continue

        success_count += 1
        task_dir = Path(result["task_dir"])
        metadata = extract_metadata_from_config(result["task_config"])
        metadata["experiment_source"] = results_path.name
        metadata["raw_experiment_id"] = task_dir.name
        metadata["experiment_id"] = f"{results_path.name}::{task_dir.name}"

        row = {
            "avg_val_score": result["avg_val_score"],
            "retrained_test_score": result["retrained_test_score"],
            "cv_ensembled_test_score": result["cv_ensembled_test_score"],
        }
        row.update(metadata)
        raw_rows.append(row)

    if verbose:
        print(
            f"\nMLPlan preprocessing results: "
            f"{success_count} success, {skip_count} skipped, {fail_count} failed"
        )

    if not raw_rows:
        print("No trajectory rows generated for MLPlan")
        return None

    results_df = pd.DataFrame(raw_rows)
    results_df = convert_negative_metric_to_loss(results_df)

    # Build single-row trajectory entries.  All overtuning/regret quantities
    # are undefined for MLPlan since there is no multi-point trajectory.
    trajectory_rows: list[dict] = []
    for _, run_row in results_df.iterrows():
        row = {
            "iteration": 0,
            "experiment_id": run_row.get("experiment_id"),
            "experiment_source": run_row.get("experiment_source"),
            "raw_experiment_id": run_row.get("raw_experiment_id"),
            "dataset_id": run_row.get("dataset_id"),
            "model": run_row.get("model"),
            "optimizer": run_row.get("optimizer"),
            "metric": run_row.get("metric"),
            "problem_type": run_row.get("problem_type"),
            "mitigation": run_row.get("mitigation"),
            "reshuffling": run_row.get("reshuffling"),
            "inner_split": run_row.get("inner_split"),
            "selection_set_size": run_row.get("selection_set_size"),
            "repetition": run_row.get("repetition"),
            "outer_fold": run_row.get("outer_fold"),
            "random_state": run_row.get("random_state"),
            "outer_n_folds": run_row.get("outer_n_folds"),
            "outer_n_repeats": run_row.get("outer_n_repeats"),
            "val_performance": run_row["avg_val_score"],
            "retrain_test_loss": run_row["retrained_test_score"],
            "ensembled_test_loss": run_row["cv_ensembled_test_score"],
            # Overtuning, regret, and meta-overfitting are undefined for MLPlan.
            "retrain_meta_overfitting": float("nan"),
            "ensemble_meta_overfitting": float("nan"),
            "retrain_regret": float("nan"),
            "ensemble_regret": float("nan"),
            "retrain_overtuning": float("nan"),
            "ensemble_overtuning": float("nan"),
            "relative_retrain_overtuning": float("nan"),
            "relative_ensemble_overtuning": float("nan"),
        }
        trajectory_rows.append(row)

    trajectories_df = pd.DataFrame(trajectory_rows)
    trajectories_df = trajectories_df[REQUIRED_TRAJECTORY_COLUMNS]

    trajectories_output_path = results_path / "trajectories_mlplan.csv"
    trajectories_df.to_csv(trajectories_output_path, index=False)

    if verbose:
        print(f"Saved MLPlan trajectories to {trajectories_output_path}")
        print(f"Shape: {trajectories_df.shape}")

    return trajectories_df


if __name__ == "__main__":
    import argparse
    import os

    preprocess_mlplan_experiment(
        results_dir="results_prd/FastAIMLP_363621_20260505_134259",
        verbose=True,
    )