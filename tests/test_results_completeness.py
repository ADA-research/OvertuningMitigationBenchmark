"""Validation tests for benchmark result completeness and integrity.

These tests check a results root (default: results_prd2/results) against:
- expected dataset x model experiment coverage from docs/datasets.md
- expected per-task folders derived from benchmark_lite naming logic
- required per-task artifacts (history.csv)
- experiment-level trajectories.csv presence and basic integrity

To run against a different results root, set environment variable:
    OVERTUNING_RESULTS_DIR=/path/to/results pytest tests/test_results_completeness.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_MD = PROJECT_ROOT / "docs" / "datasets.md"
MODELS = ("FastAIMLP", "LGBM", "RealMLP")


def _parse_datasets_markdown(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find datasets markdown file: {path}")

    result: dict[str, dict[str, object]] = {}
    column_indices: dict[str, int] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        normalized_headers = [part.lower().replace("#", "").strip() for part in parts]
        if "dataset id" in normalized_headers:
            column_indices = {
                "dataset_id": normalized_headers.index("dataset id"),
                "instances": normalized_headers.index("instances"),
                "type": normalized_headers.index("type"),
            }
            continue
        if column_indices is None or len(parts) <= max(column_indices.values()):
            continue

        dataset_id = parts[column_indices["dataset_id"]]
        instances_raw = parts[column_indices["instances"]].replace(",", "")
        dataset_type_raw = parts[column_indices["type"]].lower()

        if not dataset_id.isdigit():
            continue
        if not instances_raw.isdigit():
            continue

        instances = int(instances_raw)
        if "regression" in dataset_type_raw:
            problem_type = "regression"
        elif "multiclass" in dataset_type_raw:
            problem_type = "multiclass"
        else:
            problem_type = "binary"

        result[dataset_id] = {
            "problem_type": problem_type,
            "large_dataset": instances > 2500,
            "instances": instances,
        }

    if not result:
        raise ValueError("No datasets could be parsed from datasets.md")
    return result


DATASET_SPECS = _parse_datasets_markdown(DATASETS_MD)
DATASET_IDS = sorted(DATASET_SPECS.keys())
MODEL_DATASET_CASES = [
    (model, dataset_id)
    for model in MODELS
    for dataset_id in DATASET_IDS
    if model != "RealMLP" or not DATASET_SPECS[dataset_id]["large_dataset"]
]


def _metric_for_problem_type(problem_type: str) -> str:
    if problem_type == "binary":
        return "roc_auc"
    if problem_type == "multiclass":
        return "neg_log_loss"
    if problem_type == "regression":
        return "neg_root_mean_squared_error"
    raise ValueError(f"Unknown problem_type: {problem_type}")


def _expected_task_names(dataset_id: str, problem_type: str, large_dataset: bool) -> set[str]:
    metric = _metric_for_problem_type(problem_type)
    outer_n_repeats = 3 if large_dataset else 10
    expected: set[str] = set()

    for repeat_idx in range(outer_n_repeats):
        for optimizer in ("hebo", "smac"):
            # MLPlan endpoint.
            expected.add(
                f"{dataset_id}_rep{repeat_idx}_fold0_{optimizer}_{metric}_mlplan"
            )

            # Non-MLPlan endpoints.
            for reshuffling in (True, False):
                reshuf_prefix = "reshuf_" if reshuffling else ""

                # Racing endpoint.
                expected.add(
                    f"{dataset_id}_rep{repeat_idx}_fold0_{optimizer}_{metric}_{reshuf_prefix}bergman_aggressive"
                )

                # Thresholdout endpoint.
                expected.add(
                    f"{dataset_id}_rep{repeat_idx}_fold0_{optimizer}_{metric}_{reshuf_prefix}tho"
                )

                # Baseline endpoints with inner split x selection set.
                for inner_resampling in ("cv", "holdout"):
                    expected.add(
                        f"{dataset_id}_rep{repeat_idx}_fold0_{optimizer}_{metric}_{reshuf_prefix}{inner_resampling}"
                    )
                    expected.add(
                        f"{dataset_id}_rep{repeat_idx}_fold0_{optimizer}_{metric}_{reshuf_prefix}{inner_resampling}_sel"
                    )

    return expected


def _results_root() -> Path:
    configured = os.environ.get("OVERTUNING_RESULTS_DIR", "results_prd2/results")
    root = Path(configured)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


@pytest.fixture(scope="session")
def results_root() -> Path:
    root = _results_root()
    if not root.exists() and "OVERTUNING_RESULTS_DIR" not in os.environ:
        pytest.skip("Set OVERTUNING_RESULTS_DIR to run result-completeness checks")
    assert root.exists(), f"Results root does not exist: {root}"
    assert root.is_dir(), f"Results root is not a directory: {root}"
    return root


def _experiment_dirs_for_combo(results_root: Path, model: str, dataset_id: str) -> list[Path]:
    prefix = f"{model}_{dataset_id}_"
    return sorted(
        [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    )


def _validate_trajectories_csv(experiment_dir: Path) -> list[str]:
    issues: list[str] = []
    trajectories_path = experiment_dir / "trajectories.csv"

    if not trajectories_path.exists():
        return [f"Missing trajectories.csv in {experiment_dir.name}"]

    try:
        df = pd.read_csv(trajectories_path)
    except Exception as exc:
        return [f"Could not read trajectories.csv in {experiment_dir.name}: {exc}"]

    if df.empty:
        return [f"trajectories.csv is empty in {experiment_dir.name}"]

    required_cols = {
        "experiment_id",
        "raw_experiment_id",
        "iteration",
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
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        issues.append(
            f"trajectories.csv missing required columns in {experiment_dir.name}: {missing_cols}"
        )
        return issues

    if df["iteration"].isna().any():
        issues.append(f"Found NaN iteration values in trajectories.csv for {experiment_dir.name}")
    if (df["iteration"] < 0).any():
        issues.append(f"Found negative iteration values in trajectories.csv for {experiment_dir.name}")

    key_cols = [
        "experiment_id",
        "iteration",
    ]
    if df.duplicated(subset=key_cols).any():
        issues.append(
            f"Duplicate trajectory rows by key {key_cols} in {experiment_dir.name}"
        )

    return issues


def _validate_experiment_dir(
    experiment_dir: Path,
    dataset_id: str,
    problem_type: str,
    large_dataset: bool,
) -> list[str]:
    issues: list[str] = []
    expected_tasks = _expected_task_names(
        dataset_id=dataset_id,
        problem_type=problem_type,
        large_dataset=large_dataset,
    )

    task_dirs = sorted([p for p in experiment_dir.iterdir() if p.is_dir()])
    observed_task_names = {p.name for p in task_dirs}

    missing_task_dirs = sorted(expected_tasks - observed_task_names)
    if missing_task_dirs:
        issues.append(
            f"Missing {len(missing_task_dirs)} expected task folders in {experiment_dir.name}. "
            f"First missing examples: {missing_task_dirs[:10]}"
        )

    # Verify required task files for expected task dirs that are present.
    missing_history = []
    for task_name in sorted(expected_tasks.intersection(observed_task_names)):
        history_path = experiment_dir / task_name / "history.csv"
        if not history_path.exists():
            missing_history.append(task_name)

    if missing_history:
        issues.append(
            f"Missing history.csv in {len(missing_history)} task folders of {experiment_dir.name}. "
            f"First missing examples: {missing_history[:10]}"
        )

    issues.extend(_validate_trajectories_csv(experiment_dir))
    return issues


@pytest.mark.parametrize(
    "model,dataset_id",
    MODEL_DATASET_CASES,
    ids=[f"{model}-{dataset_id}" for model, dataset_id in MODEL_DATASET_CASES],
)
def test_results_completeness_per_model_dataset(
    results_root: Path,
    model: str,
    dataset_id: str,
) -> None:
    """For each model x dataset, verify expected experiments and per-experiment integrity."""
    spec = DATASET_SPECS[dataset_id]
    problem_type = str(spec["problem_type"])
    large_dataset = bool(spec["large_dataset"])

    experiment_dirs = _experiment_dirs_for_combo(results_root, model, dataset_id)
    assert experiment_dirs, (
        f"Missing experiment folder for {model} dataset {dataset_id}. "
        f"Expected at least one folder with prefix '{model}_{dataset_id}_' under {results_root}."
    )

    issues: list[str] = []
    for experiment_dir in experiment_dirs:
        issues.extend(
            _validate_experiment_dir(
                experiment_dir=experiment_dir,
                dataset_id=dataset_id,
                problem_type=problem_type,
                large_dataset=large_dataset,
            )
        )

    assert not issues, "\n".join(issues)


def test_no_unknown_model_dataset_experiment_folders(results_root: Path) -> None:
    """Guard against malformed model/dataset prefixes in experiment folder names."""
    known_dataset_ids = set(DATASET_IDS)
    issues: list[str] = []

    for item in sorted(results_root.iterdir()):
        if not item.is_dir():
            continue
        if not any(item.name.startswith(f"{model}_") for model in MODELS):
            continue

        parts = item.name.split("_")
        if len(parts) < 3:
            issues.append(f"Unexpected experiment folder format: {item.name}")
            continue

        model = parts[0]
        dataset_id = parts[1]
        if model not in MODELS:
            issues.append(f"Unknown model prefix in experiment folder: {item.name}")
        if dataset_id not in known_dataset_ids:
            issues.append(
                f"Unknown dataset id '{dataset_id}' in experiment folder: {item.name}"
            )

    assert not issues, "\n".join(issues)
