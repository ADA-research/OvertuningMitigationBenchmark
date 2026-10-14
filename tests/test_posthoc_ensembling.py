from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.visualizations.preprocessing.post_hoc_ensemble_preprocessing import (
    caruana_greedy_weights,
    compute_post_hoc_ensemble_trajectories_for_run,
    get_test_preds_per_iteration,
    get_val_preds_per_iteration,
    preprocess_post_hoc_ensemble_experiment,
)
from src.metrics.metric import Metric


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


def _write_task_config(task_dir: Path, run_name: str, benchmark: bool = True):
    mitigation_strategy = "none" if benchmark else "thresholdout"
    selection_size = 0.0 if benchmark else None

    cfg = {
        "dataset_id": 999001,
        "problem_type": "regression",
        "optimizer": "smac",
        "metric": "neg_root_mean_squared_error",
        "random_state": 78,
        "mitigation_strategy": mitigation_strategy,
        "racing_strategy": "none",
        "result_path": run_name,
        "search_space": {
            "classifiers": ["LGBM"],
            "regressors": ["LGBM"],
        },
        "outer_evaluation": {
            "repeat": 0,
            "fold": 0,
            "n_folds": 3,
            "n_repeats": 10,
        },
        "evaluation": {
            "reshuffle": False,
            "resampling": "cv",
            "selection_size": selection_size,
        },
    }

    with open(task_dir / "task_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def _write_history(task_dir: Path):
    # Two iterations, two folds.
    history_df = pd.DataFrame(
        {
            "iteration": [0, 0, 1, 1],
            "fold": [0, 1, 0, 1],
        }
    )
    history_df.to_csv(task_dir / "history.csv", index=False)


def _write_artifacts(task_dir: Path, include_retrain: bool = True):
    # Validation labels per fold => OOF concat labels: [0, 1, 2, 3]
    val_labels_f0 = np.array([0.0, 1.0])
    val_labels_f1 = np.array([2.0, 3.0])

    # Iteration 0 validation preds are worse (RMSE=0.5)
    # Iteration 1 validation preds are perfect (RMSE=0.0)
    val_preds = {
        (0, 0): np.array([0.5, 1.5]),
        (0, 1): np.array([2.5, 3.5]),
        (1, 0): np.array([0.0, 1.0]),
        (1, 1): np.array([2.0, 3.0]),
    }

    test_labels = np.array([10.0, 20.0])

    # CV-ensemble test predictions from fold averages:
    # iter0 => [12, 22] (RMSE=2), iter1 => [11, 21] (RMSE=1)
    test_preds = {
        (0, 0): np.array([12.0, 22.0]),
        (0, 1): np.array([12.0, 22.0]),
        (1, 0): np.array([11.0, 21.0]),
        (1, 1): np.array([11.0, 21.0]),
    }

    arrays = {"test_labels": test_labels}

    for iteration in [0, 1]:
        for fold_id in [0, 1]:
            arrays[f"iter_{iteration}/fold_{fold_id}/val_preds"] = val_preds[(iteration, fold_id)]
            arrays[f"iter_{iteration}/fold_{fold_id}/val_labels"] = val_labels_f0 if fold_id == 0 else val_labels_f1
            arrays[f"iter_{iteration}/fold_{fold_id}/test_preds"] = test_preds[(iteration, fold_id)]

    if include_retrain:
        # Retrain test predictions:
        # iter0 => [13, 23] (RMSE=3), iter1 => [10, 20] (RMSE=0)
        arrays["iter_0/retrain_test_preds"] = np.array([13.0, 23.0])
        arrays["iter_1/retrain_test_preds"] = np.array([10.0, 20.0])

    np.savez_compressed(task_dir / "artifacts.npz", **arrays)


def _create_run(experiment_dir: Path, run_name: str, benchmark: bool = True, include_retrain: bool = True):
    task_dir = experiment_dir / run_name
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_task_config(task_dir, run_name, benchmark=benchmark)
    _write_history(task_dir)
    _write_artifacts(task_dir, include_retrain=include_retrain)
    return task_dir


def test_posthoc_end_to_end_full_schema_and_two_test_scores(tmp_path):
    experiment_dir = tmp_path / "exp_posthoc"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    _create_run(experiment_dir, "run_benchmark", benchmark=True, include_retrain=True)

    trajectories_df = preprocess_post_hoc_ensemble_experiment(
        results_dir=str(experiment_dir),
        ensemble_size=2,
        n_jobs=1,
        verbose=False,
    )

    assert trajectories_df is not None
    assert list(trajectories_df.columns) == REQUIRED_TRAJECTORY_COLUMNS
    assert len(trajectories_df) == 2

    rows = trajectories_df.sort_values("iteration").reset_index(drop=True)

    # Iteration 0: only candidate 0 exists.
    assert rows.loc[0, "val_performance"] == 0.5
    assert rows.loc[0, "ensembled_test_loss"] == 2.0
    assert rows.loc[0, "retrain_test_loss"] == 3.0

    # Iteration 1: candidate 1 dominates on validation, so weights focus on iter1.
    assert rows.loc[1, "val_performance"] == 0.0
    assert rows.loc[1, "ensembled_test_loss"] == 1.0
    assert rows.loc[1, "retrain_test_loss"] == 0.0

    # Ensure both test paths are represented and can differ.
    assert (rows["ensembled_test_loss"] != rows["retrain_test_loss"]).any()

    # Verify pipeline output files are written.
    assert (experiment_dir / "results_post_hoc_ensemble.csv").exists()
    assert (experiment_dir / "trajectories_post_hoc_ensemble.csv").exists()


def test_posthoc_skips_non_benchmark_runs(tmp_path):
    experiment_dir = tmp_path / "exp_mixed"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    _create_run(experiment_dir, "run_benchmark", benchmark=True, include_retrain=True)
    _create_run(experiment_dir, "run_thresholdout", benchmark=False, include_retrain=True)

    trajectories_df = preprocess_post_hoc_ensemble_experiment(
        results_dir=str(experiment_dir),
        ensemble_size=2,
        n_jobs=1,
        verbose=False,
    )

    assert trajectories_df is not None
    # Only the benchmark run should remain.
    assert trajectories_df["raw_experiment_id"].nunique() == 1
    assert trajectories_df["raw_experiment_id"].iloc[0] == "run_benchmark"


def test_posthoc_skips_when_retrain_test_preds_missing(tmp_path):
    experiment_dir = tmp_path / "exp_missing_retrain"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _create_run(experiment_dir, "run_missing_retrain", benchmark=True, include_retrain=False)

    result = compute_post_hoc_ensemble_trajectories_for_run(
        task_dir=run_dir,
        problem_type="regression",
        ensemble_size=2,
        experiment_name=experiment_dir.name,
    )

    assert result["status"] == "SKIPPED"
    assert "No valid iterations" in result["reason"]


def test_posthoc_non_finite_values_are_sanitized(tmp_path):
    experiment_dir = tmp_path / "exp_nonfinite"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _create_run(experiment_dir, "run_nonfinite", benchmark=True, include_retrain=True)

    artifact_path = run_dir / "artifacts.npz"
    with np.load(artifact_path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    # Inject non-finite values into a few arrays to trigger fallback sanitization.
    arrays["iter_0/fold_0/val_preds"] = np.array([np.nan, np.inf])
    arrays["iter_0/fold_0/test_preds"] = np.array([10.0, -np.inf])
    arrays["iter_0/retrain_test_preds"] = np.array([np.nan, 20.0])

    np.savez_compressed(artifact_path, **arrays)

    result = compute_post_hoc_ensemble_trajectories_for_run(
        task_dir=run_dir,
        problem_type="regression",
        ensemble_size=2,
        experiment_name=experiment_dir.name,
    )

    assert result["status"] == "SUCCESS"
    assert result["run_stats"]["replaced_non_finite_values"] > 0
    assert len(result["iterations"]) > 0


def test_caruana_ensemble_uses_validation_predictions_and_averages_predictions():
    labels = np.array([0.0, 2.0, 4.0])
    candidate_a = labels + np.array([-1.0, -1.0, -1.0])
    candidate_b = labels + np.array([1.0, 1.0, 1.0])
    metric = Metric("neg_root_mean_squared_error", "regression")

    weights, ensemble_predictions = caruana_greedy_weights(
        candidate_val_preds=[candidate_a, candidate_b],
        val_labels=labels,
        metric=metric,
        ensemble_size=2,
        problem_type="regression",
    )

    assert np.allclose(weights, np.array([0.5, 0.5]))
    assert np.allclose(ensemble_predictions, labels)
    assert metric.score(labels, ensemble_predictions) == 0.0


@pytest.mark.parametrize(
    ("problem_type", "metric_name", "labels", "candidate_predictions"),
    [
        (
            "binary",
            "roc_auc",
            np.array([0, 1, 1, 0]),
            [
                np.array([[0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3]]),
                np.array([[0.6, 0.4], [0.4, 0.6], [0.4, 0.6], [0.6, 0.4]]),
            ],
        ),
        (
            "multiclass",
            "neg_log_loss",
            np.array([0, 1, 2]),
            [
                np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]),
                np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]),
            ],
        ),
    ],
    ids=["binary", "multiclass"],
)
def test_caruana_ensemble_preserves_probability_shape(
    problem_type, metric_name, labels, candidate_predictions
):
    weights, ensemble_predictions = caruana_greedy_weights(
        candidate_val_preds=candidate_predictions,
        val_labels=labels,
        metric=Metric(metric_name, problem_type),
        ensemble_size=4,
        problem_type=problem_type,
    )

    assert np.isclose(weights.sum(), 1.0)
    assert ensemble_predictions.shape == candidate_predictions[0].shape
    assert np.allclose(ensemble_predictions.sum(axis=1), 1.0)


def test_posthoc_prediction_loading_uses_oof_labels_and_averages_folds(tmp_path):
    experiment_dir = tmp_path / "exp_predictions"
    experiment_dir.mkdir()
    task_dir = _create_run(experiment_dir, "run_predictions")
    history_df = pd.read_csv(task_dir / "history.csv")

    with np.load(task_dir / "artifacts.npz") as artifacts:
        val_predictions, val_labels, skipped_val = get_val_preds_per_iteration(
            df=history_df,
            artifacts=artifacts,
            problem_type="regression",
        )
        test_predictions, test_labels, skipped_test = get_test_preds_per_iteration(
            df=history_df,
            artifacts=artifacts,
            problem_type="regression",
        )

    assert not skipped_val
    assert not skipped_test
    assert np.array_equal(val_labels[0], np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.allclose(val_predictions[0], np.array([0.5, 1.5, 2.5, 3.5]))
    assert np.array_equal(test_labels, np.array([10.0, 20.0]))
    assert np.allclose(test_predictions[0], np.array([12.0, 22.0]))
