"""
Tests for Metric class.

Critical: Metric.score() expects 2D probability arrays (shape [n, n_classes])
for binary and multiclass classification. Passing 1D class labels will fail.
This matches how Evaluator.evaluate() works (it always passes predict_proba output).

Stored predictions in fold.preds are 1D (positive class for binary) via series_to_numpy(),
but Metric.score() always operates on 2D input.
"""
from pathlib import Path

import numpy as np
import pytest
from pytest import approx
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from src.metrics.metric import Metric


DATASETS_DIR = Path(__file__).parent.parent / "src" / "datasets" / "datasets"


def binary_proba(y_true_labels, y_pred_pos_proba):
    """
    Build a 2D probability array for binary classification.
    y_pred_pos_proba: probability of positive class (class 1).
    """
    y_pred_pos = np.array(y_pred_pos_proba)
    return np.column_stack([1 - y_pred_pos, y_pred_pos])


def multiclass_proba(y_pred_classes, n_classes):
    """
    Build a 2D argmax-style probability array for multiclass (one-hot with 0.9/0.1).
    """
    n = len(y_pred_classes)
    proba = np.full((n, n_classes), 0.1 / (n_classes - 1))
    for i, c in enumerate(y_pred_classes):
        proba[i, c] = 0.9
    return proba


# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------

class TestBinaryAccuracy:
    def test_perfect_predictions(self):
        metric = Metric("accuracy", "binary")
        y_true = np.array([0, 1, 1, 0])
        y_pred = binary_proba(y_true, [0.1, 0.9, 0.8, 0.2])
        assert metric.score(y_true, y_pred) == approx(-1.0)

    def test_half_correct(self):
        metric = Metric("accuracy", "binary")
        y_true = np.array([1, 0, 0, 0])
        y_pred = binary_proba(y_true, [0.9, 0.9, 0.1, 0.1])
        # Two correct out of four → accuracy = 0.5
        assert metric.score(y_true, y_pred) == approx(-0.75)

    def test_score_is_negated(self):
        metric = Metric("accuracy", "binary")
        y_true = np.array([1, 0, 1, 0])
        y_pred = binary_proba(y_true, [0.9, 0.1, 0.9, 0.1])
        score = metric.score(y_true, y_pred)
        assert score <= 0.0

    def test_score_range(self):
        metric = Metric("accuracy", "binary")
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = binary_proba(y_true, [0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        score = metric.score(y_true, y_pred)
        assert -1.0 <= score <= 0.0

    def test_need_proba_is_false(self):
        metric = Metric("accuracy", "binary")
        assert metric.need_proba is False

    def test_threshold_at_0_5(self):
        metric = Metric("accuracy", "binary")
        y_true = np.array([1, 0])
        # Just above 0.5 → predicted 1; just below 0.5 → predicted 0
        y_pred = binary_proba(y_true, [0.51, 0.49])
        assert metric.score(y_true, y_pred) == approx(-1.0)


class TestBinaryRocAuc:
    def test_perfect_ranking(self):
        metric = Metric("roc_auc", "binary")
        y_true = np.array([0, 0, 1, 1])
        y_pred = binary_proba(y_true, [0.1, 0.2, 0.8, 0.9])
        assert metric.score(y_true, y_pred) == approx(-1.0)

    def test_worst_ranking(self):
        metric = Metric("roc_auc", "binary")
        y_true = np.array([0, 0, 1, 1])
        y_pred = binary_proba(y_true, [0.9, 0.8, 0.2, 0.1])
        assert metric.score(y_true, y_pred) == approx(-0.0)

    def test_value_matches_sklearn(self):
        metric = Metric("roc_auc", "binary")
        y_true = np.array([1, 0, 1, 0, 1, 1])
        pos_proba = np.array([0.9, 0.1, 0.8, 0.3, 0.7, 0.6])
        y_pred = binary_proba(y_true, pos_proba)

        expected = roc_auc_score(y_true, pos_proba)
        assert metric.score(y_true, y_pred) == approx(-expected)

    def test_need_proba_is_true(self):
        metric = Metric("roc_auc", "binary")
        assert metric.need_proba is True

    def test_score_is_negated(self):
        metric = Metric("roc_auc", "binary")
        y_true = np.array([1, 0, 1, 0])
        y_pred = binary_proba(y_true, [0.9, 0.1, 0.7, 0.3])
        assert metric.score(y_true, y_pred) <= 0.0


# ---------------------------------------------------------------------------
# Multiclass metrics
# ---------------------------------------------------------------------------



class TestMulticlassNegLogLoss:
    def test_better_probabilities_get_lower_objective(self):
        """
        For a minimize-only pipeline, better multiclass probabilities should
        produce a lower objective value for neg_log_loss.
        """
        metric = Metric("neg_log_loss", "multiclass")
        y_true = np.array([0, 1, 2, 1])

        # High confidence on the true class -> low log loss.
        y_pred_good = np.array([
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
            [0.10, 0.80, 0.10],
        ])

        # Low confidence / wrong emphasis -> higher log loss.
        y_pred_bad = np.array([
            [0.40, 0.30, 0.30],
            [0.35, 0.25, 0.40],
            [0.45, 0.35, 0.20],
            [0.60, 0.20, 0.20],
        ])

        score_good = metric.score(y_true, y_pred_good)
        score_bad = metric.score(y_true, y_pred_bad)

        assert score_good < score_bad

    def test_neg_log_loss_objective_is_loss_like(self):
        metric = Metric("neg_log_loss", "multiclass")
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
        ])

        # Loss-like objective values should be non-negative.
        assert metric.score(y_true, y_pred) >= 0.0


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

class TestRegressionMetric:
    def test_neg_root_mean_squared_error(self):
        metric = Metric("neg_root_mean_squared_error", "regression")
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])  # Regression uses 1D
        # Perfect prediction: RMSE = 0 → neg_rmse = 0
        assert metric.score(y_true, y_pred) == approx(0.0)

    def test_regression_score_not_negated(self):
        """sklearn's neg_root_mean_squared_error is already negative; Metric returns it negated, so positive."""
        metric = Metric("neg_root_mean_squared_error", "regression")
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])  # RMSE = 1.0 → neg_rmse = -1.0, which we negate, so we expect 1.0
        assert metric.score(y_true, y_pred) == approx(1.0)

    def test_regression_worse_prediction_lower_score(self):
        metric = Metric("neg_root_mean_squared_error", "regression")
        y_true = np.array([0.0, 0.0, 0.0])
        score_small_error = metric.score(y_true, np.array([0.1, 0.1, 0.1]))
        score_large_error = metric.score(y_true, np.array([1.0, 1.0, 1.0]))
        assert score_small_error < score_large_error


# ---------------------------------------------------------------------------
# Integration: metric used inside run_task
# ---------------------------------------------------------------------------

class TestMetricInFramework:
    def test_roc_auc_produces_valid_scores(self, unique_result_path):
        """roc_auc must produce scores in [-1, 0] for all folds."""
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.iterations = 3
        task.bo_initial_random_iterations = 2
        task.metric = "roc_auc"
        task.optimizer = "smac"
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        for run in history.history:
            for fold in run.folds:
                assert -1.0 <= fold.scores.val <= 0.0, f"roc_auc val score out of range: {fold.scores.val}"

    def test_accuracy_produces_valid_scores(self, unique_result_path):
        """accuracy must produce scores in [-1, 0]."""
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.iterations = 3
        task.bo_initial_random_iterations = 2
        task.metric = "accuracy"
        task.optimizer = "smac"
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        for run in history.history:
            for fold in run.folds:
                assert -1.0 <= fold.scores.val <= 0.0

    def test_stored_val_predictions_are_probabilities(self, unique_result_path):
        """
        Stored val predictions (fold.preds.val) must be probabilities
        in [0, 1] (positive-class probabilities for binary classification).
        """
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.iterations = 3
        task.bo_initial_random_iterations = 2
        task.metric = "roc_auc"
        task.optimizer = "smac"
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        for run in history.history:
            for fold in run.folds:
                preds = fold.preds.val
                assert preds is not None
                # All values should be probabilities ∈ [0, 1]
                assert np.all(preds >= 0.0) and np.all(preds <= 1.0), (
                    f"Val predictions are not in [0, 1]: min={preds.min()}, max={preds.max()}"
                )

    def test_stored_test_predictions_are_probabilities(self, unique_result_path):
        """Stored test predictions must also be probabilities."""
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.iterations = 3
        task.bo_initial_random_iterations = 2
        task.metric = "roc_auc"
        task.optimizer = "smac"
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        for run in history.history:
            for fold in run.folds:
                preds = fold.preds.test
                assert preds is not None
                assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    @pytest.mark.skipif(
        not (DATASETS_DIR / "363614.csv").exists(),
        reason="Requires offline dataset 363614.csv",
    )
    def test_multiclass_neg_log_loss_is_minimize_aligned(self, unique_result_path):
        """
        Framework integration test: multiclass neg_log_loss should behave like
        a loss objective for minimization.
        """
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.iterations = 3
        task.bo_initial_random_iterations = 2
        task.problem_type = "multiclass"
        task.dataset_id = 363614
        task.metric = "neg_log_loss"
        task.optimizer = "smac"
        task.debug = True
        task.offline_data_loading = True
        task.result_path = unique_result_path

        history = run_task(task)
        for run in history.history:
            for fold in run.folds:
                assert np.isfinite(fold.scores.val)
                assert fold.scores.val >= 0.0, (
                    f"Expected loss-like val score >= 0.0 for neg_log_loss, got {fold.scores.val}"
                )
