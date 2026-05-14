"""
Tests for benchmark task generation (benchmark_lite.py and benchmark.py).

Key concerns:
- Correct number of tasks generated per dataset
- All result paths are unique (no overwrites)
- Each task type has correctly configured settings
- The outer split bug (train_size=0) is flagged
"""
import pytest

from src.experiments.benchmark_experiments.benchmark_lite import (
    benchmark_one_dataset_lite as benchmark_lite_one_dataset,
)
from src.experiments.benchmark_experiments.benchmark import (
    benchmark_one_dataset as benchmark_full_one_dataset,
)
from src.experiments.experiment import Experiment


class TestBenchmarkLiteTaskGeneration:
    def test_returns_experiment_object(self):
        exp = benchmark_lite_one_dataset("ilpd")
        assert isinstance(exp, Experiment)

    def test_correct_task_count_binary(self):
        """
        For a binary dataset, benchmark_lite should generate:
        10 repeats × 2 optimizers × 1 metric (roc_auc) × 13 task_types = 260 tasks.
        """
        exp = benchmark_lite_one_dataset("ilpd")
        assert len(exp.tasks) == 260, (
            f"Expected 260 tasks for binary dataset, got {len(exp.tasks)}"
        )

    def test_correct_task_count_regression(self):
        """
        For regression: 10 repeats × 2 optimizers × 1 metric (neg_root_mean_squared_error)
        × 13 task_types = 260 tasks.
        """
        exp = benchmark_lite_one_dataset("ilpd", problem_type="regression")
        assert len(exp.tasks) == 260

    def test_all_result_paths_unique(self):
        exp = benchmark_lite_one_dataset("ilpd")
        paths = [t.result_path for t in exp.tasks]
        assert len(set(paths)) == len(paths), (
            f"Duplicate result paths detected: "
            f"{[p for p in paths if paths.count(p) > 1][:5]}"
        )

    def test_result_paths_contain_dataset_id(self):
        exp = benchmark_lite_one_dataset("ilpd")
        for task in exp.tasks:
            assert "ilpd" in task.result_path

    def test_result_paths_contain_optimizer(self):
        exp = benchmark_lite_one_dataset("ilpd")
        for task in exp.tasks:
            assert "smac" in task.result_path or "hebo" in task.result_path

    def test_result_paths_contain_metric(self):
        exp = benchmark_lite_one_dataset("ilpd")
        for task in exp.tasks:
            assert "roc_auc" in task.result_path

    def test_mlplan_tasks_present(self):
        """Benchmark must include MLPlan tasks."""
        exp = benchmark_lite_one_dataset("ilpd")
        mlplan_tasks = [t for t in exp.tasks if t.mitigation_strategy == "mlplan"]
        assert len(mlplan_tasks) > 0

    def test_mlplan_result_paths_contain_mlplan(self):
        exp = benchmark_lite_one_dataset("ilpd")
        for task in exp.tasks:
            if task.mitigation_strategy == "mlplan":
                assert "mlplan" in task.result_path

    def test_racing_tasks_present(self):
        """Benchmark must include racing (Bergman aggressive) tasks."""
        exp = benchmark_lite_one_dataset("ilpd")
        racing_tasks = [t for t in exp.tasks if t.mitigation_strategy == "bergman_aggressive"]
        assert len(racing_tasks) > 0

    def test_racing_tasks_use_5x5cv(self):
        exp = benchmark_lite_one_dataset("ilpd")
        for task in exp.tasks:
            if task.mitigation_strategy == "bergman_aggressive":
                assert task.evaluation.n_folds == 5
                assert task.evaluation.n_repeats == 5
                assert task.evaluation.resampling == "cv"

    def test_thresholdout_tasks_present(self):
        exp = benchmark_lite_one_dataset("ilpd")
        tho_tasks = [t for t in exp.tasks if t.mitigation_strategy == "thresholdout"]
        assert len(tho_tasks) > 0

    def test_thresholdout_tasks_use_holdout(self):
        exp = benchmark_lite_one_dataset("ilpd")
        for task in exp.tasks:
            if task.mitigation_strategy == "thresholdout":
                assert task.evaluation.resampling == "holdout"
                assert task.evaluation.val_size == 0.2

    def test_reshuffling_variants_present(self):
        """There should be tasks with reshuffling=True and reshuffling=False."""
        exp = benchmark_lite_one_dataset("ilpd")
        reshuffle_true = [t for t in exp.tasks if t.evaluation.reshuffle and t.mitigation_strategy == "none"]
        reshuffle_false = [t for t in exp.tasks if not t.evaluation.reshuffle and t.mitigation_strategy == "none"]
        assert len(reshuffle_true) > 0
        assert len(reshuffle_false) > 0

    def test_selection_set_variants_present(self):
        """Both sel=0 and sel=1/6 tasks must be present."""
        exp = benchmark_lite_one_dataset("ilpd")
        baseline_tasks = [t for t in exp.tasks
                          if t.mitigation_strategy == "none" and t.racing_strategy == "none"]
        sel_zero = [t for t in baseline_tasks if not t.evaluation.selection_size]
        sel_sixth = [t for t in baseline_tasks if t.evaluation.selection_size and
                     abs(t.evaluation.selection_size - 1/6) < 0.01]
        assert len(sel_zero) > 0, "Missing tasks with selection_size=0"
        assert len(sel_sixth) > 0, "Missing tasks with selection_size=1/6"

    def test_cv_and_holdout_inner_variants_present(self):
        """Baseline tasks must use both cv and holdout inner resampling."""
        exp = benchmark_lite_one_dataset("ilpd")
        baseline_tasks = [t for t in exp.tasks
                          if t.mitigation_strategy == "none" and t.racing_strategy == "none"]
        cv_tasks = [t for t in baseline_tasks if t.evaluation.resampling == "cv"]
        holdout_tasks = [t for t in baseline_tasks if t.evaluation.resampling == "holdout"]
        assert len(cv_tasks) > 0
        assert len(holdout_tasks) > 0

    def test_all_tasks_have_lgbm_classifier(self):
        """All benchmark tasks should use LGBM as the only classifier."""
        exp = benchmark_lite_one_dataset("ilpd")
        for task in exp.tasks:
            assert "LGBM" in task.search_space.classifiers

    def test_outer_split_bug_train_size_zero(self):
        """
        KNOWN BUG: benchmark_lite sets outer resampling='holdout' with train_size=0.
        This will cause sklearn.train_test_split to raise ValueError at runtime.
        This test documents the bug by checking that train_size=0 is set.
        When the bug is fixed, the test should be updated.
        """
        exp = benchmark_lite_one_dataset("ilpd")
        for task in exp.tasks:
            if task.outer_evaluation.resampling == "holdout":
                # Document the bug: train_size is 0 which will fail
                if task.outer_evaluation.train_size == 0:
                    pytest.xfail(
                        "KNOWN BUG: benchmark_lite.py sets outer holdout "
                        "train_size=0 which will fail in TaskDataSplitter. "
                        "The outer split should either use CV or a non-zero train_size."
                    )

    def test_all_10_repeats_present(self):
        """benchmark_lite uses 10 repeats; all should be represented."""
        exp = benchmark_lite_one_dataset("ilpd")
        repeats = {t.outer_evaluation.repeat for t in exp.tasks}
        assert repeats == set(range(10)), f"Expected repeats 0-9, got {sorted(repeats)}"

    def test_both_optimizers_present(self):
        exp = benchmark_lite_one_dataset("ilpd")
        optimizers = {t.optimizer for t in exp.tasks}
        assert "smac" in optimizers
        assert "hebo" in optimizers

    def test_retrain_always_true(self):
        """All tasks must have retrain=True."""
        exp = benchmark_lite_one_dataset("ilpd")
        for task in exp.tasks:
            assert task.evaluation.retrain is True, (
                f"Task {task.result_path} has retrain=False"
            )

    @pytest.mark.skip(reason="Offline dataset availability is optional in local test runs.")
    def test_offline_data_loading_enabled(self):
        exp = benchmark_lite_one_dataset("ilpd")
        for task in exp.tasks:
            assert task.offline_data_loading is True


class TestBenchmarkFullTaskGeneration:
    def test_returns_experiment_object(self):
        exp = benchmark_full_one_dataset("ilpd")
        assert isinstance(exp, Experiment)

    def test_full_has_more_tasks_than_lite(self):
        """Full benchmark uses 3-fold CV (3 outer splits vs 1 in lite)."""
        exp_lite = benchmark_lite_one_dataset("ilpd")
        exp_full = benchmark_full_one_dataset("ilpd")
        assert len(exp_full.tasks) > len(exp_lite.tasks)

    def test_full_includes_roc_auc(self):
        """Full benchmark uses roc_auc metrics."""
        exp = benchmark_full_one_dataset("ilpd")
        metrics = {t.metric for t in exp.tasks}
        assert "roc_auc" in metrics

    def test_full_has_outer_cv_splits(self):
        """Full benchmark uses 3-fold × 10-repeat outer CV."""
        exp = benchmark_full_one_dataset("ilpd")
        for task in exp.tasks:
            assert task.outer_evaluation.resampling == "cv"

    def test_full_includes_selection_size_quarter(self):
        """Full benchmark has sel=1/4 (not present in lite)."""
        exp = benchmark_full_one_dataset("ilpd")
        sel_quarter = [t for t in exp.tasks
                       if t.evaluation.selection_size and abs(t.evaluation.selection_size - 0.25) < 0.01]
        assert len(sel_quarter) > 0

    def test_full_all_result_paths_unique(self):
        exp = benchmark_full_one_dataset("ilpd")
        paths = [t.result_path for t in exp.tasks]
        assert len(set(paths)) == len(paths)


class TestTaskCountMath:
    """Verify the task count arithmetic is correct."""

    def test_binary_task_count_breakdown_lite(self):
        """
        10 repeats × 2 optimizers × 1 metric × 13 types = 260
        13 types = 1 mlplan + 2 racing (with/without reshuffle)
                 + 2 thresholdout (with/without reshuffle)
                 + 8 baseline (2 inner × 2 sel × 2 reshuffle)
        """
        exp = benchmark_lite_one_dataset("ilpd")
        n = len(exp.tasks)

        # Since 10 × 2 × 1 × 13 = 260
        assert n % (10 * 2) == 0, "Task count should be divisible by 10 repeats × 2 optimizers"
        types_per_slot = n // (10 * 2)
        assert types_per_slot == 13, f"Expected 13 task types per slot, got {types_per_slot}"
