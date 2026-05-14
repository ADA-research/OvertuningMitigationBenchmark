"""
Tests for reproducibility.

These tests are the last line of defense before the HPC run:
- Same outer split config → identical train/test split
- Same inner config + seed → identical fold order
- Same optimizer + seed → identical HPO trajectory
- Reshuffling behavior is deterministic per seed
- Different repeats in CV outer split → different splits (critical for benchmark)
"""

import warnings

import numpy as np
import pandas as pd
import pytest


def make_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 8)), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(rng.integers(0, 2, n), name="target")
    return X, y


class TestOuterSplitReproducibility:
    """
    The outer split is the first thing that happens in run_task.
    Any non-determinism here would corrupt all downstream analysis.
    """

    def test_cv_same_config_gives_same_split(self):
        from src.experiments.task.task_config import OuterEvaluationConfig
        from src.experiments.task.task_data_splitter import TaskDataSplitter

        X, y = make_data()
        cfg = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=10, fold=0, repeat=3)

        s1 = TaskDataSplitter(cfg, random_state=78)
        s2 = TaskDataSplitter(cfg, random_state=78)

        Xt1, Xte1, _, _ = s1.make_outer_split(X, y)
        Xt2, Xte2, _, _ = s2.make_outer_split(X, y)

        assert list(Xt1.index) == list(Xt2.index)
        assert list(Xte1.index) == list(Xte2.index)

    def test_all_benchmark_outer_splits_are_deterministic(self):
        """For every (fold, repeat) combination used in the full benchmark, the split is stable."""
        from src.experiments.task.task_config import OuterEvaluationConfig
        from src.experiments.task.task_data_splitter import TaskDataSplitter

        X, y = make_data()
        # Full benchmark: 3 folds × 10 repeats
        for repeat in range(10):
            for fold in range(3):
                cfg = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=10, fold=fold, repeat=repeat)
                s1 = TaskDataSplitter(cfg, random_state=78)
                s2 = TaskDataSplitter(cfg, random_state=78)
                Xt1, _, _, _ = s1.make_outer_split(X, y)
                Xt2, _, _, _ = s2.make_outer_split(X, y)
                assert list(Xt1.index) == list(Xt2.index), (
                    f"Non-deterministic split for fold={fold}, repeat={repeat}"
                )

    def test_different_repeats_give_different_splits(self):
        """Each of the 10 repeats must produce a different outer test split."""
        from src.experiments.task.task_config import OuterEvaluationConfig
        from src.experiments.task.task_data_splitter import TaskDataSplitter

        X, y = make_data()
        test_index_sets = []
        for repeat in range(10):
            cfg = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=10, fold=0, repeat=repeat)
            splitter = TaskDataSplitter(cfg, random_state=78)
            _, X_test, _, _ = splitter.make_outer_split(X, y)
            test_index_sets.append(frozenset(X_test.index))

        n_unique = len(set(test_index_sets))
        assert n_unique == 10, (
            f"Only {n_unique}/10 distinct outer test splits. "
            "Benchmark repeats would produce identical runs!"
        )

    def test_different_folds_give_different_test_sets(self):
        """The 3 CV folds must produce disjoint test sets."""
        from src.experiments.task.task_config import OuterEvaluationConfig
        from src.experiments.task.task_data_splitter import TaskDataSplitter

        X, y = make_data()
        test_sets = []
        for fold in range(3):
            cfg = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=1, fold=fold, repeat=0)
            splitter = TaskDataSplitter(cfg, random_state=78)
            _, X_test, _, _ = splitter.make_outer_split(X, y)
            test_sets.append(set(X_test.index))

        # All folds disjoint
        for i in range(3):
            for j in range(i + 1, 3):
                assert test_sets[i].isdisjoint(test_sets[j]), \
                    f"Fold {i} and fold {j} test sets overlap"

    def test_real_dataset_split_stable_across_identical_tasks(self):
        """
        Two tasks in benchmark_lite with the same (repeat, fold, optimizer) must
        operate on the same train/test outer split.
        """
        from src.datasets.offline_dataloader import OfflineDataLoader
        from src.experiments.task.task_config import OuterEvaluationConfig
        from src.experiments.task.task_data_splitter import TaskDataSplitter

        loader = OfflineDataLoader()
        X, y, _ = loader.load(1590, problem_type="binary")

        cfg = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=10, fold=0, repeat=0)
        s1 = TaskDataSplitter(cfg, random_state=78)
        s2 = TaskDataSplitter(cfg, random_state=78)

        Xt1, Xte1, _, _ = s1.make_outer_split(X, y)
        Xt2, Xte2, _, _ = s2.make_outer_split(X, y)

        assert list(Xt1.index) == list(Xt2.index)
        assert list(Xte1.index) == list(Xte2.index)


class TestInnerResamplerReproducibility:
    def test_cv_same_seed_same_folds(self):
        from src.resamplers.online_resamplers.cv_resampler import CVResampler

        X_tr, y_tr = make_data(n=400)
        X_te, y_te = make_data(n=100, seed=99)

        r1 = CVResampler(X_tr, y_tr, X_te, y_te, n_folds=5, seed=42)
        r2 = CVResampler(X_tr, y_tr, X_te, y_te, n_folds=5, seed=42)

        folds1 = [list(Xv.index) for _, (Xv, _), _, _ in r1]
        folds2 = [list(Xv.index) for _, (Xv, _), _, _ in r2]
        assert folds1 == folds2

    def test_holdout_same_seed_same_split(self):
        from src.resamplers.online_resamplers.holdout_resampler import HoldoutResampler

        X_tr, y_tr = make_data(n=400)
        X_te, y_te = make_data(n=100, seed=99)

        r1 = HoldoutResampler(X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, seed=42)
        r2 = HoldoutResampler(X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, seed=42)

        val1 = [list(Xv.index) for _, (Xv, _), _, _ in r1]
        val2 = [list(Xv.index) for _, (Xv, _), _, _ in r2]
        assert val1 == val2

    def test_reshuffle_false_same_splits_across_iterations(self):
        """Without reshuffling, every HPO iteration sees the same val split."""
        from src.resamplers.online_resamplers.holdout_resampler import HoldoutResampler

        X_tr, y_tr = make_data(n=400)
        X_te, y_te = make_data(n=100, seed=99)
        r = HoldoutResampler(X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, seed=42, reshuffle=False)

        run1_val = [list(Xv.index) for _, (Xv, _), _, _ in r]
        run2_val = [list(Xv.index) for _, (Xv, _), _, _ in r]
        assert run1_val == run2_val

    def test_reshuffle_true_different_splits_across_iterations(self):
        """With reshuffling, each HPO iteration must see a different val split."""
        from src.resamplers.online_resamplers.holdout_resampler import HoldoutResampler

        X_tr, y_tr = make_data(n=400)
        X_te, y_te = make_data(n=100, seed=99)
        r = HoldoutResampler(X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, seed=42, reshuffle=True)

        run1_val = [list(Xv.index) for _, (Xv, _), _, _ in r]
        run2_val = [list(Xv.index) for _, (Xv, _), _, _ in r]
        assert run1_val != run2_val


class TestHPOReproducibility:
    """Full run_task reproducibility with same seed should produce identical results."""

    def test_random_search_reproducible(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        results = []
        for _ in range(3):
            task = DefaultTaskConfig()
            task.optimizer = "random_search"
            task.iterations = 5
            task.random_state = 78
            task.result_path = unique_result_path
            history = run_task(task)
            results.append(history)

        assert results[0] == results[1] == results[2], (
            "Random search with same seed must produce identical histories"
        )

    def test_smac_reproducible(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        results = []
        for i in range(3):
            task = DefaultTaskConfig()
            task.optimizer = "smac"
            task.iterations = 5
            task.bo_initial_random_iterations = 2
            task.random_state = 78
            task.debug = True
            task.result_path = f"{unique_result_path}_run{i}"
            history = run_task(task)
            results.append(history)

        assert results[0] == results[1] == results[2], (
            "SMAC with same seed must produce identical histories"
        )

    def test_hebo_reproducible(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        results = []
        for i in range(3):
            task = DefaultTaskConfig()
            task.optimizer = "hebo"
            task.iterations = 5
            task.bo_initial_random_iterations = 3
            task.random_state = 78
            task.debug = True
            task.result_path = f"{unique_result_path}_run{i}"
            task.evaluation.reshuffle = False
            history = run_task(task)
            results.append(history)
        
        assert results[0] == results[1] == results[2], (
            "HEBO with same seed must produce identical histories"
        )

    def test_different_seeds_produce_different_results_smac(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task1 = DefaultTaskConfig()
        task1.random_state = 78
        task1.optimizer = "smac"
        task1.iterations = 5
        task1.bo_initial_random_iterations = 2
        task1.debug = True
        task1.result_path = f"{unique_result_path}_seed78"

        task2 = DefaultTaskConfig()
        task2.random_state = 123
        task2.optimizer = "smac"
        task2.iterations = 5
        task2.bo_initial_random_iterations = 2
        task2.debug = True
        task2.result_path = f"{unique_result_path}_seed123"

        history1 = run_task(task1)
        history2 = run_task(task2)

        assert history1 != history2, "Different seeds should produce different results"

    def test_different_seeds_produce_different_results_hebo(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task1 = DefaultTaskConfig()
        task1.random_state = 78
        task1.optimizer = "hebo"
        task1.iterations = 5
        task1.bo_initial_random_iterations = 2
        task1.debug = True
        task1.result_path = f"{unique_result_path}_seed78"

        task2 = DefaultTaskConfig()
        task2.random_state = 123
        task2.optimizer = "hebo"
        task2.iterations = 5
        task2.bo_initial_random_iterations = 2
        task2.debug = True
        task2.result_path = f"{unique_result_path}_seed123"

        history1 = run_task(task1)
        history2 = run_task(task2)

        assert history1 != history2, "Different seeds should produce different results"

    def test_reshuffle_true_reproducible(self, unique_result_path):
        """Reshuffling must also be reproducible with the same seed."""
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        results = []
        for i in range(3):
            task = DefaultTaskConfig()
            task.evaluation.resampling = "holdout"
            task.evaluation.reshuffle = True
            task.evaluation.retrain = True
            task.iterations = 5
            task.random_state = 78
            task.debug = True
            task.optimizer = "smac"
            task.bo_initial_random_iterations = 2
            task.result_path = f"{unique_result_path}_run{i}"
            history = run_task(task)
            results.append(history)

        assert results[0] == results[1] == results[2]

    def test_cv_reproducible(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        results = []
        for i in range(3):
            task = DefaultTaskConfig()
            task.evaluation.resampling = "cv"
            task.evaluation.n_folds = 3
            task.evaluation.reshuffle = False
            task.iterations = 3
            task.random_state = 78
            task.debug = True
            task.optimizer = "smac"
            task.bo_initial_random_iterations = 2
            task.result_path = f"{unique_result_path}_run{i}"
            history = run_task(task)
            results.append(history)

        assert results[0] == results[1] == results[2]


class TestBenchmarkReproducibilityEndToEnd:
    """
    Simulate the exact benchmark_lite configuration for one task
    and verify that running it twice gives identical results.
    """

    def test_baseline_cv_task_reproducible(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import TaskConfig

        def make_benchmark_task(run_id):
            task = TaskConfig()
            task.dataset_id = "ilpd"
            task.problem_type = "binary"
            task.random_state = 78
            task.offline = False
            task.offline_data_loading = True
            task.racing_strategy = "none"
            task.mitigation_strategy = "none"
            task.iterations = 5
            task.bo_initial_random_iterations = 2
            task.smac_surrogate_model = "gaussian_process"
            task.debug = True
            task.store_results_google_cloud = False
            task.store_vectors_google_cloud = False
            task.search_space.classifiers = ["LGBM"]
            task.search_space.scalers = ["StandardScaler"]
            task.search_space.encoders = ["OneHotEncoder"]
            task.search_space.dim_reducers = ["None"]
            task.search_space.feat_selectors = ["None"]
            task.search_space.imputers = ["NumericalSimpleImputer"]
            task.evaluation.retrain = True
            task.outer_evaluation.fold = 0
            task.outer_evaluation.repeat = 0
            task.outer_evaluation.n_repeats = 10
            task.outer_evaluation.resampling = "holdout"
            task.outer_evaluation.train_size = 0.8  # FIXED: not 0
            task.optimizer = "smac"
            task.metric = "roc_auc"
            task.evaluation.reshuffle = False
            task.evaluation.resampling = "cv"
            task.evaluation.n_folds = 5
            task.evaluation.selection_size = 0.0
            task.result_path = f"{unique_result_path}_run{run_id}"
            return task

        h1 = run_task(make_benchmark_task(0))
        h2 = run_task(make_benchmark_task(1))
        assert h1 == h2, "Same benchmark task config must produce identical results"
