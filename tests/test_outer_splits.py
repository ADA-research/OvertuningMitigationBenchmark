"""
Tests for outer data splitting (TaskDataSplitter).
Key concern: same config → same split (reproducibility),
and each outer repeat in CV gives a different split.
"""
import numpy as np
import pandas as pd
import pytest

from src.experiments.task.task_config import OuterEvaluationConfig
from src.experiments.task.task_data_splitter import TaskDataSplitter


def make_data(n=1000, n_features=10, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.integers(0, 2, n), name="target")
    return X, y


class TestHoldoutSplitter:
    def test_basic_split_sizes(self):
        X, y = make_data(n=1000)
        cfg = OuterEvaluationConfig(resampling="holdout", train_size=0.7)
        splitter = TaskDataSplitter(cfg, random_state=42)
        X_train, X_test, y_train, y_test = splitter.make_outer_split(X, y)

        assert len(X_train) == 700
        assert len(X_test) == 300
        assert len(y_train) == 700
        assert len(y_test) == 300

    def test_no_overlap_between_train_and_test(self):
        X, y = make_data(n=500)
        cfg = OuterEvaluationConfig(resampling="holdout", train_size=0.8)
        splitter = TaskDataSplitter(cfg, random_state=0)
        X_train, X_test, y_train, y_test = splitter.make_outer_split(X, y)

        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert len(train_idx & test_idx) == 0

    def test_covers_all_samples(self):
        X, y = make_data(n=500)
        cfg = OuterEvaluationConfig(resampling="holdout", train_size=0.8)
        splitter = TaskDataSplitter(cfg, random_state=0)
        X_train, X_test, _, _ = splitter.make_outer_split(X, y)

        assert len(X_train) + len(X_test) == len(X)

    def test_reproducibility_same_seed(self):
        X, y = make_data(n=1000)
        cfg = OuterEvaluationConfig(resampling="holdout", train_size=0.7)

        s1 = TaskDataSplitter(cfg, random_state=42)
        s2 = TaskDataSplitter(cfg, random_state=42)

        Xt1, _, yt1, _ = s1.make_outer_split(X, y)
        Xt2, _, yt2, _ = s2.make_outer_split(X, y)

        assert list(Xt1.index) == list(Xt2.index)
        assert list(yt1) == list(yt2)

    def test_different_seeds_give_different_splits(self):
        X, y = make_data(n=1000)
        cfg = OuterEvaluationConfig(resampling="holdout", train_size=0.7)

        s1 = TaskDataSplitter(cfg, random_state=1)
        s2 = TaskDataSplitter(cfg, random_state=2)

        Xt1, _, _, _ = s1.make_outer_split(X, y)
        Xt2, _, _, _ = s2.make_outer_split(X, y)

        assert list(Xt1.index) != list(Xt2.index)

    def test_absolute_train_size(self):
        X, y = make_data(n=1000)
        cfg = OuterEvaluationConfig(resampling="holdout", train_size=600)
        splitter = TaskDataSplitter(cfg, random_state=42)
        X_train, X_test, _, _ = splitter.make_outer_split(X, y)

        assert len(X_train) == 600
        assert len(X_test) == 400

    def test_train_size_zero_raises(self):
        """
        benchmark_lite.py sets train_size=0 for holdout outer split.
        This must raise because sklearn cannot make a zero-size training set.
        This test documents the known bug in benchmark_lite.py.
        """
        X, y = make_data(n=1000)
        cfg = OuterEvaluationConfig(resampling="holdout", train_size=0)
        splitter = TaskDataSplitter(cfg, random_state=78)

        with pytest.raises((ValueError, Exception)):
            splitter.make_outer_split(X, y)


class TestCVSplitter:
    def test_cv_returns_correct_fraction(self):
        """With 3-fold CV, test set is ~1/3 of the data."""
        X, y = make_data(n=900)
        cfg = OuterEvaluationConfig(
            resampling="cv",
            n_folds=3,
            n_repeats=1,
            fold=0,
            repeat=0,
        )
        splitter = TaskDataSplitter(cfg, random_state=78)
        X_train, X_test, y_train, y_test = splitter.make_outer_split(X, y)

        assert abs(len(X_test) - 300) <= 1  # allow ±1 for rounding

    def test_cv_no_overlap(self):
        X, y = make_data(n=600)
        cfg = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=1, fold=0, repeat=0)
        splitter = TaskDataSplitter(cfg, random_state=78)
        X_train, X_test, _, _ = splitter.make_outer_split(X, y)

        assert len(set(X_train.index) & set(X_test.index)) == 0

    def test_cv_covers_all_samples(self):
        X, y = make_data(n=600)
        cfg = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=1, fold=0, repeat=0)
        splitter = TaskDataSplitter(cfg, random_state=78)
        X_train, X_test, _, _ = splitter.make_outer_split(X, y)

        assert len(X_train) + len(X_test) == len(X)

    def test_cv_reproducibility_same_config(self):
        """Identical config + seed must produce identical train/test splits."""
        X, y = make_data(n=600)
        cfg = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=5, fold=1, repeat=2)

        s1 = TaskDataSplitter(cfg, random_state=78)
        s2 = TaskDataSplitter(cfg, random_state=78)

        Xt1, XT1, yt1, yT1 = s1.make_outer_split(X, y)
        Xt2, XT2, yt2, yT2 = s2.make_outer_split(X, y)

        assert list(Xt1.index) == list(Xt2.index)
        assert list(XT1.index) == list(XT2.index)

    def test_cv_different_folds_give_different_test_sets(self):
        """Different fold indices must produce different test splits."""
        X, y = make_data(n=600)
        base_cfg = dict(resampling="cv", n_folds=3, n_repeats=1, repeat=0)

        splits = []
        for fold in range(3):
            cfg = OuterEvaluationConfig(**base_cfg, fold=fold)
            splitter = TaskDataSplitter(cfg, random_state=78)
            _, X_test, _, _ = splitter.make_outer_split(X, y)
            splits.append(set(X_test.index))

        # All three test sets should be disjoint
        assert splits[0].isdisjoint(splits[1])
        assert splits[1].isdisjoint(splits[2])
        assert splits[0].isdisjoint(splits[2])

    def test_cv_different_repeats_give_different_splits(self):
        """Different repeat indices must give different train/test allocations."""
        X, y = make_data(n=600)

        cfg0 = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=10, fold=0, repeat=0)
        cfg1 = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=10, fold=0, repeat=1)

        s0 = TaskDataSplitter(cfg0, random_state=78)
        s1 = TaskDataSplitter(cfg1, random_state=78)

        _, X_test0, _, _ = s0.make_outer_split(X, y)
        _, X_test1, _, _ = s1.make_outer_split(X, y)

        # Different repeats must produce different test samples
        assert set(X_test0.index) != set(X_test1.index), (
            "Repeat 0 and Repeat 1 must produce different outer splits. "
            "If they are the same the benchmark repeat structure is broken."
        )

    def test_cv_all_ten_repeats_are_unique(self):
        """All 10 benchmark repeats (fold=0) must produce distinct test splits."""
        X, y = make_data(n=600)
        test_sets = []
        for repeat in range(10):
            cfg = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=10, fold=0, repeat=repeat)
            splitter = TaskDataSplitter(cfg, random_state=78)
            _, X_test, _, _ = splitter.make_outer_split(X, y)
            test_sets.append(frozenset(X_test.index))

        # All 10 test sets should be distinct
        assert len(set(test_sets)) == 10, (
            "Not all 10 benchmark repeats produce distinct outer test splits."
        )

    def test_cv_invalid_fold_raises(self):
        X, y = make_data(n=300)
        cfg = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=1, fold=5, repeat=0)
        splitter = TaskDataSplitter(cfg, random_state=78)

        with pytest.raises(ValueError):
            splitter.make_outer_split(X, y)

    def test_cv_real_benchmark_split_consistency(self):
        """
        Simulate benchmark_lite outer split: CV 3-fold, 10 repeats, fold=0 for each repeat.
        Ensure that the split for (repeat=0, fold=0) with n_repeats=10 is consistent across
        calls, meaning we can reconstruct from any task in the benchmark for that dataset.
        """
        from src.datasets.offline_dataloader import OfflineDataLoader

        loader = OfflineDataLoader()
        X, y, _ = loader.load(1590, problem_type="binary")

        cfg = OuterEvaluationConfig(resampling="cv", n_folds=3, n_repeats=10, fold=0, repeat=0)
        s1 = TaskDataSplitter(cfg, random_state=78)
        s2 = TaskDataSplitter(cfg, random_state=78)

        Xt1, Xtest1, _, _ = s1.make_outer_split(X, y)
        Xt2, Xtest2, _, _ = s2.make_outer_split(X, y)

        assert list(Xt1.index) == list(Xt2.index)
        assert list(Xtest1.index) == list(Xtest2.index)
