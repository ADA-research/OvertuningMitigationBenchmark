"""
Tests for online resamplers: CVResampler, HoldoutResampler,
CVSelectionSetResampler, HoldoutSelectionSetResampler.

Key invariants:
- Correct fold count per iteration
- Test set unchanged across iterations
- Reshuffling changes val splits; non-reshuffling keeps them identical
- Selection set is consistent and separate from train/val
"""
import numpy as np
import pandas as pd
import pytest

from src.resamplers.online_resamplers.cv_resampler import CVResampler
from src.resamplers.online_resamplers.cv_selection_set_resampler import CVSelectionSetResampler
from src.resamplers.online_resamplers.holdout_resampler import HoldoutResampler
from src.resamplers.online_resamplers.holdout_selection_set_resampler import HoldoutSelectionSetResampler


def make_split_data(n_train=400, n_test=100, seed=42):
    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame(rng.standard_normal((n_train, 5)), columns=list("abcde"))
    y_train = pd.Series(rng.integers(0, 2, n_train))
    X_test = pd.DataFrame(rng.standard_normal((n_test, 5)), columns=list("abcde"))
    y_test = pd.Series(rng.integers(0, 2, n_test))
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# CVResampler
# ---------------------------------------------------------------------------

class TestCVResampler:
    def test_yields_n_folds_per_iteration(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = CVResampler(X_tr, y_tr, X_te, y_te, n_folds=5, n_repeats=1, seed=0)
        folds = list(resampler)
        assert len(folds) == 5

    def test_each_fold_has_four_components(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = CVResampler(X_tr, y_tr, X_te, y_te, n_folds=3, n_repeats=1, seed=0)
        for (Xt, yt), (Xv, yv), (Xtest, ytest), (Xsel, ysel) in resampler:
            assert Xt is not None
            assert Xv is not None
            assert Xtest is not None
            # No selection set: sel should be None
            assert Xsel is None
            assert ysel is None

    def test_train_val_no_overlap(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = CVResampler(X_tr, y_tr, X_te, y_te, n_folds=5, n_repeats=1, seed=0)
        for (Xt, _), (Xv, _), _, _ in resampler:
            assert len(set(Xt.index) & set(Xv.index)) == 0

    def test_test_set_unchanged_across_folds(self):
        """The test set must be identical in every fold of the same HPO iteration."""
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = CVResampler(X_tr, y_tr, X_te, y_te, n_folds=5, n_repeats=1, seed=0)
        test_indices = []
        for _, _, (Xtest, _), _ in resampler:
            test_indices.append(list(Xtest.index))
        # All test sets should be the same
        assert all(t == test_indices[0] for t in test_indices)

    def test_repeated_cv_fold_count(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = CVResampler(X_tr, y_tr, X_te, y_te, n_folds=5, n_repeats=3, seed=0)
        folds = list(resampler)
        assert len(folds) == 15  # 3 repeats × 5 folds

    def test_no_reshuffle_same_val_splits(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = CVResampler(X_tr, y_tr, X_te, y_te, n_folds=5, n_repeats=1, seed=0, reshuffle=False)

        # First pass
        iter1 = [(list(Xv.index)) for _, (Xv, _), _, _ in resampler]
        # Second pass (resampler is reset by __iter__)
        iter2 = [(list(Xv.index)) for _, (Xv, _), _, _ in resampler]

        assert iter1 == iter2

    def test_reshuffle_changes_val_splits(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = CVResampler(X_tr, y_tr, X_te, y_te, n_folds=5, n_repeats=1, seed=0, reshuffle=True)

        iter1 = [list(Xv.index) for _, (Xv, _), _, _ in resampler]
        iter2 = [list(Xv.index) for _, (Xv, _), _, _ in resampler]

        assert iter1 != iter2, "Reshuffling should change val splits between iterations"

    def test_reshuffle_test_set_stays_same(self):
        """Even with reshuffling, the test set must never change."""
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = CVResampler(X_tr, y_tr, X_te, y_te, n_folds=3, n_repeats=1, seed=0, reshuffle=True)

        test_idx_iter1 = [list(Xtest.index) for _, _, (Xtest, _), _ in resampler]
        test_idx_iter2 = [list(Xtest.index) for _, _, (Xtest, _), _ in resampler]

        assert test_idx_iter1 == test_idx_iter2


# ---------------------------------------------------------------------------
# HoldoutResampler
# ---------------------------------------------------------------------------

class TestHoldoutResampler:
    def test_yields_one_fold_per_repeat(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = HoldoutResampler(X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, n_repeats=1, seed=0)
        folds = list(resampler)
        assert len(folds) == 1

    def test_multiple_repeats_fold_count(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = HoldoutResampler(X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, n_repeats=5, seed=0)
        folds = list(resampler)
        assert len(folds) == 5

    def test_holdout_fraction_respected(self):
        X_tr, y_tr, X_te, y_te = make_split_data(n_train=400)
        resampler = HoldoutResampler(X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, n_repeats=1, seed=0)
        for _, (Xv, _), _, _ in resampler:
            # Val set should be ~20% of 400 = ~80 samples
            assert abs(len(Xv) - 80) <= 2

    def test_no_selection_set_returned(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = HoldoutResampler(X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, n_repeats=1, seed=0)
        for _, _, _, (Xsel, ysel) in resampler:
            assert Xsel is None
            assert ysel is None

    def test_no_reshuffle_identical_val_splits(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = HoldoutResampler(X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, n_repeats=1, seed=0, reshuffle=False)

        iter1 = [list(Xv.index) for _, (Xv, _), _, _ in resampler]
        iter2 = [list(Xv.index) for _, (Xv, _), _, _ in resampler]
        assert iter1 == iter2

    def test_reshuffle_changes_val_splits(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = HoldoutResampler(X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, n_repeats=1, seed=0, reshuffle=True)

        iter1 = [list(Xv.index) for _, (Xv, _), _, _ in resampler]
        iter2 = [list(Xv.index) for _, (Xv, _), _, _ in resampler]
        assert iter1 != iter2


# ---------------------------------------------------------------------------
# CVSelectionSetResampler
# ---------------------------------------------------------------------------

class TestCVSelectionSetResampler:
    def test_yields_n_folds(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = CVSelectionSetResampler(X_tr, y_tr, X_te, y_te, n_folds=5, selection_fraction=0.1, seed=0)
        folds = list(resampler)
        assert len(folds) == 5

    def test_selection_set_present_and_non_empty(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = CVSelectionSetResampler(X_tr, y_tr, X_te, y_te, n_folds=5, selection_fraction=0.1, seed=0)
        for _, _, _, (Xsel, ysel) in resampler:
            assert Xsel is not None
            assert ysel is not None
            assert len(Xsel) > 0

    def test_selection_fraction_approximately_correct(self):
        X_tr, y_tr, X_te, y_te = make_split_data(n_train=400)
        fraction = 0.25
        resampler = CVSelectionSetResampler(X_tr, y_tr, X_te, y_te, n_folds=5, selection_fraction=fraction, seed=0)
        for _, _, _, (Xsel, _) in resampler:
            # Selection set is ~25% of 400 = ~100 samples
            expected = int(400 * fraction)
            assert abs(len(Xsel) - expected) <= 5

    def test_selection_set_same_across_folds(self):
        """Selection set is carved once before CV; must be identical across all folds."""
        X_tr, y_tr, X_te, y_te = make_split_data(n_train=400)
        resampler = CVSelectionSetResampler(X_tr, y_tr, X_te, y_te, n_folds=5, selection_fraction=0.1, seed=0)
        sel_indices = [list(Xsel.index) for _, _, _, (Xsel, _) in resampler]
        assert all(s == sel_indices[0] for s in sel_indices)

    def test_selection_set_consistent_across_iterations(self):
        """Selection set must not change between HPO iterations (reshuffling should not affect it)."""
        X_tr, y_tr, X_te, y_te = make_split_data(n_train=400)
        resampler = CVSelectionSetResampler(X_tr, y_tr, X_te, y_te, n_folds=3, selection_fraction=0.1, seed=0, reshuffle=False)

        iter1 = [list(Xsel.index) for _, _, _, (Xsel, _) in resampler]
        iter2 = [list(Xsel.index) for _, _, _, (Xsel, _) in resampler]
        assert iter1 == iter2

    def test_no_overlap_between_train_val_and_selection(self):
        X_tr, y_tr, X_te, y_te = make_split_data(n_train=400)
        resampler = CVSelectionSetResampler(X_tr, y_tr, X_te, y_te, n_folds=5, selection_fraction=0.1, seed=0)
        for (Xt, _), (Xv, _), _, (Xsel, _) in resampler:
            all_tv = set(Xt.index) | set(Xv.index)
            assert len(all_tv & set(Xsel.index)) == 0


# ---------------------------------------------------------------------------
# HoldoutSelectionSetResampler
# ---------------------------------------------------------------------------

class TestHoldoutSelectionSetResampler:
    def test_yields_one_fold(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = HoldoutSelectionSetResampler(
            X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, selection_fraction=0.1, n_repeats=1, seed=0
        )
        folds = list(resampler)
        assert len(folds) == 1

    def test_selection_set_present(self):
        X_tr, y_tr, X_te, y_te = make_split_data()
        resampler = HoldoutSelectionSetResampler(
            X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, selection_fraction=0.1, n_repeats=1, seed=0
        )
        for _, _, _, (Xsel, ysel) in resampler:
            assert Xsel is not None
            assert ysel is not None
            assert len(Xsel) > 0

    def test_no_overlap_selection_and_val(self):
        X_tr, y_tr, X_te, y_te = make_split_data(n_train=400)
        resampler = HoldoutSelectionSetResampler(
            X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, selection_fraction=0.1, n_repeats=1, seed=0
        )
        for (Xt, _), (Xv, _), _, (Xsel, _) in resampler:
            all_tv = set(Xt.index) | set(Xv.index)
            assert len(all_tv & set(Xsel.index)) == 0

    def test_selection_set_consistent_across_iterations(self):
        X_tr, y_tr, X_te, y_te = make_split_data(n_train=400)
        resampler = HoldoutSelectionSetResampler(
            X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, selection_fraction=0.15, n_repeats=1, seed=0, reshuffle=False
        )
        iter1 = [list(Xsel.index) for _, _, _, (Xsel, _) in resampler]
        iter2 = [list(Xsel.index) for _, _, _, (Xsel, _) in resampler]
        assert iter1 == iter2


# ---------------------------------------------------------------------------
# Benchmark-realistic resampler configurations
# ---------------------------------------------------------------------------

class TestBenchmarkResamplerConfigs:
    """Test the specific resampler configurations used in benchmark_lite."""

    def test_racing_resampler_5x5cv(self):
        """Racing uses 5 repeats × 5 folds = 25 total folds per config."""
        X_tr, y_tr, X_te, y_te = make_split_data(n_train=500)
        resampler = CVResampler(X_tr, y_tr, X_te, y_te, n_folds=5, n_repeats=5, seed=78)
        folds = list(resampler)
        assert len(folds) == 25

    def test_thresholdout_holdout_resampler(self):
        """Thresholdout uses holdout with val_size=0.2."""
        X_tr, y_tr, X_te, y_te = make_split_data(n_train=400)
        resampler = HoldoutResampler(X_tr, y_tr, X_te, y_te, holdout_fraction=0.2, n_repeats=1, seed=78)
        folds = list(resampler)
        assert len(folds) == 1
        (Xt, _), (Xv, _), _, _ = folds[0]
        assert len(Xv) / (len(Xt) + len(Xv)) == pytest.approx(0.2, abs=0.02)

    def test_baseline_cv_sel_16(self):
        """Baseline 5CV with 1/6 selection set."""
        X_tr, y_tr, X_te, y_te = make_split_data(n_train=600)
        resampler = CVSelectionSetResampler(
            X_tr, y_tr, X_te, y_te, n_folds=5, selection_fraction=1/6, seed=78
        )
        folds = list(resampler)
        assert len(folds) == 5
        for _, _, _, (Xsel, _) in folds:
            assert len(Xsel) > 0

    def test_mlplan_phase1_holdout_selection_5mccv(self):
        """MLPlan phase 1: HoldoutSelectionSetResampler with n_repeats=5, holdout=0.3, selection=0.3."""
        X_tr, y_tr, X_te, y_te = make_split_data(n_train=500)
        resampler = HoldoutSelectionSetResampler(
            X_tr, y_tr, X_te, y_te, holdout_fraction=0.3, selection_fraction=0.3, n_repeats=5, seed=78
        )
        folds = list(resampler)
        assert len(folds) == 5
        for _, _, _, (Xsel, _) in folds:
            assert len(Xsel) > 0
