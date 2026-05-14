"""
Tests for data classes (Split, Fold, Surrogate, Run) and RunHistory.

The data classes are the core data structures that travel through the entire
benchmark pipeline. Correctness here is foundational.
"""
import numpy as np
import pytest
from pytest import approx

from src.history.data_classes import Fold, Run, Split, Surrogate
from src.history.run_history import RunHistory


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

class TestSplit:
    def test_construction(self):
        s = Split(train=0.8, val=0.7, test=0.6)
        assert s.train == 0.8
        assert s.val == 0.7
        assert s.test == 0.6
        assert s.selection is None

    def test_with_selection(self):
        s = Split(train=0.9, val=0.8, test=0.7, selection=0.75)
        assert s.selection == 0.75

    def test_immutable(self):
        s = Split(train=0.8, val=0.7, test=0.6)
        with pytest.raises((AttributeError, TypeError)):
            s.train = 0.5  # frozen dataclass

    def test_none_values_allowed(self):
        s = Split(train=None, val=None, test=None)
        assert s.train is None
        assert s.val is None

    def test_array_values(self):
        arr_val = np.array([0.1, 0.9])
        arr_test = np.array([0.2, 0.8])
        s = Split(train=None, val=arr_val, test=arr_test)
        assert np.array_equal(s.val, arr_val)
        assert np.array_equal(s.test, arr_test)


# ---------------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------------

class TestSurrogate:
    def test_construction(self):
        sur = Surrogate(mean=0.5, std=0.1, acquisition=0.05)
        assert sur.mean == 0.5
        assert sur.std == 0.1
        assert sur.acquisition == 0.05

    def test_acquisition_optional(self):
        sur = Surrogate(mean=0.3, std=0.2)
        assert sur.acquisition is None

    def test_immutable(self):
        sur = Surrogate(mean=0.5, std=0.1)
        with pytest.raises((AttributeError, TypeError)):
            sur.mean = 0.0


# ---------------------------------------------------------------------------
# Fold
# ---------------------------------------------------------------------------

class TestFold:
    def test_construction(self):
        fold = Fold(
            fold_id=0,
            scores=Split(train=0.8, val=0.7, test=0.6),
            preds=Split(val=np.array([0.1, 0.9]), test=np.array([0.2, 0.8])),
            labels=Split(val=np.array([1, 0]), test=np.array([0, 1])),
            times=Split(train=1.0, val=0.1, test=0.2),
        )
        assert fold.fold_id == 0
        assert fold.scores.val == 0.7
        assert fold.scores.test == 0.6
        assert fold.times.train == 1.0

    def test_none_fold_id_allowed(self):
        fold = Fold(fold_id=None, scores=Split(val=0.5, test=0.4))
        assert fold.fold_id is None

    def test_immutable(self):
        fold = Fold(fold_id=0, scores=Split(val=0.5))
        with pytest.raises((AttributeError, TypeError)):
            fold.fold_id = 1


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

class TestRun:
    def _make_fold(self, fold_id, val, test, train=None, selection=None):
        return Fold(
            fold_id=fold_id,
            scores=Split(
                train=train if train is not None else val * 0.9,
                val=val,
                test=test,
                selection=selection,
            ),
        )

    def test_construction(self):
        run = Run(config={"model": "LGBM"}, iteration=0)
        assert run.config == {"model": "LGBM"}
        assert run.iteration == 0
        assert run.folds == []
        assert run.early_stopped is False # False by default
        assert run.early_stopped_makarova is None # None by default


    def test_add_fold(self):
        run = Run(iteration=0, total_folds=3)
        for i in range(3):
            run.add_fold(self._make_fold(i, 0.3, 0.4))
        assert len(run.folds) == 3

    def test_add_fold_deduplicates_by_fold_id(self):
        run = Run(iteration=0, total_folds=2)
        fold = self._make_fold(0, 0.3, 0.4)
        run.add_fold(fold)
        run.add_fold(fold)  # Same fold_id
        assert len(run.folds) == 1

    def test_add_fold_none_id_not_deduplicated(self):
        """Folds with None fold_id are always added (no dedup key)."""
        run = Run(iteration=0, total_folds=3)
        for _ in range(3):
            run.add_fold(Fold(fold_id=None, scores=Split(val=0.3, test=0.4)))
        assert len(run.folds) == 3

    def test_fully_evaluated_property(self):
        run = Run(iteration=0, total_folds=3)
        for i in range(3):
            run.add_fold(self._make_fold(i, 0.3, 0.4))
        assert run.fully_evaluated is True

    def test_not_fully_evaluated_with_fewer_folds(self):
        run = Run(iteration=0, total_folds=5)
        run.add_fold(self._make_fold(0, 0.3, 0.4))
        assert run.fully_evaluated is False

    def test_average_val_score(self):
        run = Run(iteration=0, total_folds=3)
        for i, val in enumerate([0.2, 0.4, 0.6]):
            run.add_fold(self._make_fold(i, val, val + 0.1))
        assert run.average_val_score() == approx(0.4)

    def test_average_test_score_with_retrain(self):
        run = Run(iteration=0, total_folds=2)
        run.add_fold(self._make_fold(0, 0.3, 0.4))
        run.add_fold(self._make_fold(1, 0.4, 0.5))
        retrain_fold = Fold(fold_id=None, scores=Split(train=None, val=None, test=0.35))
        run.add_retrain_evaluation(retrain_fold)

        # With retrain, test score comes from retrain fold only
        assert run.average_test_score() == approx(0.35)

    def test_average_test_score_without_retrain(self):
        run = Run(iteration=0, total_folds=2)
        run.add_fold(self._make_fold(0, 0.3, 0.4))
        run.add_fold(self._make_fold(1, 0.5, 0.6))

        assert run.average_test_score() == approx(0.5)

    def test_average_selection_score_with_fold_selection(self):
        run = Run(iteration=0, total_folds=2)
        run.add_fold(self._make_fold(0, 0.3, 0.4, selection=0.35))
        run.add_fold(self._make_fold(1, 0.4, 0.5, selection=0.45))
        sel_scores = [f.scores.selection for f in run.folds]
        avg_sel = sum(sel_scores) / len(sel_scores)
        assert avg_sel == approx(0.4)

    def test_early_stopped_default_false(self):
        run = Run(iteration=0)
        assert run.early_stopped is False

    def test_early_stopped_makarova_default_none(self):
        run = Run(iteration=0)
        assert run.early_stopped_makarova is None


# ---------------------------------------------------------------------------
# RunHistory
# ---------------------------------------------------------------------------

class TestRunHistory:
    def _make_run(self, val_scores, test_scores=None, retrain_test=None, iteration=0):
        """Create a Run with given val scores across folds."""
        if test_scores is None:
            test_scores = [v + 0.05 for v in val_scores]
        run = Run(config={"model": "dummy_model"}, iteration=iteration, total_folds=len(val_scores))
        for i, (v, t) in enumerate(zip(val_scores, test_scores)):
            fold = Fold(fold_id=i, scores=Split(train=v * 0.9, val=v, test=t))
            run.add_fold(fold)
        if retrain_test is not None:
            run.retrain = Fold(fold_id=None, scores=Split(test=retrain_test))
        return run

    def test_first_run_becomes_incumbent(self):
        history = RunHistory()
        run = self._make_run([0.3, 0.4], iteration=0)
        history.add_run(run)

        assert history.incumbent is run
        assert history.best_test_incumbent is run

    def test_better_run_replaces_incumbent(self):
        history = RunHistory()
        run1 = self._make_run([0.5, 0.5], iteration=0)  # worse
        run2 = self._make_run([0.2, 0.2], iteration=1)  # better

        history.add_run(run1)
        history.add_run(run2)

        assert history.incumbent is run2

    def test_worse_run_doesnt_replace_incumbent(self):
        history = RunHistory()
        run1 = self._make_run([0.2, 0.2], iteration=0)  # good
        run2 = self._make_run([0.5, 0.5], iteration=1)  # worse

        history.add_run(run1)
        history.add_run(run2)

        assert history.incumbent is run1

    def test_best_test_incumbent_is_monotone(self):
        """best_test_incumbent should never get worse over time."""
        history = RunHistory()
        best_test = float("inf")

        for i, (v, t) in enumerate([(0.5, 0.6), (0.3, 0.4), (0.2, 0.5), (0.1, 0.3)]):
            run = self._make_run([v], [t], iteration=i)
            history.add_run(run)
            current_best = history.best_test_incumbent.average_test_score()
            assert current_best <= best_test + 1e-9, (
                f"best_test_incumbent score increased at iteration {i}: "
                f"{current_best} > {best_test}"
            )
            best_test = min(best_test, current_best)

    def test_overtuning_is_nonnegative(self):
        history = RunHistory()
        for i, (v, t) in enumerate([(0.5, 0.4), (0.3, 0.35), (0.2, 0.5)]):
            run = self._make_run([v], [t], iteration=i)
            history.add_run(run)

        assert history.overtuning() >= 0.0

    def test_overtuning_zero_when_incumbent_is_best_on_test(self):
        history = RunHistory()
        # Each new run is also better on test
        for i, (v, t) in enumerate([(0.5, 0.5), (0.3, 0.3), (0.1, 0.1)]):
            run = self._make_run([v], [t], iteration=i)
            history.add_run(run)

        assert history.overtuning() == approx(0.0)

    def test_overtuning_positive_when_val_better_but_test_worse(self):
        history = RunHistory()
        # Run 1: good val and good test
        run1 = self._make_run([0.3], [0.3], iteration=0)
        # Run 2: better val but worse test → overtuning
        run2 = self._make_run([0.1], [0.5], iteration=1)

        history.add_run(run1)
        history.add_run(run2)

        assert history.incumbent is run2  # run2 has better val
        assert history.best_test_incumbent is run1  # run1 has better test
        assert history.overtuning() > 0

    def test_meta_overfitting(self):
        history = RunHistory()
        run = self._make_run([0.3], [0.5], iteration=0)
        history.add_run(run)

        # meta_overfitting = -(val - test) = test - val
        expected = run.average_test_score() - run.average_val_score()
        assert history.meta_overfitting() == approx(expected)

    def test_equality_operator(self):
        history1 = RunHistory()
        history2 = RunHistory()
        run1a = self._make_run([0.3], [0.4], iteration=0)
        run1b = self._make_run([0.3], [0.4], iteration=0)  # Same scores

        history1.add_run(run1a)
        history2.add_run(run1b)

        # Histories with same scores should be equal
        assert history1 == history2

    def test_inequality_operator_different_scores(self):
        history1 = RunHistory()
        history2 = RunHistory()
        history1.add_run(self._make_run([0.3], iteration=0))
        history2.add_run(self._make_run([0.4], iteration=0))

        assert history1 != history2

    def test_inequality_different_run_count(self):
        history1 = RunHistory()
        history2 = RunHistory()
        history1.add_run(self._make_run([0.3], iteration=0))
        history1.add_run(self._make_run([0.4], iteration=1))
        history2.add_run(self._make_run([0.3], iteration=0))

        assert history1 != history2

    def test_add_run_with_early_stopped(self):
        """Early-stopped runs are added but should not become incumbent."""
        history = RunHistory()
        good_run = self._make_run([0.3], iteration=0)
        history.add_run(good_run)

        early_stopped = self._make_run([0.1], iteration=1)
        early_stopped.early_stopped = True
        history.add_run(early_stopped)

        assert history.incumbent is good_run

    def test_trajectory_length_equals_run_count(self):
        history = RunHistory()
        for i in range(5):
            history.add_run(self._make_run([0.3 - i * 0.01], iteration=i))

        assert len(history.trajectory["iteration"]) == 5
        assert len(history.trajectory["val_score"]) == 5
        assert len(history.trajectory["overtuning"]) == 5

    def test_trajectory_incumbents_monotone(self):
        """Trajectory val scores should be non-increasing (incumbent only improves)."""
        history = RunHistory()
        for i, v in enumerate([0.5, 0.3, 0.4, 0.2, 0.35]):
            history.add_run(self._make_run([v], iteration=i))

        for j in range(1, len(history.trajectory["val_score"])):
            assert history.trajectory["val_score"][j] <= history.trajectory["val_score"][j - 1] + 1e-9
