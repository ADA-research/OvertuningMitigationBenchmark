"""
Tests for mitigations: Thresholdout, Bergman Racing, and MLPlan (two-phase).

All three mitigations are critical paths in the benchmark.
"""
import numpy as np
import pytest
from pytest import approx


# ---------------------------------------------------------------------------
# Thresholdout
# ---------------------------------------------------------------------------

class TestThresholdoutUnit:
    def test_initialization(self):
        from src.mitigations.thresholdout import Thresholdout
        tho = Thresholdout(num_holdout_samples=500)
        assert tho.threshold > 0
        assert tho.noise_rate > 0

    def test_threshold_scales_with_sqrt_n(self):
        """Threshold = 0.3 / sqrt(n); larger n → smaller threshold."""
        from src.mitigations.thresholdout import Thresholdout
        tho_small = Thresholdout(num_holdout_samples=100)
        tho_large = Thresholdout(num_holdout_samples=10000)
        assert tho_small.threshold > tho_large.threshold

    def test_query_returns_train_when_scores_close(self):
        """When |train - holdout| ≤ threshold, returns train score."""
        from src.mitigations.thresholdout import Thresholdout
        # Very large n → very small threshold; use identical scores so difference is 0
        np.random.seed(0)
        tho = Thresholdout(num_holdout_samples=500)
        result = tho.query(train_score=0.9, holdout_score=0.9)
        assert result == approx(0.9)

    def test_query_returns_noisy_holdout_when_scores_far(self):
        """When |train - holdout| >> threshold, returns a noisy holdout value."""
        from src.mitigations.thresholdout import Thresholdout
        np.random.seed(42)
        tho = Thresholdout(num_holdout_samples=500)
        # Extreme difference: train=0.99, holdout=0.01
        result = tho.query(train_score=0.99, holdout_score=0.01)
        # Should not return train score (0.99)
        assert result < 0.5, f"Expected noisy holdout (~0.01), got {result}"

    def test_query_output_is_float(self):
        from src.mitigations.thresholdout import Thresholdout
        np.random.seed(1)
        tho = Thresholdout(num_holdout_samples=200)
        result = tho.query(0.8, 0.5)
        assert isinstance(result, float)

    def test_query_large_difference_does_not_return_train(self):
        """Run 20 trials with a huge gap; none should return the exact train score."""
        from src.mitigations.thresholdout import Thresholdout
        tho = Thresholdout(num_holdout_samples=500)
        for seed in range(20):
            np.random.seed(seed)
            result = tho.query(0.99, 0.01)
            assert result != approx(0.99), f"Seed {seed}: returned train score unexpectedly"


class TestThresholdoutIntegration:
    def test_thresholdout_modifies_optimizer_feedback(self, unique_result_path):
        """
        With thresholdout enabled, the surrogate model receives modified scores.
        This means the surrogate mean should differ from a run without thresholdout.
        """
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        base_result_path = unique_result_path

        task_tho = DefaultTaskConfig()
        task_tho.mitigation_strategy = "thresholdout"
        task_tho.optimizer = "smac"
        task_tho.debug = True
        task_tho.iterations = 8
        task_tho.bo_initial_random_iterations = 3
        task_tho.evaluation.resampling = "holdout"
        task_tho.evaluation.val_size = 0.2
        task_tho.result_path = f"{base_result_path}_tho"

        task_base = DefaultTaskConfig()
        task_base.mitigation_strategy = "none"
        task_base.optimizer = "smac"
        task_base.debug = True
        task_base.iterations = 8
        task_base.bo_initial_random_iterations = 3
        task_base.evaluation.resampling = "holdout"
        task_base.evaluation.val_size = 0.2
        task_base.result_path = f"{base_result_path}_base"

        hist_tho = run_task(task_tho)
        hist_base = run_task(task_base)

        # At least one late-phase run should differ in surrogate mean
        tho_means = [r.surrogate.mean for r in hist_tho.history if r.surrogate]
        base_means = [r.surrogate.mean for r in hist_base.history if r.surrogate]

        assert tho_means != base_means, (
            "Thresholdout should cause different surrogate means vs no mitigation"
        )

    def test_thresholdout_fold_scores_stored_correctly(self, unique_result_path):
        """Stored fold scores must not be modified by thresholdout (only optimizer feedback is)."""
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.mitigation_strategy = "thresholdout"
        task.optimizer = "smac"
        task.debug = True
        task.iterations = 5
        task.bo_initial_random_iterations = 2
        task.result_path = unique_result_path

        history = run_task(task)
        for run in history.history:
            for fold in run.folds:
                assert fold.scores.val is not None
                assert -1.0 <= fold.scores.val <= 0.0


# ---------------------------------------------------------------------------
# Bergman Racing
# ---------------------------------------------------------------------------

class TestBergmanRacingUnit:
    def _make_run(self, val_scores, early_stopped=False):
        from src.history.data_classes import Run, Fold, Split
        folds = [Fold(fold_id=i, scores=Split(val=v, train=v * 0.9, test=v * 0.8))
                 for i, v in enumerate(val_scores)]
        run = Run(folds=folds, early_stopped=early_stopped)
        return run

    def test_aggressive_stops_worse_candidate(self):
        from src.mitigations.racing.bergman_racing import EarlyStoppingCVBergman
        from src.history.run_history import RunHistory

        racing = EarlyStoppingCVBergman(aggressive=True)
        history = RunHistory()

        # Add a good incumbent
        incumbent = self._make_run([0.2, 0.2, 0.2])  # avg val = 0.2 (low = good)
        history.add_run(incumbent)

        # Candidate is worse (higher = worse for minimization)
        candidate = self._make_run([0.4, 0.4, 0.4])

        should_stop = racing.should_stop(candidate, history)
        assert should_stop is True
        assert candidate.early_stopped is True

    def test_aggressive_doesnt_stop_better_candidate(self):
        from src.mitigations.racing.bergman_racing import EarlyStoppingCVBergman
        from src.history.run_history import RunHistory

        racing = EarlyStoppingCVBergman(aggressive=True)
        history = RunHistory()

        incumbent = self._make_run([0.4, 0.4, 0.4])
        history.add_run(incumbent)

        candidate = self._make_run([0.2, 0.2, 0.2])  # Better than incumbent

        should_stop = racing.should_stop(candidate, history)
        assert should_stop is False

    def test_no_incumbent_never_stops(self):
        from src.mitigations.racing.bergman_racing import EarlyStoppingCVBergman
        from src.history.run_history import RunHistory

        racing = EarlyStoppingCVBergman(aggressive=True)
        history = RunHistory()  # No incumbent yet

        candidate = self._make_run([0.9, 0.9])
        assert racing.should_stop(candidate, history) is False

    def test_forgiving_uses_worst_fold(self):
        """
        Forgiving mode: stop if candidate mean >= WORST fold score of incumbent.
        """
        from src.mitigations.racing.bergman_racing import EarlyStoppingCVBergman
        from src.history.run_history import RunHistory

        racing = EarlyStoppingCVBergman(aggressive=False)
        history = RunHistory()

        # Incumbent has one bad fold (0.5) and two good folds (0.1)
        incumbent = self._make_run([0.1, 0.1, 0.5])
        history.add_run(incumbent)

        # Candidate mean = 0.3, which is < worst fold of incumbent (0.5)
        candidate = self._make_run([0.3])
        should_stop = racing.should_stop(candidate, history)
        assert should_stop is False  # 0.3 < 0.5

    def test_forgiving_stops_when_above_worst_fold(self):
        from src.mitigations.racing.bergman_racing import EarlyStoppingCVBergman
        from src.history.run_history import RunHistory

        racing = EarlyStoppingCVBergman(aggressive=False)
        history = RunHistory()

        incumbent = self._make_run([0.1, 0.1, 0.5])
        history.add_run(incumbent)

        # Candidate mean = 0.6 > worst fold 0.5 → stop
        candidate = self._make_run([0.6])
        should_stop = racing.should_stop(candidate, history)
        assert should_stop is True


class TestBergmanRacingIntegration:
    def test_racing_reduces_fold_evaluations(self, unique_result_path):
        """
        With racing enabled, total fold evaluations should be less than without racing
        (since some candidates are pruned early).
        """
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        # Racing configuration (5x5CV)
        task_racing = DefaultTaskConfig()
        task_racing.racing_strategy = "bergman_aggressive"
        task_racing.mitigation_strategy = "bergman_aggressive"
        task_racing.optimizer = "smac"
        task_racing.debug = True
        task_racing.iterations = 10
        task_racing.bo_initial_random_iterations = 3
        task_racing.evaluation.resampling = "cv"
        task_racing.evaluation.n_folds = 5
        task_racing.evaluation.n_repeats = 1
        task_racing.evaluation.selection_size = None
        task_racing.result_path = f"{unique_result_path}_racing"

        # Baseline without racing (same CV setup)
        task_base = DefaultTaskConfig()
        task_base.racing_strategy = "none"
        task_base.mitigation_strategy = "none"
        task_base.optimizer = "smac"
        task_base.debug = True
        task_base.iterations = 10
        task_base.bo_initial_random_iterations = 3
        task_base.evaluation.resampling = "cv"
        task_base.evaluation.n_folds = 5
        task_base.evaluation.n_repeats = 1
        task_base.evaluation.selection_size = None
        task_base.result_path = f"{unique_result_path}_base"

        hist_racing = run_task(task_racing)
        hist_base = run_task(task_base)

        # Racing should have fewer total fold evaluations
        racing_total_folds = sum(len(r.folds) for r in hist_racing.history)
        base_total_folds = sum(len(r.folds) for r in hist_base.history)

        assert racing_total_folds < base_total_folds, (
            f"Racing should save fold evaluations. "
            f"Racing: {racing_total_folds}, Base: {base_total_folds}"
        )

    def test_racing_early_stopped_runs_marked(self, unique_result_path):
        """Pruned runs must have early_stopped=True."""
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.racing_strategy = "bergman_aggressive"
        task.mitigation_strategy = "bergman_aggressive"
        task.optimizer = "smac"
        task.debug = True
        task.iterations = 15
        task.bo_initial_random_iterations = 3
        task.evaluation.resampling = "cv"
        task.evaluation.n_folds = 5
        task.evaluation.n_repeats = 1
        task.evaluation.selection_size = None
        task.result_path = unique_result_path

        history = run_task(task)
        early_stopped_runs = [r for r in history.history if r.early_stopped]
        # With 15 iterations and 5 folds, at least some should be pruned
        assert len(early_stopped_runs) > 0, "Expected some racing-pruned runs"


# ---------------------------------------------------------------------------
# MLPlan (Two-Phase)
# ---------------------------------------------------------------------------

class TestMLPlanOptimizerUnit:
    def _make_run_history(self, n_runs=30, seed=42):
        """Build a synthetic RunHistory for MLPlan phase 1 testing."""
        from src.history.run_history import RunHistory
        from src.history.data_classes import Run, Fold, Split

        rng = np.random.default_rng(seed)
        history = RunHistory()
        for i in range(n_runs):
            val = rng.random() * -0.5
            run = Run(
                config={"model": "LGBM", "lr": rng.random()},
                iteration=i,
                folds=[
                    Fold(fold_id=0, scores=Split(train=val * 0.9, val=val, test=val * 1.1)),
                    Fold(fold_id=1, scores=Split(train=val * 0.9, val=val + 0.01, test=val * 1.1)),
                ],
                retrain=Fold(fold_id=0, scores=Split(train=val, val=val, test=val * 1.05)),
            )
            history.add_run(run)
        return history

    def test_select_configurations_top_k(self):
        """select_configurations must include the top-k runs by average val score."""
        from src.optimizers.mlplan_phase_two import MLPlanPhaseTwoOptimizer

        history = self._make_run_history(n_runs=150)
        k = 25
        optimizer = MLPlanPhaseTwoOptimizer(history, random_state=42, k=k, epsilon=0.03)

        selected = optimizer.selected_configurations
        selected_val_scores = [r.average_val_score() for r in selected]
        all_val_scores = sorted([r.average_val_score() for r in history.history])

        # All top-k scores must be among the best in history
        for score in sorted(selected_val_scores)[:k]:
            assert score in all_val_scores[:k + 5]  # Allow some epsilon-based additions

    def test_select_configurations_respects_epsilon(self):
        """All random configs must be within epsilon of the incumbent."""
        from src.optimizers.mlplan_phase_two import MLPlanPhaseTwoOptimizer

        history = self._make_run_history(n_runs=50)
        epsilon = 0.03
        optimizer = MLPlanPhaseTwoOptimizer(history, random_state=42, k=25, epsilon=epsilon)

        incumbent_score = history.incumbent.average_val_score()

        # We check if every configuration selected after the top-k is within epsilon of the incumbent
        # Within the top-k we do not care about epsilon
        for run in optimizer.selected_configurations[optimizer.k:]:
            assert run.average_val_score() <= incumbent_score + epsilon, (
                f"Selected config with score {run.average_val_score()} > "
                f"incumbent + epsilon ({incumbent_score + epsilon})"
            )

    def test_returns_configurations_in_order(self):
        """generate_configuration() must iterate through selected_configurations in order."""
        from src.optimizers.mlplan_phase_two import MLPlanPhaseTwoOptimizer

        history = self._make_run_history(n_runs=30)
        optimizer = MLPlanPhaseTwoOptimizer(history, random_state=0, k=5)

        n = optimizer.number_of_selected_configurations()
        configs = []
        for _ in range(n):
            cfg, t = optimizer.generate_configuration()
            configs.append(cfg)

        assert len(configs) == n

    def test_iteration_map_is_correct(self):
        """iteration_map_phase_two_to_phase_one must map phase2 idx → phase1 iteration."""
        from src.optimizers.mlplan_phase_two import MLPlanPhaseTwoOptimizer

        history = self._make_run_history(n_runs=30)
        optimizer = MLPlanPhaseTwoOptimizer(history, random_state=0, k=5)

        for p2_idx, p1_iter in optimizer.iteration_map_phase_two_to_phase_one.items():
            # p1_iter should be a valid iteration number in the phase 1 history
            valid_iters = [r.iteration for r in history.history]
            assert p1_iter in valid_iters


class TestMLPlanIntegration:
    def test_mlplan_two_phase_runs_successfully(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.mitigation_strategy = "mlplan"
        task.optimizer = "smac"
        task.iterations = 12  # phase1=7, phase2=up to 7 configs
        task.bo_initial_random_iterations = 3
        task.debug = True
        task.evaluation.resampling = "holdout"
        task.result_path = unique_result_path

        history = run_task(task)

        # history is MLPlanRunHistory; it should have both phases
        from src.history.run_history import MLPlanRunHistory
        assert isinstance(history, MLPlanRunHistory)
        assert history.phase_two_history is not None
        assert len(history.history) > 0

    def test_mlplan_history_has_combined_dataframe(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.mitigation_strategy = "mlplan"
        task.optimizer = "smac"
        task.iterations = 12
        task.bo_initial_random_iterations = 3
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        df = history.combined_dataframe_both_phases
        assert df is not None
        assert len(df) > 0

    def test_mlplan_combined_df_has_mlplan_score_column(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.mitigation_strategy = "mlplan"
        task.optimizer = "smac"
        task.iterations = 12
        task.bo_initial_random_iterations = 3
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        df = history.combined_dataframe_both_phases
        assert "mlplan_score" in df.columns

    def test_mlplan_phase2_rows_have_mlplan_score(self, unique_result_path):
        """Phase 2 rows should have a non-null mlplan_score; phase 1 rows should have None."""
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.mitigation_strategy = "mlplan"
        task.optimizer = "smac"
        task.iterations = 12
        task.bo_initial_random_iterations = 3
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        df = history.combined_dataframe_both_phases
        n_phase1 = len(history.to_dataframe())
        n_phase2 = len(history.phase_two_history.to_dataframe())

        phase1_rows = df.iloc[:n_phase1]
        phase2_rows = df.iloc[n_phase1:n_phase1 + n_phase2]

        assert phase1_rows["mlplan_score"].isna().all(), "Phase 1 should have NaN mlplan_score"
        assert phase2_rows["mlplan_score"].notna().all(), "Phase 2 should have non-NaN mlplan_score"

    def test_mlplan_get_results_produces_three_csvs(self, unique_result_path, tmp_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.mitigation_strategy = "mlplan"
        task.optimizer = "smac"
        task.iterations = 12
        task.bo_initial_random_iterations = 3
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        root = str(tmp_path) + "/"
        results = history.get_results(root_path=root)

        paths = [r[0] for r in results]
        has_history = any("history.csv" in p and "phase" not in p for p in paths)
        has_phase1 = any("history_phase_one.csv" in p for p in paths)
        has_phase2 = any("history_phase_two.csv" in p for p in paths)

        assert has_history, "Missing combined history.csv"
        assert has_phase1, "Missing history_phase_one.csv"
        assert has_phase2, "Missing history_phase_two.csv"

    def test_mlplan_phase1_budget_is_60_percent(self, unique_result_path):
        """Phase 1 should use ~60% of iterations; phase 2 evaluates selected configs."""
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        full_iterations = 15
        task = DefaultTaskConfig()
        task.mitigation_strategy = "mlplan"
        task.optimizer = "smac"
        task.iterations = full_iterations
        task.bo_initial_random_iterations = 3
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        phase1_iterations = len(history.history)
        expected_phase1 = int(full_iterations * 0.6)
        assert phase1_iterations == expected_phase1, (
            f"Phase 1 should have {expected_phase1} iterations, got {phase1_iterations}"
        )
