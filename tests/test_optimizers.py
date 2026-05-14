"""
Tests for SMAC and HEBO optimizers.

Both optimizers are used in the benchmark (50% each).
Tests cover:
- Configuration generation and tell cycle
- Surrogate predictions after initial iterations
- Makarova early stopping computation (math only)
- Integration with the full run_task pipeline
"""
import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float, CategoricalHyperparameter
from pytest import approx

from src.optimizers.bo_smac import SMACOptimizer
from src.optimizers.bo_hebo import HEBOOptimizer


def make_simple_cs(seed=42):
    cs = ConfigurationSpace(seed=seed)
    cs.add(Float("x1", bounds=(0.0, 1.0)))
    cs.add(Float("x2", bounds=(0.0, 1.0)))
    return cs


# ---------------------------------------------------------------------------
# SMAC Optimizer
# ---------------------------------------------------------------------------

class TestSMACOptimizerBasic:
    def test_generates_configuration(self, tmp_path):
        cs = make_simple_cs()
        optimizer = SMACOptimizer(
            search_space=cs,
            initial_iterations=2,
            surrogate_model="gaussian_process",
            random_state=42,
            output_directory=str(tmp_path / "smac_basic"),
        )
        config, t = optimizer.generate_configuration()
        assert config is not None
        assert t >= 0.0

    def test_tell_updates_model(self, tmp_path):
        cs = make_simple_cs()
        optimizer = SMACOptimizer(
            search_space=cs,
            initial_iterations=2,
            surrogate_model="gaussian_process",
            random_state=42,
            output_directory=str(tmp_path / "smac_tell"),
        )
        for i in range(4):
            config, _ = optimizer.generate_configuration()
            optimizer.tell(-0.5 + i * 0.1)  # negative scores (minimization)

    def test_surrogate_none_before_initial_iters_done(self, tmp_path):
        cs = make_simple_cs()
        optimizer = SMACOptimizer(
            search_space=cs,
            initial_iterations=5,
            surrogate_model="gaussian_process",
            random_state=42,
            output_directory=str(tmp_path / "smac_surrogate"),
        )
        # Initial iterations: surrogate not trained yet
        config, _ = optimizer.generate_configuration()
        optimizer.tell(-0.5)
        # Only 1 iteration done; surrogate not available in initial phase
        surrogate = optimizer.get_surrogate_predictions(config)
        assert surrogate is None

    def test_surrogate_available_after_initial_iters(self, tmp_path):
        cs = make_simple_cs()
        n_initial = 3
        optimizer = SMACOptimizer(
            search_space=cs,
            initial_iterations=n_initial,
            surrogate_model="gaussian_process",
            random_state=42,
            output_directory=str(tmp_path / "smac_surrogate2"),
        )
        configs = []
        for i in range(n_initial + 2):
            config, _ = optimizer.generate_configuration()
            configs.append(config)
            optimizer.tell(-0.5 + i * 0.05)

        # After initial iterations, surrogate should be available
        surrogate = optimizer.get_surrogate_predictions(configs[-1])
        assert surrogate is not None
        assert surrogate.mean is not None
        assert surrogate.std is not None

    def test_hp_count_correct(self, tmp_path):
        cs = make_simple_cs()
        optimizer = SMACOptimizer(
            search_space=cs,
            initial_iterations=2,
            surrogate_model="gaussian_process",
            random_state=42,
            output_directory=str(tmp_path / "smac_hp"),
        )
        # 2 Float hyperparameters → count should be 2
        assert optimizer.count_number_of_hyperparameters() == 2

    def test_reproducibility_same_seed(self, tmp_path):
        cs = make_simple_cs(seed=99)

        def run_smac(run_id):
            opt = SMACOptimizer(
                search_space=cs,
                initial_iterations=2,
                surrogate_model="gaussian_process",
                random_state=42,
                output_directory=str(tmp_path / f"smac_repro_{run_id}"),
            )
            configs = []
            for i in range(3):
                cfg, _ = opt.generate_configuration()
                configs.append(dict(cfg))
                opt.tell(-0.5)
            return configs

        c1 = run_smac("a")
        c2 = run_smac("b")
        assert c1 == c2


class TestSMACMakarova:
    def test_nadeau_bengio_threshold_basic(self):
        scores = [0.8, 0.82, 0.78, 0.81, 0.79]
        threshold = SMACOptimizer._calculate_nadeau_bengio_threshold(scores)
        assert threshold > 0
        # Manual check
        vals = np.array(scores)
        expected = np.sqrt(np.var(vals)) * np.sqrt(1 / 5 + 1 / 4)
        assert threshold == approx(expected)

    def test_nadeau_bengio_empty_returns_zero(self):
        assert SMACOptimizer._calculate_nadeau_bengio_threshold([]) == 0.0

    def test_nadeau_bengio_single_returns_zero(self):
        assert SMACOptimizer._calculate_nadeau_bengio_threshold([0.8]) == 0.0

    def test_nadeau_bengio_none_returns_zero(self):
        assert SMACOptimizer._calculate_nadeau_bengio_threshold(None) == 0.0

    @pytest.mark.parametrize("scores", [
        [0.8, 0.9],
        [0.7, 0.8, 0.9],
        [0.6, 0.65, 0.7, 0.75, 0.8],
    ])
    def test_nadeau_bengio_positive_for_varying_scores(self, scores):
        threshold = SMACOptimizer._calculate_nadeau_bengio_threshold(scores)
        assert threshold > 0

    def test_identical_scores_give_zero_threshold(self):
        scores = [0.8, 0.8, 0.8, 0.8, 0.8]
        assert SMACOptimizer._calculate_nadeau_bengio_threshold(scores) == approx(0.0)

    def test_beta_t_zero_returns_one(self):
        cs = make_simple_cs()
        opt = SMACOptimizer.__new__(SMACOptimizer)
        opt.search_space = cs
        result = SMACOptimizer._compute_beta_t(opt, 0.1, 0.2, 0)
        assert result == 1.0

    def test_beta_t_increases_with_t(self, tmp_path):
        cs = make_simple_cs()
        opt = SMACOptimizer(
            search_space=cs,
            initial_iterations=2,
            surrogate_model="gaussian_process",
            random_state=42,
            output_directory=str(tmp_path / "smac_betat"),
        )
        beta_10 = opt._compute_beta_t(0.1, 0.2, 10)
        beta_50 = opt._compute_beta_t(0.1, 0.2, 50)
        beta_100 = opt._compute_beta_t(0.1, 0.2, 100)
        assert beta_50 > beta_10
        assert beta_100 > beta_50

    def test_makarova_insufficient_data_returns_false(self, tmp_path):
        cs = make_simple_cs()
        opt = SMACOptimizer(
            search_space=cs,
            initial_iterations=5,
            surrogate_model="gaussian_process",
            random_state=42,
            output_directory=str(tmp_path / "smac_mak"),
        )
        for i in range(4):
            cfg, _ = opt.generate_configuration()
            opt.tell(-0.5)

        result = opt.early_stopping_makarova_triggered([0.8] * 5)
        assert result is False


class TestSMACIntegration:
    def test_framework_run_smac(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.optimizer = "smac"
        task.iterations = 4
        task.bo_initial_random_iterations = 2
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        assert len(history.history) == 4
        # First two runs (initial) have no surrogate; last two do
        for run in history.history[:2]:
            assert run.surrogate is None
        for run in history.history[2:]:
            assert run.surrogate is not None
            assert run.surrogate.mean is not None

    def test_framework_run_smac_cv(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.optimizer = "smac"
        task.iterations = 3
        task.bo_initial_random_iterations = 2
        task.evaluation.resampling = "cv"
        task.evaluation.n_folds = 3
        task.evaluation.n_repeats = 1
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        assert len(history.history) == 3
        for run in history.history:
            assert len(run.folds) == 3


# ---------------------------------------------------------------------------
# HEBO Optimizer
# ---------------------------------------------------------------------------

class TestHEBOOptimizerBasic:
    def test_generates_configuration(self):
        cs = make_simple_cs()
        optimizer = HEBOOptimizer(search_space=cs, initial_iterations=2, random_state=42)
        config, t = optimizer.generate_configuration()
        assert config is not None
        assert t >= 0.0

    def test_tell_and_generate_cycle(self):
        cs = make_simple_cs()
        optimizer = HEBOOptimizer(search_space=cs, initial_iterations=2, random_state=42)
        for i in range(5):
            config, _ = optimizer.generate_configuration()
            optimizer.tell(-float(i) * 0.1)

    def test_reproducibility_same_seed(self):
        cs = make_simple_cs(seed=99)

        def run_hebo():
            opt = HEBOOptimizer(search_space=cs, initial_iterations=3, random_state=42)
            configs = []
            for i in range(10):
                cfg, _ = opt.generate_configuration()
                configs.append(dict(cfg))
                opt.tell(-0.5)
            return configs

        c1 = run_hebo()
        c2 = run_hebo()
        c3 = run_hebo()
        assert c1 == c2 == c3 

    def test_none_categorical_values_do_not_crash_transform(self):
        cs = ConfigurationSpace(seed=42)
        cs.add_hyperparameter(
            CategoricalHyperparameter("cat1", choices=[None, "pbld", "pl", "plr"])
        )

        optimizer = HEBOOptimizer(search_space=cs, initial_iterations=2, random_state=42)
        sampled_df = optimizer.design_space.sample(32)

        # HEBO can emit NaN for None-valued categories; this must be normalized.
        normalized_df = optimizer._normalize_nullable_categorical_nans(sampled_df)
        _, xe = optimizer.design_space.transform(normalized_df)

        assert xe.shape[0] == 32

        cfg = optimizer._hebo_df_to_configuration(sampled_df.iloc[[0]])
        assert cfg["cat1"] in [None, "pbld", "pl", "plr"]


class TestHEBOMakarova:
    """HEBO shares the same Makarova implementation as SMAC."""

    def test_nadeau_bengio_threshold_basic(self):
        scores = [0.8, 0.82, 0.78, 0.81, 0.79]
        threshold = HEBOOptimizer._calculate_nadeau_bengio_threshold(scores)
        assert threshold > 0

    def test_nadeau_bengio_empty_returns_zero(self):
        assert HEBOOptimizer._calculate_nadeau_bengio_threshold([]) == 0.0

    def test_nadeau_bengio_none_returns_zero(self):
        assert HEBOOptimizer._calculate_nadeau_bengio_threshold(None) == 0.0

    def test_beta_t_zero_returns_one(self):
        cs = make_simple_cs()
        opt = HEBOOptimizer(search_space=cs, initial_iterations=2, random_state=42)
        assert opt._compute_beta_t(0.1, 0.2, 0) == 1.0

    def test_beta_t_increases_with_t(self):
        cs = make_simple_cs()
        opt = HEBOOptimizer(search_space=cs, initial_iterations=2, random_state=42)
        assert opt._compute_beta_t(0.1, 0.2, 50) > opt._compute_beta_t(0.1, 0.2, 10)
        assert opt._compute_beta_t(0.1, 0.2, 100) > opt._compute_beta_t(0.1, 0.2, 50)

    def test_makarova_insufficient_data_returns_false(self):
        cs = make_simple_cs()
        opt = HEBOOptimizer(search_space=cs, initial_iterations=5, random_state=42)
        for i in range(4):
            opt.generate_configuration()
            opt.tell(-0.5)

        result = opt.early_stopping_makarova_triggered([0.8] * 5)
        assert result is False  # Too few iterations


class TestHEBOIntegration:
    def test_framework_run_hebo(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.optimizer = "hebo"
        task.iterations = 4
        task.bo_initial_random_iterations = 2
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        assert len(history.history) == 4
        # All runs should have valid val scores
        for run in history.history:
            assert len(run.folds) >= 1
            for fold in run.folds:
                assert fold.scores.val is not None

    def test_framework_run_hebo_cv(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.optimizer = "hebo"
        task.iterations = 4
        task.bo_initial_random_iterations = 2
        task.evaluation.resampling = "cv"
        task.evaluation.n_folds = 5
        task.evaluation.n_repeats = 1
        task.metric = "roc_auc"
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        assert len(history.history) == 4
        for run in history.history:
            assert len(run.folds) == 5

    def test_hebo_roc_auc_with_cv(self, unique_result_path):
        """HEBO + roc_auc + 5CV matches benchmark branch."""
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.optimizer = "hebo"
        task.metric = "roc_auc"
        task.iterations = 3
        task.bo_initial_random_iterations = 2
        task.evaluation.resampling = "cv"
        task.evaluation.n_folds = 5
        task.evaluation.n_repeats = 1
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        for run in history.history:
            for fold in run.folds:
                assert -1.0 <= fold.scores.val <= 0.0
