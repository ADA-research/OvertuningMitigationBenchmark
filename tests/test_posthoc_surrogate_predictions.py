from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from src.search_space.search_space import SearchSpace
from src.visualizations.preprocessing.post_hoc_surrogate_preprocessing import (
    REQUIRED_TRAJECTORY_COLUMNS,
    cast_value_to_hyperparameter,
    compute_post_hoc_surrogate_incumbents_hebo,
    compute_post_hoc_surrogate_incumbents_smac,
    compute_post_hoc_surrogate_trajectories_for_run,
    configurations_to_hebo_dataframe,
    load_iteration_configs,
    preprocess_post_hoc_surrogate_experiment,
    reconstruct_configuration,
    reconstruct_configurations_for_iterations,
)


def _benchmark_task_config(optimizer: str = "smac") -> dict:
    return {
        "task_id": 0,
        "random_state": 78,
        "problem_type": "regression",
        "dataset_id": "363625",
        "debug": False,
        "optimizer": optimizer,
        "metric": "neg_root_mean_squared_error",
        "offline": False,
        "offline_data_loading": True,
        "store_results_google_cloud": False,
        "store_vectors_google_cloud": False,
        "outer_evaluation": {
            "resampling": "cv",
            "train_size": 0,
            "n_folds": 3,
            "n_repeats": 1,
            "fold": 0,
            "repeat": 0,
        },
        "evaluation": {
            "resampling": "holdout",
            "val_size": 0.2,
            "selection_size": 0.0,
            "reshuffle": False,
            "retrain": True,
            "n_folds": 1,
            "n_repeats": 1,
        },
        "search_space": {
            "classifiers": ["RealMLP"],
            "regressors": ["RealMLP"],
            "preprocessors": {
                "scalers": ["StandardScaler"],
                "dim_reducers": ["None"],
                "imputers": ["NumericalSimpleImputer"],
                "encoders": ["OneHotEncoder"],
                "feat_selectors": ["None"],
            },
        },
        "max_iterations": 250,
        "bo_initial_random_iterations": 2,
        "smac_surrogate_model": "gaussian_process",
        "smac_surrogate_random_forest_n_trees": 10,
        "mitigation_strategy": "none",
        "racing_strategy": "none",
        "result_path": "unit_test_run",
    }


def _write_run(
    experiment_dir: Path,
    run_name: str,
    optimizer: str = "smac",
    benchmark: bool = True,
) -> Path:
    run_dir = experiment_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    task_config = _benchmark_task_config(optimizer=optimizer)
    if not benchmark:
        task_config["mitigation_strategy"] = "thresholdout"
        task_config["evaluation"]["selection_size"] = None
    task_config["result_path"] = run_name

    with open(run_dir / "task_config.yaml", "w") as handle:
        yaml.safe_dump(task_config, handle)

    history_df = pd.DataFrame(
        {
            "iteration": [0, 1],
            "fold": [0, 0],
            "test": [0.6, 0.3],
            "avg_val_score": [0.4, 0.1],
            "avg_test_score": [0.5, 0.2],
        }
    )
    history_df.to_csv(run_dir / "history.csv", index=False)

    space = SearchSpace(task_config).get_space()
    cfg0 = space.sample_configuration()
    cfg1 = space.sample_configuration()

    configs_payload = {
        "iter_0": {str(key): str(value) for key, value in dict(cfg0).items()},
        "iter_1": {str(key): str(value) for key, value in dict(cfg1).items()},
    }
    with open(run_dir / "configs.yaml", "w") as handle:
        yaml.safe_dump(configs_payload, handle, sort_keys=False)

    np.savez_compressed(run_dir / "artifacts.npz", test_labels=np.array([0.0, 1.0]))

    return run_dir


def test_cast_value_to_hyperparameter_and_reconstruct_configuration():
    space = ConfigurationSpace(seed=0)
    hp_cat = CategoricalHyperparameter("cat", choices=["a", None, True])
    hp_int = UniformIntegerHyperparameter("int_hp", lower=1, upper=10)
    hp_float = UniformFloatHyperparameter("float_hp", lower=0.0, upper=1.0)
    hp_const = Constant("const_hp", value=78)
    space.add([hp_cat, hp_int, hp_float, hp_const])

    assert cast_value_to_hyperparameter("None", hp_cat) is None
    assert cast_value_to_hyperparameter("True", hp_cat) is True
    assert cast_value_to_hyperparameter("7", hp_int) == 7
    assert cast_value_to_hyperparameter("0.125", hp_float) == pytest.approx(0.125)
    assert cast_value_to_hyperparameter("78", hp_const) == 78

    cfg = reconstruct_configuration(
        space,
        {
            "cat": "None",
            "int_hp": "3",
            "float_hp": "0.5",
            "const_hp": "78",
        },
    )

    assert cfg["cat"] is None
    assert cfg["int_hp"] == 3
    assert cfg["float_hp"] == pytest.approx(0.5)
    assert cfg["const_hp"] == 78


def test_load_iteration_configs_and_missing_iteration(tmp_path):
    configs_path = tmp_path / "configs.yaml"
    with open(configs_path, "w") as handle:
        yaml.safe_dump({"iter_0": {"x": "1"}, "iter_2": {"x": "3"}}, handle)

    parsed = load_iteration_configs(configs_path)
    assert sorted(parsed.keys()) == [0, 2]

    space = ConfigurationSpace(seed=0)
    space.add(UniformIntegerHyperparameter("x", lower=0, upper=10))

    with pytest.raises(KeyError):
        reconstruct_configurations_for_iterations(space, parsed, [0, 1, 2])


def test_configurations_to_hebo_dataframe_uses_design_space_columns():
    space = ConfigurationSpace(seed=0)
    space.add(CategoricalHyperparameter("model", choices=["A", "B"]))
    cfg0 = Configuration(space, values={"model": "A"})
    cfg1 = Configuration(space, values={"model": "B"})

    class DesignSpaceStub:
        paras = {"model": object(), "unused": object()}

    df = configurations_to_hebo_dataframe([cfg0, cfg1], DesignSpaceStub())
    assert list(df.columns) == ["model", "unused"]
    assert df.loc[0, "model"] == "A"
    assert pd.isna(df.loc[0, "unused"])


def test_compute_post_hoc_surrogate_incumbents_smac_with_stub(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.y_obs = None

        def train(self, X, y):
            self.y_obs = np.asarray(y, dtype=float).reshape(-1)

        def predict(self, X):
            X = np.asarray(X)
            means = np.array([[self.y_obs[i] + 0.1 * i] for i in range(len(X))], dtype=float)
            vars_ = np.ones_like(means)
            return means, vars_

    class DummySMACBackend:
        def __init__(self):
            self._model = DummyModel()

    class DummySMACOptimizer:
        def __init__(self, **kwargs):
            self.smac = DummySMACBackend()

    monkeypatch.setattr(
        "src.visualizations.preprocessing.post_hoc_surrogate_preprocessing.SMACOptimizer",
        DummySMACOptimizer,
    )

    space = ConfigurationSpace(seed=0)
    space.add(UniformFloatHyperparameter("x", lower=0.0, upper=1.0))
    configs = [
        Configuration(space, values={"x": 0.1}),
        Configuration(space, values={"x": 0.2}),
        Configuration(space, values={"x": 0.3}),
    ]

    incumbent_indices, incumbent_means = compute_post_hoc_surrogate_incumbents_smac(
        configurations=configs,
        val_scores=[0.9, 0.4, 0.2],
        search_space=space,
        task_config={"bo_initial_random_iterations": 2, "smac_surrogate_model": "gaussian_process"},
        show_progress=False,
        desc="test",
    )

    assert incumbent_indices == [0, 1, 2]
    assert len(incumbent_means) == 3


def test_compute_post_hoc_surrogate_incumbents_hebo_with_stub(monkeypatch):
    class DummyDesignSpace:
        paras = {"x": object()}

        def transform(self, df):
            values = df[["x"]].to_numpy(dtype=float)
            return values, np.zeros((len(values), 0))

    class DummyModel:
        def fit(self, X, Xe, y):
            self.y = np.asarray(y, dtype=float).reshape(-1)

    class DummyHEBOOptimizer:
        def __init__(self, **kwargs):
            self.design_space = DummyDesignSpace()

        def get_hebo_surrogate_model(self):
            return DummyModel()

    class DummyMean:
        def __init__(self, model):
            self.model = model

        def __call__(self, X, Xe):
            return self.model.y[: len(X)].reshape(-1, 1)

    monkeypatch.setattr(
        "src.visualizations.preprocessing.post_hoc_surrogate_preprocessing.HEBOOptimizer",
        DummyHEBOOptimizer,
    )
    monkeypatch.setattr(
        "src.visualizations.preprocessing.post_hoc_surrogate_preprocessing.Mean",
        DummyMean,
    )

    space = ConfigurationSpace(seed=0)
    space.add(UniformFloatHyperparameter("x", lower=0.0, upper=1.0))
    configs = [
        Configuration(space, values={"x": 0.1}),
        Configuration(space, values={"x": 0.2}),
        Configuration(space, values={"x": 0.3}),
    ]

    incumbent_indices, incumbent_means = compute_post_hoc_surrogate_incumbents_hebo(
        configurations=configs,
        val_scores=[0.6, 0.4, 0.2],
        search_space=space,
        task_config={"bo_initial_random_iterations": 2, "random_state": 78},
        show_progress=False,
        desc="test",
    )

    assert incumbent_indices == [0, 1, 2]
    assert len(incumbent_means) == 3


def test_post_hoc_surrogate_warmup_skips_model_selection_before_threshold(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.train_calls = 0
            self.y_obs = None

        def train(self, X, y):
            self.train_calls += 1
            self.y_obs = np.asarray(y, dtype=float).reshape(-1)

        def predict(self, X):
            means = np.asarray(self.y_obs, dtype=float).reshape(-1, 1)
            vars_ = np.ones_like(means)
            return means, vars_

    class DummySMACBackend:
        def __init__(self):
            self._model = DummyModel()

    class DummySMACOptimizer:
        instances = []

        def __init__(self, **kwargs):
            self.smac = DummySMACBackend()
            DummySMACOptimizer.instances.append(self)

    class DummyDesignSpace:
        paras = {"x": object()}

        def transform(self, df):
            return df[["x"]].to_numpy(dtype=float), np.zeros((len(df), 0))

    class DummyHEBOModel:
        def __init__(self):
            self.fit_calls = 0
            self.y = None

        def fit(self, X, Xe, y):
            self.fit_calls += 1
            self.y = np.asarray(y, dtype=float).reshape(-1)

    class DummyHEBOOptimizer:
        instances = []

        def __init__(self, **kwargs):
            self.design_space = DummyDesignSpace()
            self.model = DummyHEBOModel()
            DummyHEBOOptimizer.instances.append(self)

        def get_hebo_surrogate_model(self):
            return self.model

    class DummyMean:
        def __init__(self, model):
            self.model = model

        def __call__(self, X, Xe):
            return self.model.y[: len(X)].reshape(-1, 1)

    monkeypatch.setattr(
        "src.visualizations.preprocessing.post_hoc_surrogate_preprocessing.SMACOptimizer",
        DummySMACOptimizer,
    )
    monkeypatch.setattr(
        "src.visualizations.preprocessing.post_hoc_surrogate_preprocessing.HEBOOptimizer",
        DummyHEBOOptimizer,
    )
    monkeypatch.setattr(
        "src.visualizations.preprocessing.post_hoc_surrogate_preprocessing.Mean",
        DummyMean,
    )

    space = ConfigurationSpace(seed=0)
    space.add(UniformFloatHyperparameter("x", lower=0.0, upper=1.0))
    configs = [
        Configuration(space, values={"x": 0.1}),
        Configuration(space, values={"x": 0.2}),
        Configuration(space, values={"x": 0.3}),
        Configuration(space, values={"x": 0.4}),
    ]
    val_scores = [0.6, 0.2, 0.4, 0.1]
    task_config = {"bo_initial_random_iterations": 3, "random_state": 78}

    smac_indices, _ = compute_post_hoc_surrogate_incumbents_smac(
        configurations=configs,
        val_scores=val_scores,
        search_space=space,
        task_config=task_config,
        show_progress=False,
        desc="test",
    )
    hebo_indices, _ = compute_post_hoc_surrogate_incumbents_hebo(
        configurations=configs,
        val_scores=val_scores,
        search_space=space,
        task_config=task_config,
        show_progress=False,
        desc="test",
    )

    assert smac_indices[:3] == [0, 1, 1]
    assert hebo_indices[:3] == [0, 1, 1]
    assert DummySMACOptimizer.instances[0].smac._model.train_calls == 1
    assert DummyHEBOOptimizer.instances[0].model.fit_calls == 1


def test_compute_post_hoc_surrogate_trajectories_for_run_dispatch_and_skips(tmp_path, monkeypatch):
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir()

    non_benchmark = _write_run(experiment_dir, "run_non_benchmark", optimizer="smac", benchmark=False)
    result_non_benchmark = compute_post_hoc_surrogate_trajectories_for_run(
        non_benchmark,
        experiment_name="exp",
        show_progress=False,
    )
    assert result_non_benchmark["status"] == "SKIPPED"

    random_search = _write_run(experiment_dir, "run_random", optimizer="random_search", benchmark=True)
    result_random = compute_post_hoc_surrogate_trajectories_for_run(
        random_search,
        experiment_name="exp",
        show_progress=False,
    )
    assert result_random["status"] == "SKIPPED"

    called = {"smac": False, "hebo": False}

    def _fake_smac(**kwargs):
        called["smac"] = True
        return [0, 1], [0.4, 0.1]

    def _fake_hebo(**kwargs):
        called["hebo"] = True
        return [0, 1], [0.4, 0.1]

    monkeypatch.setattr(
        "src.visualizations.preprocessing.post_hoc_surrogate_preprocessing.compute_post_hoc_surrogate_incumbents_smac",
        _fake_smac,
    )
    monkeypatch.setattr(
        "src.visualizations.preprocessing.post_hoc_surrogate_preprocessing.compute_post_hoc_surrogate_incumbents_hebo",
        _fake_hebo,
    )

    run_smac = _write_run(experiment_dir, "run_smac", optimizer="smac", benchmark=True)
    result_smac = compute_post_hoc_surrogate_trajectories_for_run(
        run_smac,
        experiment_name="exp",
        show_progress=False,
    )
    assert result_smac["status"] == "SUCCESS"
    assert called["smac"] is True
    assert called["hebo"] is False

    called = {"smac": False, "hebo": False}
    run_hebo = _write_run(experiment_dir, "run_hebo", optimizer="hebo", benchmark=True)
    result_hebo = compute_post_hoc_surrogate_trajectories_for_run(
        run_hebo,
        experiment_name="exp",
        show_progress=False,
    )
    assert result_hebo["status"] == "SUCCESS"
    assert called["smac"] is False
    assert called["hebo"] is True


def test_preprocess_post_hoc_surrogate_experiment_end_to_end(tmp_path, monkeypatch):
    experiment_dir = tmp_path / "exp_posthoc_surrogate"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    _write_run(experiment_dir, "run_benchmark_smac", optimizer="smac", benchmark=True)
    _write_run(experiment_dir, "run_non_benchmark", optimizer="smac", benchmark=False)

    monkeypatch.setattr(
        "src.visualizations.preprocessing.post_hoc_surrogate_preprocessing.compute_post_hoc_surrogate_incumbents_smac",
        lambda **kwargs: ([0, 1], [0.4, 0.1]),
    )

    trajectories_df = preprocess_post_hoc_surrogate_experiment(
        results_dir=str(experiment_dir),
        n_jobs=1,
        verbose=False,
    )

    assert trajectories_df is not None
    assert list(trajectories_df.columns) == REQUIRED_TRAJECTORY_COLUMNS
    assert trajectories_df["raw_experiment_id"].nunique() == 1
    assert trajectories_df["raw_experiment_id"].iloc[0] == "run_benchmark_smac"

    rows = trajectories_df.sort_values("iteration").reset_index(drop=True)
    assert rows.loc[0, "val_performance"] == pytest.approx(0.4)
    assert rows.loc[1, "val_performance"] == pytest.approx(0.1)

    assert (experiment_dir / "results_post_hoc_surrogate.csv").exists()
    assert (experiment_dir / "trajectories_post_hoc_surrogate.csv").exists()
