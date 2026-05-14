"""Tests for model search spaces, wrappers, and lightweight train/test loops."""
import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from src.experiments.task.task_config import DefaultTaskConfig
from src.models.classifiers.fastaimlp import FastAIMLPClassifier
from src.models.classifiers.lgbm import LGBMModel
from src.models.pipeline import PipelineComponent
from src.models.regressors.fastaimlp import FastAIMLPRegressorModel
from src.models.regressors.lgbm import LGBMRegressorModel
from src.search_space.search_space import SearchSpace


AUTOGLUON_AVAILABLE = importlib.util.find_spec("autogluon") is not None
REALMLP_AVAILABLE = importlib.util.find_spec("pytabkit") is not None


def _fastaimlp_runtime_available():
    if not AUTOGLUON_AVAILABLE:
        return False

    try:
        import autogluon.tabular.models.fastainn.imports_helper  # noqa: F401
    except Exception:
        return False

    return True


FASTAIMLP_RUNTIME_AVAILABLE = _fastaimlp_runtime_available()

MODEL_PARAMS = [
    pytest.param(
        "FastAIMLP",
        id="fastaimlp",
        marks=pytest.mark.skipif(not AUTOGLUON_AVAILABLE, reason="autogluon is not installed"),
    ),
    pytest.param(
        "RealMLP",
        id="realmlp",
        marks=pytest.mark.skipif(not REALMLP_AVAILABLE, reason="pytabkit is not installed"),
    ),
    pytest.param("LGBM", id="lgbm"),
]


MODEL_SPECS = {
    "FastAIMLP": {
        "encoder": "OrdinalEncoder",
        "classification_hyperparameters": {
            "FastAIMLP_layers",
            "FastAIMLP_emb_drop",
            "FastAIMLP_ps",
            "FastAIMLP_bs",
            "FastAIMLP_lr",
            "FastAIMLP_epochs",
        },
        "regression_hyperparameters": {
            "FastAIMLP_layers",
            "FastAIMLP_emb_drop",
            "FastAIMLP_ps",
            "FastAIMLP_bs",
            "FastAIMLP_lr",
            "FastAIMLP_epochs",
        },
    },
    "RealMLP": {
        "encoder": "OrdinalEncoder",
        "classification_hyperparameters": {
            "RealMLP_hidden_layer_sizes",
            "RealMLP_num_embedding_type",
            "RealMLP_activation_function",
            "RealMLP_dropout_prob",
            "RealMLP_weight_decay",
            "RealMLP_learning_rate",
            "RealMLP_use_scaling_layer",
            "RealMLP_w_init_std",
        },
        "regression_hyperparameters": {
            "RealMLP_hidden_layer_sizes",
            "RealMLP_num_embedding_type",
            "RealMLP_activation_function",
            "RealMLP_dropout_prob",
            "RealMLP_weight_decay",
            "RealMLP_learning_rate",
            "RealMLP_use_scaling_layer",
            "RealMLP_w_init_std",
        },
    },
    "LGBM": {
        "encoder": "OrdinalEncoder",
        "classification_hyperparameters": {
            "LGBM_n_estimators",
            "LGBM_bagging_freq",
            "LGBM_learning_rate",
            "LGBM_num_leaves",
            "LGBM_feature_fraction",
            "LGBM_bagging_fraction",
            "LGBM_min_data_in_leaf",
            "LGBM_min_sum_hessian_in_leaf",
            "LGBM_lambda_l1_use",
            "LGBM_lambda_l2_use",
        },
        "regression_hyperparameters": {
            "LGBM_n_estimators",
            "LGBM_bagging_freq",
            "LGBM_learning_rate",
            "LGBM_num_leaves",
            "LGBM_feature_fraction",
            "LGBM_bagging_fraction",
            "LGBM_min_data_in_leaf",
            "LGBM_min_sum_hessian_in_leaf",
            "LGBM_lambda_l1_use",
            "LGBM_lambda_l2_use",
        },
    },
}


def get_classifier_wrapper(model_name):
    if model_name == "FastAIMLP":
        return FastAIMLPClassifier
    if model_name == "RealMLP":
        from src.models.classifiers.realmlp import RealMLPModel

        return RealMLPModel
    if model_name == "LGBM":
        return LGBMModel
    raise ValueError(f"Unsupported model: {model_name}")


def get_regressor_wrapper(model_name):
    if model_name == "FastAIMLP":
        return FastAIMLPRegressorModel
    if model_name == "RealMLP":
        from src.models.regressors.realmlp import RealMLPRegressorModel

        return RealMLPRegressorModel
    if model_name == "LGBM":
        return LGBMRegressorModel
    raise ValueError(f"Unsupported model: {model_name}")


def make_model_task(model_name, problem_type="binary"):
    task = DefaultTaskConfig()
    task.problem_type = problem_type
    task.random_state = 78
    task.search_space.scalers = ["StandardScaler"]
    task.search_space.encoders = [MODEL_SPECS[model_name]["encoder"]]
    task.search_space.dim_reducers = ["None"]
    task.search_space.feat_selectors = ["None"]
    task.search_space.imputers = ["NumericalSimpleImputer"]

    if problem_type == "regression":
        task.search_space.classifiers = []
        task.search_space.regressors = [model_name]
        task.metric = "neg_root_mean_squared_error"
    else:
        task.search_space.classifiers = [model_name]
        task.search_space.regressors = []
        task.metric = "accuracy"

    return task


def make_mixed_binary_data(n=72, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "num_a": rng.standard_normal(n),
            "num_b": rng.standard_normal(n),
            "num_c": rng.standard_normal(n),
            "cat_a": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
            "cat_b": pd.Categorical(rng.choice(["x", "y"], size=n)),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n), name="target")
    return X, y


def make_mixed_regression_data(n=60, seed=24):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "num_a": rng.standard_normal(n),
            "num_b": rng.standard_normal(n),
            "num_c": rng.standard_normal(n),
            "cat_a": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        }
    )
    y = pd.Series(
        1.5 * X["num_a"] - 0.75 * X["num_b"] + rng.standard_normal(n) * 0.1,
        name="target",
    )
    return X, y


def sample_model_config(model_name, problem_type="binary"):
    task = make_model_task(model_name=model_name, problem_type=problem_type)
    space = SearchSpace(task.to_dict()).get_space()
    return dict(space.sample_configuration())


class TestModelSearchSpace:
    def test_fastaimlp_canonical_model_name(self):
        task = make_model_task(model_name="FastAIMLP", problem_type="binary")
        config = dict(SearchSpace(task.to_dict()).get_space().sample_configuration())
        assert config["model"] == "FastAIMLP"

    @pytest.mark.parametrize("model_name", MODEL_PARAMS)
    def test_binary_space_contains_expected_hyperparameters(self, model_name):
        config = sample_model_config(model_name=model_name, problem_type="binary")

        assert config["model"] == model_name
        assert config["encoder"] == MODEL_SPECS[model_name]["encoder"]
        assert config["scaler"] == "StandardScaler"
        assert MODEL_SPECS[model_name]["classification_hyperparameters"] <= set(config)

    @pytest.mark.parametrize("model_name", MODEL_PARAMS)
    def test_regression_space_contains_expected_hyperparameters(self, model_name):
        config = sample_model_config(model_name=model_name, problem_type="regression")

        assert config["model"] == model_name
        assert config["encoder"] == MODEL_SPECS[model_name]["encoder"]
        assert config["scaler"] == "StandardScaler"
        assert MODEL_SPECS[model_name]["regression_hyperparameters"] <= set(config)

    @pytest.mark.parametrize("model_name", MODEL_PARAMS)
    def test_sampling_is_reproducible_for_model_space(self, model_name):
        config1 = sample_model_config(model_name=model_name, problem_type="binary")
        config2 = sample_model_config(model_name=model_name, problem_type="binary")
        assert config1 == config2


class TestModelWrappers:
    @pytest.mark.parametrize("model_name", MODEL_PARAMS)
    def test_classifier_wrapper_constructs(self, model_name):
        config = sample_model_config(model_name=model_name, problem_type="binary")
        model = get_classifier_wrapper(model_name)().construct(config)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    @pytest.mark.parametrize("model_name", MODEL_PARAMS)
    def test_regressor_wrapper_constructs(self, model_name):
        config = sample_model_config(model_name=model_name, problem_type="regression")
        model = get_regressor_wrapper(model_name)().construct(config)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    @pytest.mark.parametrize("model_name", MODEL_PARAMS)
    def test_pipeline_constructs_for_binary_model(self, model_name):
        config = sample_model_config(model_name=model_name, problem_type="binary")
        estimator = PipelineComponent(problem_type="binary").construct(config)
        assert estimator is not None
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict_proba")

    @pytest.mark.parametrize("model_name", MODEL_PARAMS)
    def test_pipeline_constructs_for_regression_model(self, model_name):
        config = sample_model_config(model_name=model_name, problem_type="regression")
        estimator = PipelineComponent(problem_type="regression").construct(config)
        assert estimator is not None
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")


class TestModelTrainTestLoops:
    @pytest.mark.parametrize("model_name", MODEL_PARAMS)
    def test_binary_model_runs_stratified_train_test_loops(self, model_name):
        if model_name == "FastAIMLP" and not FASTAIMLP_RUNTIME_AVAILABLE:
            pytest.skip("fastaimlp runtime dependencies are not available")

        X, y = make_mixed_binary_data()
        config = sample_model_config(model_name=model_name, problem_type="binary")
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=78)

        fold_sizes = []
        for train_idx, test_idx in splitter.split(X, y):
            estimator = PipelineComponent(problem_type="binary").construct(config)
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = estimator.predict_proba(X.iloc[test_idx])

            fold_sizes.append(len(test_idx))
            assert proba.shape == (len(test_idx), 2)
            assert np.isfinite(proba).all()
            assert np.all((proba >= 0.0) & (proba <= 1.0))
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

        assert fold_sizes == [36, 36]

    @pytest.mark.parametrize("model_name", MODEL_PARAMS)
    def test_regression_model_runs_train_test_loops(self, model_name):
        if model_name == "FastAIMLP" and not FASTAIMLP_RUNTIME_AVAILABLE:
            pytest.skip("fastaimlp runtime dependencies are not available")

        X, y = make_mixed_regression_data()
        config = sample_model_config(model_name=model_name, problem_type="regression")
        splitter = KFold(n_splits=2, shuffle=True, random_state=78)

        fold_sizes = []
        for train_idx, test_idx in splitter.split(X):
            estimator = PipelineComponent(problem_type="regression").construct(config)
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = estimator.predict(X.iloc[test_idx])

            fold_sizes.append(len(test_idx))
            assert preds.shape == (len(test_idx),)
            assert np.isfinite(preds).all()

        assert fold_sizes == [30, 30]

    @pytest.mark.parametrize("model_name", MODEL_PARAMS)
    def test_binary_model_repeated_loops_keep_output_contract(self, model_name):
        if model_name == "FastAIMLP" and not FASTAIMLP_RUNTIME_AVAILABLE:
            pytest.skip("fastaimlp runtime dependencies are not available")

        X, y = make_mixed_binary_data(seed=7)
        config = sample_model_config(model_name=model_name, problem_type="binary")
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=11)

        predictions = []
        fold_sizes = []
        for train_idx, test_idx in splitter.split(X, y):
            estimator = PipelineComponent(problem_type="binary").construct(config)
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_sizes.append(len(test_idx))
            predictions.append(estimator.predict_proba(X.iloc[test_idx]))

        repeated_predictions = []
        for train_idx, test_idx in splitter.split(X, y):
            estimator = PipelineComponent(problem_type="binary").construct(config)
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            repeated_predictions.append(estimator.predict_proba(X.iloc[test_idx]))

        for first, second in zip(predictions, repeated_predictions):
            assert first.shape == second.shape
            assert first.shape[1] == 2
            assert np.isfinite(first).all()
            assert np.isfinite(second).all()
            assert np.all((first >= 0.0) & (first <= 1.0))
            assert np.all((second >= 0.0) & (second <= 1.0))
            assert np.allclose(first.sum(axis=1), 1.0, atol=1e-5)
            assert np.allclose(second.sum(axis=1), 1.0, atol=1e-5)

        assert fold_sizes == [36, 36]


def make_mixed_multiclass_data(n=90, n_classes=3, seed=55):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "num_a": rng.standard_normal(n),
            "num_b": rng.standard_normal(n),
            "num_c": rng.standard_normal(n),
            "cat_a": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        }
    )
    y = pd.Series(rng.integers(0, n_classes, size=n), name="target")
    return X, y


def _make_fake_autogluon(captured, predict_proba_output):
    """Return a fake NNFastAiTabularModel class that records calls."""

    class FakeModel:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def fit(self, **kwargs):
            captured["fit_kwargs"] = kwargs
            return self

        def predict(self, X):
            return pd.Series(np.zeros(len(X), dtype=int))

        def predict_proba(self, X):
            return predict_proba_output(X)

    return FakeModel


@pytest.mark.skipif(not AUTOGLUON_AVAILABLE, reason="autogluon is not installed")
class TestFastAIMLPContract:
    # ------------------------------------------------------------------
    # Layers lookup
    # ------------------------------------------------------------------

    def test_all_layers_options_resolve_correctly(self, monkeypatch):
        from src.models.classifiers.fastaimlp import LAYERS_OPTIONS

        expected = {
            0: [200],
            1: [400],
            2: [200, 100],
            3: [400, 200],
            4: [800, 400],
            5: [200, 100, 50],
            6: [400, 200, 100],
        }
        assert LAYERS_OPTIONS == expected

    @pytest.mark.parametrize("key,expected_layers", [
        (0, [200]),
        (3, [400, 200]),
        (6, [400, 200, 100]),
    ])
    def test_classifier_layers_key_resolved_in_hyperparameters(self, monkeypatch, key, expected_layers):
        captured = {}
        monkeypatch.setattr(
            "src.models.classifiers.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.DataFrame([[0.4, 0.6]] * len(X), columns=[0, 1])),
        )
        model = FastAIMLPClassifier().construct({"FastAIMLP_layers": key, "_model_threads": 1})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0, 1]))
        assert captured["init_kwargs"]["hyperparameters"]["layers"] == expected_layers

    @pytest.mark.parametrize("key,expected_layers", [
        (0, [200]),
        (5, [200, 100, 50]),
    ])
    def test_regressor_layers_key_resolved_in_hyperparameters(self, monkeypatch, key, expected_layers):
        captured = {}
        monkeypatch.setattr(
            "src.models.regressors.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.Series(np.ones(len(X)))),
        )
        model = FastAIMLPRegressorModel().construct({"FastAIMLP_layers": key, "_model_threads": 1})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0.1, 0.2]))
        assert captured["init_kwargs"]["hyperparameters"]["layers"] == expected_layers

    def test_classifier_no_layers_key_omits_layers_from_hyperparameters(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.models.classifiers.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.DataFrame([[0.5, 0.5]] * len(X), columns=[0, 1])),
        )
        model = FastAIMLPClassifier().construct({"_model_threads": 1})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0, 1]))
        assert "layers" not in captured["init_kwargs"]["hyperparameters"]

    # ------------------------------------------------------------------
    # problem_type forwarding
    # ------------------------------------------------------------------

    def test_classifier_defaults_to_binary_problem_type(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.models.classifiers.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.DataFrame([[0.4, 0.6]] * len(X), columns=[0, 1])),
        )
        model = FastAIMLPClassifier().construct({"_model_threads": 1})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0, 1]))
        assert captured["init_kwargs"]["problem_type"] == "binary"

    def test_classifier_uses_multiclass_problem_type_from_config(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.models.classifiers.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.DataFrame([[0.3, 0.3, 0.4]] * len(X), columns=[0, 1, 2])),
        )
        model = FastAIMLPClassifier().construct({"_problem_type": "multiclass", "_model_threads": 1})
        model.fit(pd.DataFrame({"a": [0.0, 1.0, 0.5]}), pd.Series([0, 1, 2]))
        assert captured["init_kwargs"]["problem_type"] == "multiclass"

    def test_regressor_always_uses_regression_problem_type(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.models.regressors.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.Series(np.ones(len(X)))),
        )
        model = FastAIMLPRegressorModel().construct({"_model_threads": 1})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0.1, 0.2]))
        assert captured["init_kwargs"]["problem_type"] == "regression"

    def test_classifier_forwards_random_state_as_random_seed(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.models.classifiers.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.DataFrame([[0.4, 0.6]] * len(X), columns=[0, 1])),
        )
        model = FastAIMLPClassifier().construct({"_model_threads": 1, "random_state": 123})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0, 1]))
        assert captured["init_kwargs"]["hyperparameters"]["random_seed"] == 123

    def test_regressor_forwards_random_state_as_random_seed(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.models.regressors.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.Series(np.ones(len(X)))),
        )
        model = FastAIMLPRegressorModel().construct({"_model_threads": 1, "random_state": 321})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0.1, 0.2]))
        assert captured["init_kwargs"]["hyperparameters"]["random_seed"] == 321

    def test_classifier_constructed_hyperparameters_survive_sklearn_clone(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.models.classifiers.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.DataFrame([[0.4, 0.6]] * len(X), columns=[0, 1])),
        )
        model = FastAIMLPClassifier().construct({
            "_problem_type": "binary",
            "_model_threads": 2,
            "random_state": 123,
            "FastAIMLP_layers": 3,
            "FastAIMLP_lr": 0.01,
        })
        cloned = clone(model)
        cloned.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0, 1]))

        assert captured["init_kwargs"]["problem_type"] == "binary"
        assert captured["init_kwargs"]["hyperparameters"]["layers"] == [400, 200]
        assert captured["init_kwargs"]["hyperparameters"]["random_seed"] == 123
        assert captured["fit_kwargs"]["num_cpus"] == 2

    def test_regressor_constructed_hyperparameters_survive_sklearn_clone(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.models.regressors.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.Series(np.ones(len(X)))),
        )
        model = FastAIMLPRegressorModel().construct({
            "_model_threads": 4,
            "random_state": 321,
            "FastAIMLP_layers": 5,
            "FastAIMLP_lr": 0.005,
        })
        cloned = clone(model)
        cloned.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0.1, 0.2]))

        assert captured["init_kwargs"]["problem_type"] == "regression"
        assert captured["init_kwargs"]["hyperparameters"]["layers"] == [200, 100, 50]
        assert captured["init_kwargs"]["hyperparameters"]["random_seed"] == 321
        assert captured["fit_kwargs"]["num_cpus"] == 4

    # ------------------------------------------------------------------
    # CPU / GPU enforcement
    # ------------------------------------------------------------------

    def test_classifier_forces_num_gpus_zero(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.models.classifiers.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.DataFrame([[0.4, 0.6]] * len(X), columns=[0, 1])),
        )
        model = FastAIMLPClassifier().construct({"_model_threads": 1})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0, 1]))
        assert captured["fit_kwargs"]["num_gpus"] == 0

    def test_regressor_forces_num_gpus_zero(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.models.regressors.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.Series(np.ones(len(X)))),
        )
        model = FastAIMLPRegressorModel().construct({"_model_threads": 1})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0.1, 0.2]))
        assert captured["fit_kwargs"]["num_gpus"] == 0

    @pytest.mark.parametrize("threads", [1, 2, 8])
    def test_classifier_forwards_model_threads_as_num_cpus(self, monkeypatch, threads):
        captured = {}
        monkeypatch.setattr(
            "src.models.classifiers.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.DataFrame([[0.4, 0.6]] * len(X), columns=[0, 1])),
        )
        model = FastAIMLPClassifier().construct({"_model_threads": threads})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0, 1]))
        assert captured["fit_kwargs"]["num_cpus"] == threads

    @pytest.mark.parametrize("threads", [1, 4])
    def test_regressor_forwards_model_threads_as_num_cpus(self, monkeypatch, threads):
        captured = {}
        monkeypatch.setattr(
            "src.models.regressors.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.Series(np.ones(len(X)))),
        )
        model = FastAIMLPRegressorModel().construct({"_model_threads": threads})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0.1, 0.2]))
        assert captured["fit_kwargs"]["num_cpus"] == threads

    # ------------------------------------------------------------------
    # predict_proba shape and probability contract
    # ------------------------------------------------------------------

    def test_classifier_predict_proba_binary_1d_output_converted_to_2d(self, monkeypatch):
        """AutoGluon may return 1D probabilities for binary; we must convert to (n, 2)."""
        monkeypatch.setattr(
            "src.models.classifiers.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon({}, lambda X: np.array([0.3, 0.7])),
        )
        model = FastAIMLPClassifier().construct({"_model_threads": 1})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0, 1]))
        proba = model.predict_proba(pd.DataFrame({"a": [0.0, 1.0]}))
        assert proba.shape == (2, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.allclose(proba[:, 0], [0.7, 0.3])
        assert np.allclose(proba[:, 1], [0.3, 0.7])

    def test_classifier_predict_proba_binary_2d_passes_through(self, monkeypatch):
        proba_2d = pd.DataFrame([[0.3, 0.7], [0.6, 0.4]], columns=[0, 1])
        monkeypatch.setattr(
            "src.models.classifiers.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon({}, lambda X: proba_2d),
        )
        model = FastAIMLPClassifier().construct({"_model_threads": 1})
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0, 1]))
        proba = model.predict_proba(pd.DataFrame({"a": [0.0, 1.0]}))
        assert proba.shape == (2, 2)

    def test_classifier_predict_proba_multiclass_shape(self, monkeypatch):
        n_classes = 4
        fake_proba = pd.DataFrame(
            [[0.1, 0.2, 0.3, 0.4]] * 3,
            columns=list(range(n_classes)),
        )
        monkeypatch.setattr(
            "src.models.classifiers.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon({}, lambda X: fake_proba),
        )
        model = FastAIMLPClassifier().construct({"_problem_type": "multiclass", "_model_threads": 1})
        model.fit(pd.DataFrame({"a": [0.0, 1.0, 0.5]}), pd.Series([0, 1, 2]))
        proba = model.predict_proba(pd.DataFrame({"a": [0.0, 1.0, 0.5]}))
        assert proba.shape == (3, n_classes)

    def test_classifier_all_hyperparameters_passed_correctly(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.models.classifiers.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.DataFrame([[0.4, 0.6]] * len(X), columns=[0, 1])),
        )
        config = {
            "_model_threads": 3,
            "FastAIMLP_layers": 3,
            "FastAIMLP_emb_drop": 0.2,
            "FastAIMLP_ps": 0.3,
            "FastAIMLP_bs": 256,
            "FastAIMLP_lr": 0.01,
            "FastAIMLP_epochs": 30,
        }
        model = FastAIMLPClassifier().construct(config)
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0, 1]))
        assert captured["init_kwargs"]["hyperparameters"] == {
            "layers": [400, 200],
            "emb_drop": 0.2,
            "ps": 0.3,
            "bs": 256,
            "lr": 0.01,
            "epochs": 30,
        }

    def test_regressor_all_hyperparameters_passed_correctly(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.models.regressors.fastaimlp.NNFastAiTabularModel",
            _make_fake_autogluon(captured, lambda X: pd.Series(np.ones(len(X)))),
        )
        config = {
            "_model_threads": 4,
            "FastAIMLP_layers": 5,
            "FastAIMLP_emb_drop": 0.1,
            "FastAIMLP_ps": 0.1,
            "FastAIMLP_bs": 512,
            "FastAIMLP_lr": 0.005,
            "FastAIMLP_epochs": 25,
        }
        model = FastAIMLPRegressorModel().construct(config)
        model.fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0.1, 0.2]))
        assert captured["init_kwargs"]["hyperparameters"] == {
            "layers": [200, 100, 50],
            "emb_drop": 0.1,
            "ps": 0.1,
            "bs": 512,
            "lr": 0.005,
            "epochs": 25,
        }
        assert captured["fit_kwargs"]["num_gpus"] == 0
        assert captured["fit_kwargs"]["num_cpus"] == 4


@pytest.mark.skipif(
    not _fastaimlp_runtime_available(),
    reason="fastaimlp runtime dependencies are not available",
)
class TestFastAIMLPRuntime:
    def test_binary_classification_proba_contract(self):
        X, y = make_mixed_binary_data()
        config = sample_model_config(model_name="FastAIMLP", problem_type="binary")
        config["_problem_type"] = "binary"
        pipeline = PipelineComponent(problem_type="binary").construct(config)
        pipeline.fit(X.iloc[:54], y.iloc[:54])
        proba = pipeline.predict_proba(X.iloc[54:])
        n = len(X) - 54
        assert proba.shape == (n, 2)
        assert np.isfinite(proba).all()
        assert np.all((proba >= 0.0) & (proba <= 1.0))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_multiclass_classification_proba_contract(self):
        X, y = make_mixed_multiclass_data(n_classes=3)
        task = make_model_task(model_name="FastAIMLP", problem_type="multiclass")
        config = dict(SearchSpace(task.to_dict()).get_space().sample_configuration())
        config["_problem_type"] = "multiclass"
        pipeline = PipelineComponent(problem_type="multiclass").construct(config)
        pipeline.fit(X.iloc[:66], y.iloc[:66])
        proba = pipeline.predict_proba(X.iloc[66:])
        n = len(X) - 66
        assert proba.shape == (n, 3)
        assert np.isfinite(proba).all()
        assert np.all((proba >= 0.0) & (proba <= 1.0))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_regression_predict_contract(self):
        X, y = make_mixed_regression_data()
        config = sample_model_config(model_name="FastAIMLP", problem_type="regression")
        pipeline = PipelineComponent(problem_type="regression").construct(config)
        pipeline.fit(X.iloc[:45], y.iloc[:45])
        preds = pipeline.predict(X.iloc[45:])
        n = len(X) - 45
        assert preds.shape == (n,)
        assert np.isfinite(preds).all()