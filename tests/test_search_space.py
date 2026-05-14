"""
Tests for search space and pipeline construction.

The benchmark uses LGBM with StandardScaler, OneHotEncoder, NumericalSimpleImputer, no dim-reduction.
We test that this specific configuration and the broader search space work correctly.
"""
import pytest
from ConfigSpace import ConfigurationSpace

from src.search_space.search_space import SearchSpace
from src.experiments.task.task_config import DefaultTaskConfig, TaskConfig, SearchSpaceConfig


def make_binary_config(classifiers=None, scalers=None, encoders=None,
                        imputers=None, dim_reducers=None, feat_selectors=None):
    task = DefaultTaskConfig()
    if classifiers:
        task.search_space.classifiers = classifiers
    if scalers:
        task.search_space.scalers = scalers
    if encoders:
        task.search_space.encoders = encoders
    if imputers:
        task.search_space.imputers = imputers
    if dim_reducers:
        task.search_space.dim_reducers = dim_reducers
    if feat_selectors:
        task.search_space.feat_selectors = feat_selectors
    return task.to_dict()


class TestSearchSpaceConstruction:
    def test_builds_binary_space(self):
        cfg = make_binary_config()
        space = SearchSpace(cfg).get_space()
        assert isinstance(space, ConfigurationSpace)

    def test_builds_with_lgbm_only(self):
        cfg = make_binary_config(classifiers=["LGBM"])
        space = SearchSpace(cfg).get_space()
        config = space.sample_configuration()
        assert config["model"] == "LGBM"

    def test_benchmark_lgbm_config_has_expected_hps(self):
        """
        The benchmark uses only LGBM. Sampled config must contain LGBM-specific parameters.
        """
        cfg = make_binary_config(
            classifiers=["LGBM"],
            scalers=["StandardScaler"],
            encoders=["OneHotEncoder"],
            imputers=["NumericalSimpleImputer"],
            dim_reducers=["None"],
            feat_selectors=["None"],
        )
        space = SearchSpace(cfg).get_space()
        config = space.sample_configuration()
        d = dict(config)

        assert d["model"] == "LGBM"
        assert d["scaler"] == "StandardScaler"
        assert d["encoder"] == "OneHotEncoder"
        assert d["imputer"] == "NumericalSimpleImputer"

    def test_sampling_is_reproducible_with_seed(self):
        cfg = make_binary_config()
        space1 = SearchSpace(cfg).get_space()
        space2 = SearchSpace(cfg).get_space()
        # Same seed → same first sample
        c1 = space1.sample_configuration()
        c2 = space2.sample_configuration()
        assert dict(c1) == dict(c2)

    def test_multiple_classifiers_all_can_be_sampled(self):
        for clf in ["LGBM", "RandomForest", "DecisionTree", "LogisticRegression"]:
            cfg = make_binary_config(classifiers=[clf])
            space = SearchSpace(cfg).get_space()
            config = space.sample_configuration()
            assert config["model"] == clf

    def test_multiclass_space_builds(self):
        task = DefaultTaskConfig()
        task.problem_type = "multiclass"
        task.search_space.classifiers = ["LGBM"]
        space = SearchSpace(task.to_dict()).get_space()
        assert isinstance(space, ConfigurationSpace)

    def test_regression_space_builds(self):
        task = DefaultTaskConfig()
        task.problem_type = "regression"
        task.search_space.regressors = ["LGBM"]
        task.search_space.classifiers = []
        space = SearchSpace(task.to_dict()).get_space()
        assert isinstance(space, ConfigurationSpace)

    def test_unsupported_problem_type_raises(self):
        task = DefaultTaskConfig()
        task.problem_type = "unknown_type"
        with pytest.raises(ValueError):
            SearchSpace(task.to_dict()).get_space()


class TestPreprocessorSearchSpace:
    @pytest.mark.parametrize("scaler", [
        "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler",
        "Normalizer", "QuantileTransformer", "PowerTransformer",
    ])
    def test_all_scalers_build(self, scaler):
        cfg = make_binary_config(scalers=[scaler])
        space = SearchSpace(cfg).get_space()
        config = space.sample_configuration()
        assert config["scaler"] == scaler

    @pytest.mark.parametrize("imputer", [
        "NumericalSimpleImputer", "IterativeImputer", "KNNImputer",
    ])
    def test_all_imputers_build(self, imputer):
        cfg = make_binary_config(imputers=[imputer])
        space = SearchSpace(cfg).get_space()
        config = space.sample_configuration()
        assert config["imputer"] == imputer

    @pytest.mark.parametrize("encoder", ["OneHotEncoder", "OrdinalEncoder"])
    def test_all_encoders_build(self, encoder):
        cfg = make_binary_config(encoders=[encoder])
        space = SearchSpace(cfg).get_space()
        config = space.sample_configuration()
        assert config["encoder"] == encoder

    @pytest.mark.parametrize("dim_reducer", ["None", "PCA", "FastICA"])
    def test_all_dim_reducers_build(self, dim_reducer):
        cfg = make_binary_config(dim_reducers=[dim_reducer])
        space = SearchSpace(cfg).get_space()
        config = space.sample_configuration()
        assert config["dim_reducer"] == dim_reducer

    @pytest.mark.parametrize("feat_selector", [
        "None", "VarianceThreshold", "SelectKBest", "SelectPercentile",
    ])
    def test_all_feat_selectors_build(self, feat_selector):
        cfg = make_binary_config(feat_selectors=[feat_selector])
        space = SearchSpace(cfg).get_space()
        config = space.sample_configuration()
        assert config["feat_selector"] == feat_selector


class TestPipelineConstruction:
    """Tests that PipelineComponent.construct() correctly builds sklearn pipelines."""

    def test_lgbm_pipeline_constructs(self):
        from src.models.pipeline import PipelineComponent
        cfg = make_binary_config(classifiers=["LGBM"])
        space = SearchSpace(cfg).get_space()
        config = space.sample_configuration()

        pipeline = PipelineComponent(problem_type="binary")
        estimator = pipeline.construct(config)
        assert estimator is not None

    def test_pipeline_has_predict_proba(self):
        from src.models.pipeline import PipelineComponent
        cfg = make_binary_config(classifiers=["LGBM"])
        space = SearchSpace(cfg).get_space()
        config = space.sample_configuration()

        pipeline = PipelineComponent(problem_type="binary")
        estimator = pipeline.construct(config)
        assert hasattr(estimator, "predict_proba")

    @pytest.mark.parametrize("clf", [
        "LGBM", "RandomForest", "DecisionTree", "LogisticRegression",
        "GradientBoosting", "ExtraTrees", "KNeighbors",
    ])
    def test_classifiers_all_construct(self, clf):
        from src.models.pipeline import PipelineComponent
        cfg = make_binary_config(classifiers=[clf])
        space = SearchSpace(cfg).get_space()
        config = space.sample_configuration()

        pipeline = PipelineComponent(problem_type="binary")
        estimator = pipeline.construct(config)
        assert estimator is not None

    def test_pipeline_fit_and_predict(self):
        """Constructed pipeline must fit on data and produce probability predictions."""
        import numpy as np
        import pandas as pd
        from src.models.pipeline import PipelineComponent

        cfg = make_binary_config(classifiers=["LGBM"])
        space = SearchSpace(cfg).get_space()
        config = space.sample_configuration()

        pipeline = PipelineComponent(problem_type="binary")
        estimator = pipeline.construct(config)

        rng = np.random.default_rng(0)
        n = 100
        X = pd.DataFrame(rng.standard_normal((n, 5)), columns=list("abcde"))
        y = pd.Series(rng.integers(0, 2, n))

        estimator.fit(X[:80], y[:80])
        preds = estimator.predict_proba(X[80:])
        assert preds.shape == (20, 2)
        # Probabilities should sum to 1.0 per row
        assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-5)
