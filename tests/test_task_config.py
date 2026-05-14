"""
Tests for TaskConfig: construction, serialization, and field validation.
"""
import yaml
import pytest

from src.experiments.task.task_config import (
    DefaultTaskConfig,
    InnerEvaluationConfig,
    OuterEvaluationConfig,
    SearchSpaceConfig,
    TaskConfig,
)


class TestTaskConfigConstruction:
    def test_default_task_config_fields(self):
        task = DefaultTaskConfig()
        assert task.problem_type == "binary"
        assert task.optimizer == "smac"
        assert task.metric == "accuracy"
        assert task.iterations == 5
        assert task.bo_initial_random_iterations == 25  # DefaultTaskConfig default
        assert task.debug is True
        assert task.offline_data_loading is True
        assert task.dataset_id == 1590
        assert task.random_state == 0
        assert task.mitigation_strategy == "none"
        assert task.racing_strategy == "none"

    def test_default_inner_evaluation(self):
        task = DefaultTaskConfig()
        assert task.evaluation.resampling == "holdout"
        assert task.evaluation.val_size == 0.2
        assert task.evaluation.retrain is True
        assert task.evaluation.n_folds == 1
        assert task.evaluation.n_repeats == 1
        assert task.evaluation.reshuffle is False

    def test_default_outer_evaluation(self):
        task = DefaultTaskConfig()
        assert task.outer_evaluation.resampling == "holdout"
        assert task.outer_evaluation.train_size == 500
        assert task.outer_evaluation.fold == 0
        assert task.outer_evaluation.repeat == 0

    def test_default_search_space(self):
        task = DefaultTaskConfig()
        assert "LGBM" in task.search_space.classifiers
        assert "StandardScaler" in task.search_space.scalers
        assert "NumericalSimpleImputer" in task.search_space.imputers

    def test_task_config_field_mutation(self):
        task = DefaultTaskConfig()
        task.iterations = 100
        task.optimizer = "hebo"
        task.metric = "roc_auc"
        task.mitigation_strategy = "thresholdout"
        task.racing_strategy = "bergman_aggressive"
        assert task.iterations == 100
        assert task.optimizer == "hebo"
        assert task.metric == "roc_auc"
        assert task.mitigation_strategy == "thresholdout"
        assert task.racing_strategy == "bergman_aggressive"

    def test_inner_evaluation_config_fields(self):
        cfg = InnerEvaluationConfig(
            resampling="cv",
            val_size=0.25,
            selection_size=0.15,
            n_folds=5,
            reshuffle=True,
            retrain=True,
            n_repeats=3,
        )
        assert cfg.resampling == "cv"
        assert cfg.val_size == 0.25
        assert cfg.selection_size == 0.15
        assert cfg.n_folds == 5
        assert cfg.reshuffle is True
        assert cfg.retrain is True
        assert cfg.n_repeats == 3

    def test_outer_evaluation_config_fields(self):
        cfg = OuterEvaluationConfig(
            resampling="cv",
            train_size=0.8,
            n_folds=3,
            n_repeats=10,
            fold=2,
            repeat=4,
        )
        assert cfg.resampling == "cv"
        assert cfg.n_folds == 3
        assert cfg.n_repeats == 10
        assert cfg.fold == 2
        assert cfg.repeat == 4

    def test_search_space_config_fields(self):
        ss = SearchSpaceConfig(
            classifiers=["LGBM", "RandomForest"],
            regressors=["LGBM"],
            scalers=["StandardScaler"],
            dim_reducers=["None"],
            imputers=["NumericalSimpleImputer"],
            encoders=["OneHotEncoder"],
            feat_selectors=["None"],
        )
        assert ss.classifiers == ["LGBM", "RandomForest"]
        assert ss.regressors == ["LGBM"]
        assert ss.encoders == ["OneHotEncoder"]


class TestTaskConfigToDict:
    def test_to_dict_has_required_keys(self):
        task = DefaultTaskConfig()
        d = task.to_dict()

        required_top_level = [
            "task_id", "random_state", "problem_type", "dataset_id",
            "debug", "optimizer", "metric", "offline", "offline_data_loading",
            "store_results_google_cloud", "store_vectors_google_cloud",
            "outer_evaluation", "evaluation", "search_space",
            "max_iterations", "bo_initial_random_iterations", "smac_surrogate_model",
            "smac_surrogate_random_forest_n_trees",
            "mitigation_strategy", "racing_strategy",
        ]
        for key in required_top_level:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_search_space_structure(self):
        task = DefaultTaskConfig()
        d = task.to_dict()
        ss = d["search_space"]
        assert "classifiers" in ss
        assert "regressors" in ss
        assert "preprocessors" in ss
        preprocessors = ss["preprocessors"]
        for pp_key in ["scalers", "dim_reducers", "imputers", "encoders", "feat_selectors"]:
            assert pp_key in preprocessors, f"Missing preprocessor key: {pp_key}"

    def test_to_dict_evaluation_structure(self):
        task = DefaultTaskConfig()
        d = task.to_dict()
        ev = d["evaluation"]
        for key in ["resampling", "val_size", "selection_size", "reshuffle", "retrain", "n_folds", "n_repeats"]:
            assert key in ev, f"Missing evaluation key: {key}"

    def test_to_dict_outer_evaluation_structure(self):
        task = DefaultTaskConfig()
        d = task.to_dict()
        oe = d["outer_evaluation"]
        for key in ["resampling", "train_size", "n_folds", "n_repeats", "fold", "repeat"]:
            assert key in oe, f"Missing outer_evaluation key: {key}"

    def test_to_dict_values_match_config(self):
        task = DefaultTaskConfig()
        task.iterations = 42
        task.optimizer = "hebo"
        task.metric = "roc_auc"
        task.evaluation.n_folds = 7
        task.outer_evaluation.repeat = 3

        d = task.to_dict()
        assert d["max_iterations"] == 42
        assert d["optimizer"] == "hebo"
        assert d["metric"] == "roc_auc"
        assert d["evaluation"]["n_folds"] == 7
        assert d["outer_evaluation"]["repeat"] == 3

    def test_to_dict_search_space_classifiers_list(self):
        task = DefaultTaskConfig()
        task.search_space.classifiers = ["LGBM", "RandomForest"]
        d = task.to_dict()
        assert d["search_space"]["classifiers"] == ["LGBM", "RandomForest"]


class TestTaskConfigToYaml:
    def test_to_yaml_returns_string(self, tmp_path):
        task = DefaultTaskConfig()
        yaml_str = task.to_yaml(None)
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0

    def test_to_yaml_writes_file(self, tmp_path):
        task = DefaultTaskConfig()
        path = str(tmp_path / "task_config.yaml")
        task.to_yaml(path)
        assert (tmp_path / "task_config.yaml").exists()

    def test_to_yaml_readable_yaml(self, tmp_path):
        task = DefaultTaskConfig()
        path = str(tmp_path / "task_config.yaml")
        task.to_yaml(path)
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert isinstance(loaded, dict)
        assert "optimizer" in loaded
        assert "metric" in loaded
        assert "search_space" in loaded

    def test_to_yaml_contains_correct_values(self, tmp_path):
        task = DefaultTaskConfig()
        task.optimizer = "hebo"
        task.metric = "roc_auc"
        task.iterations = 77

        path = str(tmp_path / "task_config.yaml")
        task.to_yaml(path)
        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["optimizer"] == "hebo"
        assert loaded["metric"] == "roc_auc"
        assert loaded["max_iterations"] == 77  # key is max_iterations in to_dict

    def test_to_yaml_format_differs_from_from_config_format(self, tmp_path):
        """
        Documents a known API inconsistency:
        to_yaml() saves classifiers nested under search_space,
        but from_config() expects classifiers at the top level.
        Reloading a to_yaml() output via from_config() will raise a KeyError.
        """
        task = DefaultTaskConfig()
        path = str(tmp_path / "task_config.yaml")
        task.to_yaml(path)

        with open(path) as f:
            loaded = yaml.safe_load(f)

        # to_yaml puts classifiers nested; from_config expects them at top level
        assert "classifiers" not in loaded, (
            "to_yaml() should NOT have top-level 'classifiers' key "
            "(it's nested under search_space.classifiers)"
        )
        assert "classifiers" in loaded["search_space"]

    def test_search_space_usable_from_to_dict(self):
        """to_dict() output can be used to construct a SearchSpace."""
        from src.search_space.search_space import SearchSpace

        task = DefaultTaskConfig()
        d = task.to_dict()
        space = SearchSpace(d).get_space()
        assert space is not None
        config = space.sample_configuration()
        assert config["model"] == "LGBM"
