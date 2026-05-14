"""Problem-type coverage using the current offline benchmark data path."""
from pathlib import Path

import pytest

from src.experiments.task.task import run_task
from src.experiments.task.task_config import DefaultTaskConfig
from src.search_space.search_space import SearchSpace


DATASETS_DIR = Path(__file__).parent.parent / "src" / "datasets" / "datasets"


def _dataset_csv_exists(dataset_id):
    return (DATASETS_DIR / f"{dataset_id}.csv").exists()


BINARY_CLASSIFIERS = [
    "AdaBoost",
    "Bagging",
    "DecisionTree",
    "ElasticNet",
    "ExtraTree",
    "ExtraTrees",
    "GaussianNB",
    "GradientBoosting",
    "KNeighbors",
    "Lasso",
    "LinearDiscriminantAnalysis",
    "LogisticRegression",
    "MLP",
    "PassiveAggressive",
    "RandomForest",
    "Ridge",
    "SGD",
    "XGBoost",
    "LGBM",
    "CatBoost",
]

MULTICLASS_CLASSIFIERS = [
    "AdaBoost",
    "Bagging",
    "CatBoost",
    "DecisionTree",
    "ExtraTree",
    "ExtraTrees",
    "GaussianNB",
    "GradientBoosting",
    "KNeighbors",
    "LinearDiscriminantAnalysis",
    "LogisticRegression",
    "MLP",
    "PassiveAggressive",
    "RandomForest",
    "Ridge",
    "SGD",
    "XGBoost",
    "LGBM",
]

REGRESSORS = [
    "AdaBoost",
    "Bagging",
    "CatBoost",
    "DecisionTree",
    "ElasticNet",
    "ExtraTree",
    "ExtraTrees",
    "GradientBoosting",
    "KNeighbors",
    "Lasso",
    "LGBM",
    "MLP",
    "RandomForest",
    "Ridge",
    "SGD",
    "XGBoost",
]


DATASET_SPECS = [
    pytest.param("regression", 363698, "neg_root_mean_squared_error", id="regression_363698"),
    pytest.param("binary", 363613, "roc_auc", id="binary_363613"),
    pytest.param("multiclass", 363614, "roc_auc", id="multiclass_363614"),
    pytest.param("regression", 363625, "neg_root_mean_squared_error", id="regression_363625"),
    pytest.param("binary", 363616, "roc_auc", id="binary_363616"),
    pytest.param("binary", 363618, "roc_auc", id="binary_363618"),
    pytest.param("binary", 363619, "roc_auc", id="binary_363619"),
    pytest.param("binary", 363620, "roc_auc", id="binary_363620"),
    pytest.param("binary", 363621, "roc_auc", id="binary_363621"),
    pytest.param("binary", 363623, "roc_auc", id="binary_363623"),
    pytest.param("binary", 363624, "roc_auc", id="binary_363624"),
    pytest.param("regression", 363675, "neg_root_mean_squared_error", id="regression_363675"),
    pytest.param("binary", 363626, "roc_auc", id="binary_363626"),
    pytest.param("binary", 363627, "roc_auc", id="binary_363627"),
    pytest.param("binary", 363628, "roc_auc", id="binary_363628"),
    pytest.param("binary", 363629, "roc_auc", id="binary_363629"),
    pytest.param("binary", 363630, "roc_auc", id="binary_363630"),
    pytest.param("regression", 363612, "neg_root_mean_squared_error", id="regression_363612"),
    pytest.param("binary", 363632, "roc_auc", id="binary_363632"),
    pytest.param("binary", 363671, "roc_auc", id="binary_363671"),
    pytest.param("regression", 363615, "neg_root_mean_squared_error", id="regression_363615"),
    pytest.param("binary", 363673, "roc_auc", id="binary_363673"),
    pytest.param("binary", 363674, "roc_auc", id="binary_363674"),
    pytest.param("regression", 363697, "neg_root_mean_squared_error", id="regression_363697"),
    pytest.param("binary", 363676, "roc_auc", id="binary_363676"),
    pytest.param("multiclass", 363685, "roc_auc", id="multiclass_363685"),
    pytest.param("regression", 363708, "neg_root_mean_squared_error", id="regression_363708"),
    pytest.param("binary", 363679, "roc_auc", id="binary_363679"),
    pytest.param("binary", 363681, "roc_auc", id="binary_363681"),
    pytest.param("binary", 363682, "roc_auc", id="binary_363682"),
    pytest.param("binary", 363683, "roc_auc", id="binary_363683"),
    pytest.param("binary", 363684, "roc_auc", id="binary_363684"),
    pytest.param("multiclass", 363707, "roc_auc", id="multiclass_363707"),
    pytest.param("regression", 363686, "neg_root_mean_squared_error", id="regression_363686"),
    pytest.param("binary", 363689, "roc_auc", id="binary_363689"),
    pytest.param("binary", 363691, "roc_auc", id="binary_363691"),
    pytest.param("regression", 363678, "neg_root_mean_squared_error", id="regression_363678"),
    pytest.param("binary", 363694, "roc_auc", id="binary_363694"),
    pytest.param("binary", 363696, "roc_auc", id="binary_363696"),
    pytest.param("regression", 363705, "neg_root_mean_squared_error", id="regression_363705"),
    pytest.param("regression", 363672, "neg_root_mean_squared_error", id="regression_363672"),
    pytest.param("multiclass", 363711, "roc_auc", id="multiclass_363711"),
    pytest.param("binary", 363700, "roc_auc", id="binary_363700"),
    pytest.param("multiclass", 363702, "roc_auc", id="multiclass_363702"),
    pytest.param("multiclass", 363677, "roc_auc", id="multiclass_363677"),
    pytest.param("regression", 363693, "neg_root_mean_squared_error", id="regression_363693"),
    pytest.param("binary", 363706, "roc_auc", id="binary_363706"),
    pytest.param("multiclass", 363704, "roc_auc", id="multiclass_363704"),
    pytest.param("regression", 363631, "neg_root_mean_squared_error", id="regression_363631"),
    pytest.param("multiclass", 363699, "roc_auc", id="multiclass_363699"),
    pytest.param("binary", 363712, "roc_auc", id="binary_363712"),
]

AVAILABLE_DATASET_SPECS = [
    spec for spec in DATASET_SPECS if _dataset_csv_exists(spec.values[1])
]


def make_task(problem_type, dataset_id, metric, model_name):
    task = DefaultTaskConfig()
    task.problem_type = problem_type
    task.dataset_id = dataset_id
    task.metric = metric
    task.optimizer = "random_search"
    task.iterations = 1
    task.debug = True
    task.offline_data_loading = True
    task.search_space.scalers = ["StandardScaler"]
    task.search_space.encoders = ["OrdinalEncoder"]
    task.search_space.dim_reducers = ["None"]
    task.search_space.feat_selectors = ["None"]
    task.search_space.imputers = ["NumericalSimpleImputer"]

    if problem_type == "regression":
        task.search_space.classifiers = []
        task.search_space.regressors = [model_name]
    else:
        task.search_space.classifiers = [model_name]
        task.search_space.regressors = []

    return task


class TestProblemTypeSearchSpaces:
    @pytest.mark.parametrize("model_name", BINARY_CLASSIFIERS)
    def test_binary_problem_type_supports_all_binary_models(self, model_name):
        task = make_task("binary", 1590, "accuracy", model_name)
        config = SearchSpace(task.to_dict()).get_space().sample_configuration()
        assert config["model"] == model_name

    @pytest.mark.parametrize("model_name", MULTICLASS_CLASSIFIERS)
    def test_multiclass_problem_type_supports_all_multiclass_models(self, model_name):
        task = make_task("multiclass", 363614, "accuracy", model_name)
        config = SearchSpace(task.to_dict()).get_space().sample_configuration()
        assert config["model"] == model_name

    @pytest.mark.parametrize("model_name", REGRESSORS)
    def test_regression_problem_type_supports_all_regressors(self, model_name):
        task = make_task("regression", 363697, "neg_root_mean_squared_error", model_name)
        config = SearchSpace(task.to_dict()).get_space().sample_configuration()
        assert config["model"] == model_name


class TestProblemTypeRunTaskSmoke:
    @pytest.mark.parametrize(
        ("problem_type", "dataset_id", "metric"),
        AVAILABLE_DATASET_SPECS,
    )
    def test_run_task_smoke_for_each_problem_type(self, problem_type, dataset_id, metric):
        model_name = "LGBM"
        task = make_task(problem_type, dataset_id, metric, model_name)

        history = run_task(task)

        assert len(history.history) == 1
        assert str(history.history[0].config["model"]) == model_name

        for fold in history.history[0].folds:
            assert fold.scores.val is not None