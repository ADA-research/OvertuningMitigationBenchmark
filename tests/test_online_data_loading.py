"""Tests for the online TabArena data loader without hitting the network."""
from types import SimpleNamespace

import pandas as pd
import pytest

from src.datasets.tabarena_dataloader import TabarenaDataLoader


class FakeDataset:
    def __init__(self, X, y):
        self._X = X
        self._y = y

    def get_data(self, target=None, dataset_format="dataframe"):
        return self._X.copy(), self._y.copy(), None, None


class FakeTask:
    def __init__(self, task_id, task_type, class_labels, dataset):
        self.task_id = task_id
        self.task_type = task_type
        self.class_labels = class_labels
        self.target_name = "target"
        self._dataset = dataset

    def get_dataset(self):
        return self._dataset


@pytest.fixture
def fake_tabarena(monkeypatch):
    import src.datasets.tabarena_dataloader as tabarena_module

    tasks = {
        101: FakeTask(
            task_id=101,
            task_type="Supervised Classification",
            class_labels=["no", "yes"],
            dataset=FakeDataset(
                X=pd.DataFrame(
                    {
                        "numeric": [1.0, 2.0, 3.0],
                        "category": ["a", "b", "a"],
                    }
                ),
                y=pd.Series(["yes", "no", "yes"], name="target"),
            ),
        ),
        202: FakeTask(
            task_id=202,
            task_type="Supervised Classification",
            class_labels=["red", "green", "blue"],
            dataset=FakeDataset(
                X=pd.DataFrame({"numeric": [5.0, 6.0, 7.0]}),
                y=pd.Series(["red", "green", "blue"], name="target"),
            ),
        ),
        303: FakeTask(
            task_id=303,
            task_type="Supervised Regression",
            class_labels=None,
            dataset=FakeDataset(
                X=pd.DataFrame({"feature": [10.0, 11.0, 12.0]}),
                y=pd.Series([0.5, 1.5, 2.5], name="target"),
            ),
        ),
    }

    suite = SimpleNamespace(tasks=list(tasks))

    monkeypatch.setattr(tabarena_module.openml.study, "get_suite", lambda _: suite)
    monkeypatch.setattr(tabarena_module.openml.tasks, "get_task", lambda task_id: tasks[task_id])

    return tasks


class TestTabarenaDataLoaderTaskEnumeration:
    def test_filters_binary_multiclass_and_regression_tasks(self, fake_tabarena):
        loader = TabarenaDataLoader()

        assert loader.get_all_binary() == [101]
        assert loader.get_all_multiclass() == [202]
        assert loader.get_all_classification() == [101, 202]
        assert loader.get_all_regression() == [303]
        assert loader.get_all_tasks() == [101, 202, 303]


class TestTabarenaDataLoaderLoad:
    def test_binary_load_encodes_targets_and_categories(self, fake_tabarena):
        loader = TabarenaDataLoader()

        X, y, label_map = loader.load(101)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X["category"].dtype.name == "category"
        assert sorted(y.unique()) == [0, 1]
        assert label_map == {0: "no", 1: "yes"}

    def test_multiclass_load_returns_numeric_targets(self, fake_tabarena):
        loader = TabarenaDataLoader()

        _, y, label_map = loader.load(202)

        assert sorted(y.unique()) == [0, 1, 2]
        assert label_map == {0: "blue", 1: "green", 2: "red"}

    def test_regression_load_preserves_target_values(self, fake_tabarena):
        loader = TabarenaDataLoader()

        X, y, label_map = loader.load(303)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert y.tolist() == [0.5, 1.5, 2.5]
        assert label_map is None