"""
Tests for offline data loading — every benchmark dataset must load correctly.
"""
import os
from pathlib import Path

import pandas as pd
import pytest

from src.datasets.offline_dataloader import OfflineDataLoader

DATASETS_DIR = Path(__file__).parent.parent / "src" / "datasets" / "datasets"

# Collect all integer-ID datasets available on disk
_all_csv = [f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")]
NUMERIC_DATASET_IDS = sorted(
    int(f[:-4]) for f in _all_csv
    if f[:-4].isdigit()
)


class TestOfflineDataLoaderAdult:
    """Tests for the special-cased adult dataset (id=1590)."""

    def test_loads_without_error(self):
        loader = OfflineDataLoader()
        X, y, label_map = loader.load(1590, problem_type="binary")
        assert X is not None
        assert y is not None

    def test_returns_dataframe_and_series(self):
        loader = OfflineDataLoader()
        X, y, label_map = loader.load(1590, problem_type="binary")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_label_map_returned_for_binary(self):
        loader = OfflineDataLoader()
        X, y, label_map = loader.load(1590, problem_type="binary")
        assert label_map is not None
        assert isinstance(label_map, dict)
        assert len(label_map) == 2  # Binary: 2 classes

    def test_targets_are_integer_encoded(self):
        loader = OfflineDataLoader()
        X, y, label_map = loader.load(1590, problem_type="binary")
        unique_labels = sorted(y.unique())
        assert unique_labels == [0, 1]

    def test_no_target_column_in_features(self):
        loader = OfflineDataLoader()
        X, y, _ = loader.load(1590, problem_type="binary")
        assert "target" not in X.columns

    def test_row_count_matches(self):
        loader = OfflineDataLoader()
        X, y, _ = loader.load(1590, problem_type="binary")
        assert len(X) == len(y)
        assert len(X) > 0

    def test_object_columns_are_categorical(self):
        loader = OfflineDataLoader()
        X, y, _ = loader.load(1590, problem_type="binary")
        for col in X.select_dtypes(include="object").columns:
            assert X[col].dtype.name == "category", f"Column {col} should be category"

    def test_feature_count_is_positive(self):
        loader = OfflineDataLoader()
        X, _, _ = loader.load(1590, problem_type="binary")
        assert X.shape[1] > 0


class TestOfflineDataLoaderAllDatasets:
    """Every numeric dataset CSV must load correctly."""

    @pytest.mark.parametrize("dataset_id", NUMERIC_DATASET_IDS)
    def test_loads_returns_correct_types(self, dataset_id):
        loader = OfflineDataLoader()
        result = loader.load(dataset_id)
        X, y, label_map = result

        assert isinstance(X, pd.DataFrame), f"Dataset {dataset_id}: X should be DataFrame"
        assert isinstance(y, pd.Series), f"Dataset {dataset_id}: y should be Series"
        assert len(X) == len(y), f"Dataset {dataset_id}: X and y row mismatch"
        assert len(X) > 0, f"Dataset {dataset_id}: Empty dataset"
        assert X.shape[1] > 0, f"Dataset {dataset_id}: No features"

    @pytest.mark.parametrize("dataset_id", NUMERIC_DATASET_IDS)
    def test_no_target_column_in_features(self, dataset_id):
        loader = OfflineDataLoader()
        X, _, _ = loader.load(dataset_id)
        assert "target" not in X.columns

    @pytest.mark.parametrize("dataset_id", NUMERIC_DATASET_IDS)
    def test_no_nan_in_target(self, dataset_id):
        loader = OfflineDataLoader()
        _, y, _ = loader.load(dataset_id)
        assert y.isna().sum() == 0, f"Dataset {dataset_id}: NaNs in target"

    def test_total_dataset_count_matches_local_minimum(self):
        """Local runs only require at least one numeric offline dataset CSV."""
        assert len(NUMERIC_DATASET_IDS) >= 1, "Expected at least one numeric offline dataset CSV"

    def test_local_minimum_dataset_363700_present(self):
        """The lightweight local test profile expects 363700.csv to be available."""
        assert 363700 in NUMERIC_DATASET_IDS, "Expected dataset 363700.csv in offline datasets"


class TestOfflineDataLoaderStringId:
    """Test loading datasets via string IDs (used in dev/toy runs)."""

    def test_ilpd_by_string(self):
        loader = OfflineDataLoader()
        X, y, label_map = loader.load("ilpd", problem_type="binary")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_label_map_present_for_classification_string_id(self):
        loader = OfflineDataLoader()
        _, _, label_map = loader.load("ilpd", problem_type="binary")
        assert label_map is not None


class TestDatasetIntegrity:
    """Cross-dataset consistency checks."""

    def test_datasets_have_distinct_shapes(self):
        loader = OfflineDataLoader()
        shapes = set()
        # Check a sample of 5 datasets to keep this fast
        sample_ids = NUMERIC_DATASET_IDS[:5]
        for ds_id in sample_ids:
            X, _, _ = loader.load(ds_id)
            shapes.add(X.shape)
        # All sampled datasets should have a different shape (not identical copies)
        assert len(shapes) == len(sample_ids), "Some datasets appear to be duplicates"

    def test_benchmark_dataset_simple_run(self):
        """Verify dataset 1590 can be used in a basic task without errors."""
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.iterations = 2
        task.bo_initial_random_iterations = 2
        task.debug = True
        task.result_path = "test_data_loading_simple_run"
        history = run_task(task)
        assert len(history.history) == 2
