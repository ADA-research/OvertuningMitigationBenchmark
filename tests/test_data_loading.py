"""
Tests for offline data loading — every benchmark dataset must load correctly.
"""
import os
from pathlib import Path

import pandas as pd
import pytest

from src.datasets.offline_dataloader import OfflineDataLoader
from src.datasets.dataset_config import DATASET_INFO, DatasetConfig

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


class TestDatasetConfig:
    """Tests for DATASET_INFO dictionary and DatasetConfig class."""

    def test_all_offline_datasets_in_dataset_info(self):
        """Every offline dataset on disk should have an entry in DATASET_INFO."""
        missing = [ds_id for ds_id in NUMERIC_DATASET_IDS if ds_id not in DATASET_INFO]
        assert missing == [], f"Datasets missing from DATASET_INFO: {missing}"

    def test_dataset_info_has_51_entries(self):
        """The benchmark uses 51 datasets."""
        assert len(DATASET_INFO) == 51

    def test_all_entries_are_valid_tuples(self):
        """Each entry must be a 3-tuple of (int, int, str)."""
        for ds_id, info in DATASET_INFO.items():
            assert len(info) == 3, f"Dataset {ds_id}: expected 3-tuple, got {len(info)}"
            n_samples, n_features, problem_type = info
            assert isinstance(n_samples, int), f"Dataset {ds_id}: n_samples not int"
            assert isinstance(n_features, int), f"Dataset {ds_id}: n_features not int"
            assert problem_type in ("binary", "multiclass", "regression"), (
                f"Dataset {ds_id}: invalid problem_type '{problem_type}'"
            )

    def test_problem_type_counts(self):
        """Sanity check the distribution of problem types."""
        types = [info[2] for info in DATASET_INFO.values()]
        assert types.count("binary") > 0
        assert types.count("multiclass") > 0
        assert types.count("regression") > 0

    @pytest.mark.parametrize("dataset_id", NUMERIC_DATASET_IDS)
    def test_sample_count_matches_loaded_data(self, dataset_id):
        """DATASET_INFO sample count should match the actual loaded dataset."""
        if dataset_id not in DATASET_INFO:
            pytest.skip(f"Dataset {dataset_id} not in DATASET_INFO")
        loader = OfflineDataLoader()
        X, y, _ = loader.load(dataset_id)
        expected_samples = DATASET_INFO[dataset_id][0]
        assert len(X) == expected_samples, (
            f"Dataset {dataset_id}: DATASET_INFO says {expected_samples} samples, "
            f"but loaded {len(X)}"
        )

    @pytest.mark.parametrize("dataset_id", NUMERIC_DATASET_IDS)
    def test_feature_count_matches_loaded_data(self, dataset_id):
        """DATASET_INFO feature count should match the actual loaded dataset."""
        if dataset_id not in DATASET_INFO:
            pytest.skip(f"Dataset {dataset_id} not in DATASET_INFO")
        loader = OfflineDataLoader()
        X, _, _ = loader.load(dataset_id)
        expected_features = DATASET_INFO[dataset_id][1]
        assert X.shape[1] == expected_features, (
            f"Dataset {dataset_id}: DATASET_INFO says {expected_features} features, "
            f"but loaded {X.shape[1]}"
        )

    def test_dataset_config_construction(self):
        """DatasetConfig should initialize correctly from a valid dataset ID."""
        cfg = DatasetConfig(363621)
        assert cfg.n_samples == 748
        assert cfg.problem_type == "binary"

    def test_dataset_config_large_flag(self):
        """Datasets with >2500 samples should be flagged as large."""
        small = DatasetConfig(363621)  # 748 samples
        large = DatasetConfig(363613)  # 32769 samples
        assert small.is_large() is False
        assert large.is_large() is True

    def test_dataset_config_get_problem_type(self):
        """get_problem_type should return the correct type string."""
        assert DatasetConfig(363698).get_problem_type() == "regression"
        assert DatasetConfig(363613).get_problem_type() == "binary"
        assert DatasetConfig(363614).get_problem_type() == "multiclass"

    def test_dataset_config_unknown_id_raises(self):
        """An unknown dataset ID should raise an error."""
        with pytest.raises((KeyError, TypeError)):
            DatasetConfig(999999)
