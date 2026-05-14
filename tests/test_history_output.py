"""
Tests for RunHistory output: to_dataframe(), get_results(), stored vectors.

Critical for HPC: every task writes artifacts that will be analyzed later.
Vectors (preds, labels) must be loadable after write.
"""
import numpy as np
import pandas as pd
import pytest

from src.history.artifacts import ARTIFACT_ARCHIVE_NAME, build_artifact_key, load_run_artifact
from src.history.data_classes import Fold, Run, Split
from src.history.run_history import RunHistory, MLPlanRunHistory

# Expected columns per benchmark.md specification
EXPECTED_HISTORY_COLUMNS = [
    "iteration",
    "fold",
    "train",
    "val",
    "selection",
    "test",
    "train_time",
    "val_inference_time",
    "test_inference_time",
    "avg_val_score",
    "avg_test_score",
    "avg_selection_score",
    "avg_train_score",
    "early_stopped_makarova",
    "surrogate_mean",
    "surrogate_std",
    "optimizer_suggest_time",
]


def make_run(val, test, train=None, selection=None, iteration=0, n_folds=1, retrain_test=None, surrogate=None):
    """Helper: build a Run with given scores."""
    from src.history.data_classes import Surrogate
    t = train if train is not None else val * 0.9
    folds = [
        Fold(
            fold_id=i,
            scores=Split(train=t, val=val, test=test, selection=selection),
            preds=Split(
                val=np.array([0.1, 0.9]),
                test=np.array([0.2, 0.8]),
                selection=np.array([0.3, 0.7]) if selection is not None else None,
            ),
            labels=Split(
                val=np.array([1, 0]),
                test=np.array([0, 1]),
                selection=np.array([1, 0]) if selection is not None else None,
            ),
            times=Split(train=0.1, val=0.01, test=0.02, selection=0.01 if selection is not None else None),
        )
        for i in range(n_folds)
    ]
    run = Run(
        config={"model": "LGBM", "lr": 0.1, "n_estimators": 100},
        iteration=iteration,
        total_folds=n_folds,
        folds=folds,
        surrogate=surrogate,
        optimizer_suggest_time=0.005,
    )
    if retrain_test is not None:
        run.retrain = Fold(
            fold_id=None,
            scores=Split(train=t, val=val, test=retrain_test),
            preds=Split(train=np.array([0.8, 0.2]), test=np.array([0.3, 0.7])),
            labels=Split(train=np.array([1, 0]), test=np.array([0, 1])),
            times=Split(train=0.15, test=0.02),
        )
    return run


# ---------------------------------------------------------------------------
# to_dataframe() tests
# ---------------------------------------------------------------------------

class TestRunHistoryToDataframe:
    def test_returns_dataframe(self):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, retrain_test=0.35, iteration=0))
        df = history.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_has_all_required_columns(self):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, retrain_test=0.35, iteration=0))
        df = history.to_dataframe()
        for col in EXPECTED_HISTORY_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_one_row_per_fold_evaluation(self):
        history = RunHistory()
        # 3 runs, 2 folds each → 6 rows
        for i in range(3):
            history.add_run(make_run(0.3, 0.4, n_folds=2, retrain_test=0.35, iteration=i))
        df = history.to_dataframe()
        assert len(df) == 6

    def test_iteration_column_values(self):
        history = RunHistory()
        for i in range(3):
            history.add_run(make_run(0.3, 0.4, retrain_test=0.35, iteration=i))
        df = history.to_dataframe()
        assert sorted(df["iteration"].unique()) == [0, 1, 2]

    def test_fold_column_values(self):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, n_folds=5, retrain_test=0.35, iteration=0))
        df = history.to_dataframe()
        assert sorted(df["fold"].unique()) == [0, 1, 2, 3, 4]

    def test_avg_val_score_consistent_across_folds(self):
        """avg_val_score should be the same for all rows of the same run."""
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, n_folds=3, retrain_test=0.35, iteration=0))
        df = history.to_dataframe()
        run_rows = df[df["iteration"] == 0]
        assert run_rows["avg_val_score"].nunique() == 1

    def test_avg_test_score_is_retrain_score(self):
        """When retrain is present, avg_test_score should use the retrained test score."""
        history = RunHistory()
        retrain_test = 0.42
        history.add_run(make_run(0.3, 0.4, retrain_test=retrain_test, iteration=0))
        df = history.to_dataframe()
        assert df["avg_test_score"].iloc[0] == pytest.approx(retrain_test)

    def test_selection_column_none_when_no_selection_set(self):
        """When no selection set is used, selection column should be None/NaN."""
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, selection=None, retrain_test=0.35, iteration=0))
        df = history.to_dataframe()
        assert df["selection"].isna().all()

    def test_selection_column_populated_when_selection_set(self):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, selection=0.32, retrain_test=0.35, iteration=0))
        df = history.to_dataframe()
        assert not df["selection"].isna().all()

    def test_surrogate_columns_none_before_training(self):
        """Runs without surrogate should have NaN surrogate columns."""
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, retrain_test=0.35, iteration=0))  # No surrogate
        df = history.to_dataframe()
        assert df["surrogate_mean"].isna().all()
        assert df["surrogate_std"].isna().all()

    def test_surrogate_columns_populated_when_available(self):
        from src.history.data_classes import Surrogate
        history = RunHistory()
        sur = Surrogate(mean=0.42, std=0.05)
        history.add_run(make_run(0.3, 0.4, retrain_test=0.35, iteration=0, surrogate=sur))
        df = history.to_dataframe()
        assert df["surrogate_mean"].iloc[0] == pytest.approx(0.42)
        assert df["surrogate_std"].iloc[0] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# get_results() and stored artifacts
# ---------------------------------------------------------------------------

class TestRunHistoryGetResults:
    def test_get_results_returns_list(self, tmp_path):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, retrain_test=0.35, iteration=0))
        results = history.get_results(root_path=str(tmp_path) + "/")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_history_csv_in_results(self, tmp_path):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, retrain_test=0.35, iteration=0))
        results = history.get_results(root_path=str(tmp_path) + "/")
        paths = [r[0] for r in results]
        assert any("history.csv" in p for p in paths)

    def test_retrain_preds_npz_in_results(self, tmp_path):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, retrain_test=0.35, iteration=0))
        results = history.get_results(root_path=str(tmp_path) + "/")
        # Retrain preds are now stored inside the aggregated artifacts.npz
        artifact_result = next((r for r in results if ARTIFACT_ARCHIVE_NAME in r[0]), None)
        assert artifact_result is not None
        assert any("retrain_test_preds" in k for k in artifact_result[1].keys())

    def test_val_npz_in_results(self, tmp_path):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, retrain_test=0.35, iteration=0))
        results = history.get_results(root_path=str(tmp_path) + "/")
        paths = [r[0] for r in results]
        assert any(ARTIFACT_ARCHIVE_NAME in p for p in paths)

    def test_test_labels_npz_in_results(self, tmp_path):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, retrain_test=0.35, iteration=0))
        results = history.get_results(root_path=str(tmp_path) + "/")
        paths = [r[0] for r in results]
        assert any(ARTIFACT_ARCHIVE_NAME in p for p in paths)

    def test_selection_npz_in_results_when_used(self, tmp_path):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, selection=0.32, retrain_test=0.35, iteration=0))
        results = history.get_results(root_path=str(tmp_path) + "/")
        paths = [r[0] for r in results]
        assert any(ARTIFACT_ARCHIVE_NAME in p for p in paths)

    def test_no_selection_npz_when_not_used(self, tmp_path):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, selection=None, retrain_test=0.35, iteration=0))
        results = history.get_results(root_path=str(tmp_path) + "/")
        paths = [r[0] for r in results]
        assert not any("sel.npz" in p for p in paths)
        assert any(ARTIFACT_ARCHIVE_NAME in p for p in paths)


class TestStoredVectorsLoadable:
    """Test that written NPZ files can be reloaded and match original data."""

    def _write_results(self, history, root_path):
        """Write all results using the Experiment writer."""
        from src.experiments.experiment import Experiment
        results = history.get_results(root_path=root_path)
        Experiment.write_results_to_file(results)
        return results

    def test_val_predictions_roundtrip(self, tmp_path):
        original_preds = np.array([0.1356, 0.9346, 0.3346, 0.7346], dtype=np.float16)
        original_labels = np.array([1, 0, 0, 1], dtype=np.int16)

        history = RunHistory()
        run = Run(
            config={"model": "LGBM", "lr": 0.1, "n_estimators": 100},
            iteration=0,
            total_folds=1,
            folds=[
                Fold(
                    fold_id=0,
                    scores=Split(train=0.2, val=0.3, test=0.4),
                    preds=Split(val=original_preds, test=original_preds),
                    labels=Split(val=original_labels, test=original_labels),
                    times=Split(train=0.1, val=0.01, test=0.02),
                )
            ],
            retrain=Fold(
                fold_id=None,
                scores=Split(train=0.2, val=0.3, test=0.4),
                preds=Split(train=original_preds, test=original_preds),
                labels=Split(train=original_labels, test=original_labels),
                times=Split(train=0.1, test=0.02),
            ),
        )
        history.add_run(run)

        root = str(tmp_path) + "/"
        self._write_results(history, root)

        archive_path = tmp_path / ARTIFACT_ARCHIVE_NAME
        assert archive_path.exists(), f"Expected artifacts archive at {archive_path}"
        with np.load(str(archive_path)) as loaded:
            assert build_artifact_key(0, 0, "val_preds") in loaded
            assert build_artifact_key(0, 0, "val_labels") in loaded

        loaded_preds = load_run_artifact(tmp_path, iteration=0, fold=0, name="val_preds")
        loaded_labels = load_run_artifact(tmp_path, iteration=0, fold=0, name="val_labels")
        assert np.array_equal(loaded_preds, original_preds)
        assert np.array_equal(loaded_labels, original_labels)

    def test_history_csv_has_correct_columns(self, tmp_path):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, retrain_test=0.35, iteration=0))

        root = str(tmp_path) + "/"
        results = history.get_results(root_path=root)
        from src.experiments.experiment import Experiment
        Experiment.write_results_to_file(results)

        csv_path = tmp_path / "history.csv"
        assert csv_path.exists()
        df = pd.read_csv(str(csv_path))
        for col in EXPECTED_HISTORY_COLUMNS:
            assert col in df.columns, f"history.csv missing column: {col}"

    def test_retrain_preds_loadable(self, tmp_path):
        history = RunHistory()
        run = make_run(0.3, 0.4, retrain_test=0.35, iteration=0)
        history.add_run(run)

        root = str(tmp_path) + "/"
        results = history.get_results(root_path=root)
        from src.experiments.experiment import Experiment
        Experiment.write_results_to_file(results)

        # Retrain preds are now inside the aggregated artifacts.npz
        archive_path = tmp_path / ARTIFACT_ARCHIVE_NAME
        assert archive_path.exists()
        with np.load(str(archive_path)) as loaded:
            assert "iter_0/retrain_test_preds" in loaded

    def test_config_yaml_written_per_run(self, tmp_path):
        history = RunHistory()
        run = make_run(0.3, 0.4, retrain_test=0.35, iteration=0)
        run.config = {"model": "LGBM", "lr": 0.1, "n_estimators": 100}
        history.add_run(run)

        root = str(tmp_path) + "/"
        results = history.get_results(root_path=root)
        from src.experiments.experiment import Experiment
        Experiment.write_results_to_file(results)

        # Configs are now aggregated into a single configs.yaml at root level
        import yaml
        config_path = tmp_path / "configs.yaml"
        assert config_path.exists()
        with open(config_path) as f:
            loaded_configs = yaml.safe_load(f)
        assert loaded_configs["iter_0"]["model"] == "LGBM"

    def test_test_labels_loadable(self, tmp_path):
        history = RunHistory()
        history.add_run(make_run(0.3, 0.4, retrain_test=0.35, iteration=0))

        root = str(tmp_path) + "/"
        results = history.get_results(root_path=root)
        from src.experiments.experiment import Experiment
        Experiment.write_results_to_file(results)

        archive_path = tmp_path / ARTIFACT_ARCHIVE_NAME
        assert archive_path.exists()
        loaded = load_run_artifact(tmp_path, iteration=None, fold=None, name="test_labels")
        assert np.array_equal(loaded, np.array([0, 1]))

    def test_load_run_artifact_supports_legacy_layout(self, tmp_path):
        legacy_dir = tmp_path / "0" / "0"
        legacy_dir.mkdir(parents=True)
        np.savez_compressed(legacy_dir / "val.npz", labels=np.array([1, 0]), preds=np.array([0.2, 0.8]))

        loaded_preds = load_run_artifact(tmp_path, iteration=0, fold=0, name="val_preds")
        loaded_labels = load_run_artifact(tmp_path, iteration=0, fold=0, name="val_labels")

        assert np.array_equal(loaded_preds, np.array([0.2, 0.8]))
        assert np.array_equal(loaded_labels, np.array([1, 0]))


# ---------------------------------------------------------------------------
# End-to-end: run_task → to_dataframe → stored artifacts
# ---------------------------------------------------------------------------

class TestEndToEndHistoryOutput:
    def test_run_task_produces_correct_csv_structure(self, unique_result_path, tmp_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig
        from src.experiments.experiment import Experiment

        task = DefaultTaskConfig()
        task.iterations = 3
        task.bo_initial_random_iterations = 2
        task.evaluation.resampling = "cv"
        task.evaluation.n_folds = 3
        task.evaluation.selection_size = 0
        task.metric = "roc_auc"
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        df = history.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        for col in EXPECTED_HISTORY_COLUMNS:
            assert col in df.columns

        # 3 runs × 3 folds = 9 rows
        assert len(df) == 9

    def test_run_task_with_selection_set_has_selection_column(self, unique_result_path):
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig

        task = DefaultTaskConfig()
        task.iterations = 3
        task.bo_initial_random_iterations = 2
        task.evaluation.resampling = "cv"
        task.evaluation.n_folds = 3
        task.evaluation.selection_size = 1 / 6
        task.metric = "roc_auc"
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        df = history.to_dataframe()

        assert "selection" in df.columns
        assert not df["selection"].isna().all(), "Selection scores should be populated"
        assert df["avg_selection_score"].notna().all()

    def test_full_result_write_and_reload(self, unique_result_path, tmp_path):
        """Complete pipeline: run_task → get_results() → write → reload CSV."""
        from src.experiments.task.task import run_task
        from src.experiments.task.task_config import DefaultTaskConfig
        from src.experiments.experiment import Experiment

        task = DefaultTaskConfig()
        task.iterations = 3
        task.bo_initial_random_iterations = 2
        task.debug = True
        task.result_path = unique_result_path

        history = run_task(task)
        root = str(tmp_path) + "/"
        results = history.get_results(root_path=root)
        Experiment.write_results_to_file(results)

        # Reload and verify
        csv_path = tmp_path / "history.csv"
        assert csv_path.exists()
        df = pd.read_csv(str(csv_path))
        assert len(df) == 3  # 3 iterations × 1 holdout fold
        assert "val" in df.columns
        assert "avg_test_score" in df.columns
