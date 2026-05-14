"""
Tests for Experiment orchestration:
- run_sequential() executes tasks
- Failed task returns "FAILED ..." without stopping others
- Artifacts are created: task_config.yaml, history.csv
- Continue-experiment skip logic works correctly
- Results cleanup
"""
import os
import numpy as np
import shutil
import pytest
from pathlib import Path

from src.history.artifacts import load_run_artifact


def _make_quick_task(result_path_suffix, iterations=3, debug=True):
    from src.experiments.task.task_config import DefaultTaskConfig
    task = DefaultTaskConfig()
    task.iterations = iterations
    task.bo_initial_random_iterations = 2
    task.debug = debug
    task.store_results_google_cloud = False  # No cloud upload in tests
    task.store_vectors_google_cloud = False
    task.result_path = result_path_suffix
    return task


def _make_experiment(name, tasks, results_path):
    """Build an Experiment with a custom results_path (bypasses timestamp)."""
    from src.experiments.experiment import Experiment
    exp = Experiment.__new__(Experiment)
    exp.name = name
    exp.tasks = tasks
    exp.results_path = str(results_path)
    os.makedirs(exp.results_path, exist_ok=True)
    return exp


class TestExperimentSequential:
    def test_run_sequential_executes_all_tasks(self, tmp_path):
        tasks = [
            _make_quick_task(f"task_{i}") for i in range(3)
        ]
        exp = _make_experiment("test_sequential", tasks, tmp_path / "results")
        results = exp.run_sequential()
        assert len(results) == 3

    def test_run_sequential_returns_done_status(self, tmp_path):
        tasks = [_make_quick_task("done_task")]
        exp = _make_experiment("test_done", tasks, tmp_path / "results")
        results = exp.run_sequential()
        assert results[0].startswith("DONE")

    def test_failed_task_does_not_stop_others(self, tmp_path):
        """A task that raises an exception should return FAILED, not abort the run."""
        from src.experiments.task.task_config import DefaultTaskConfig
        import copy

        good_task = _make_quick_task("good_task")

        # Create a bad task by using an invalid optimizer
        bad_task = copy.deepcopy(good_task)
        bad_task.optimizer = "nonexistent_optimizer_xyz"
        bad_task.result_path = "bad_task"

        another_good = _make_quick_task("another_good_task")

        exp = _make_experiment("test_failure", [good_task, bad_task, another_good], tmp_path / "results")
        results = exp.run_sequential()

        assert len(results) == 3
        assert results[0].startswith("DONE")
        assert results[1].startswith("FAILED")
        assert results[2].startswith("DONE")

    def test_failed_task_message_contains_result_path(self, tmp_path):
        import copy
        bad_task = _make_quick_task("my_bad_task")
        bad_task.optimizer = "nonexistent_xyz"

        exp = _make_experiment("test_fail_msg", [bad_task], tmp_path / "results")
        results = exp.run_sequential()
        assert "my_bad_task" in results[0]


class TestExperimentArtifacts:
    def test_task_config_yaml_created(self, tmp_path):
        task = _make_quick_task("artifact_task")
        exp = _make_experiment("test_artifacts", [task], tmp_path / "results")
        exp.run_sequential()

        config_path = tmp_path / "results" / "artifact_task" / "task_config.yaml"
        assert config_path.exists(), f"task_config.yaml not found at {config_path}"

    def test_history_csv_created(self, tmp_path):
        task = _make_quick_task("history_task")
        exp = _make_experiment("test_history", [task], tmp_path / "results")
        exp.run_sequential()

        history_path = tmp_path / "results" / "history_task" / "history.csv"
        assert history_path.exists(), f"history.csv not found at {history_path}"

    def test_history_csv_has_data(self, tmp_path):
        task = _make_quick_task("csv_check_task")
        exp = _make_experiment("test_csv", [task], tmp_path / "results")
        exp.run_sequential()

        import pandas as pd
        history_path = tmp_path / "results" / "csv_check_task" / "history.csv"
        df = pd.read_csv(str(history_path))
        assert len(df) == 3  # 3 iterations

    def test_task_config_yaml_is_valid(self, tmp_path):
        task = _make_quick_task("yaml_check_task")
        exp = _make_experiment("test_yaml", [task], tmp_path / "results")
        exp.run_sequential()

        import yaml
        config_path = tmp_path / "results" / "yaml_check_task" / "task_config.yaml"
        with open(config_path) as f:
            loaded = yaml.safe_load(f)
        assert isinstance(loaded, dict)
        assert "optimizer" in loaded

    def test_retrain_preds_npz_created(self, tmp_path):
        task = _make_quick_task("npz_task")
        exp = _make_experiment("test_npz", [task], tmp_path / "results")
        exp.run_sequential()


        artifact = np.load(tmp_path / "results" / "npz_task" / "artifacts.npz")

        assert "iter_0/retrain_test_preds" in artifact.files, "retrain_test_preds not found in artifacts.npz"
        

    def test_val_npz_created_and_loadable(self, tmp_path):
        task = _make_quick_task("val_npz_task")
        exp = _make_experiment("test_val_npz", [task], tmp_path / "results")
        exp.run_sequential()

        artifact_path = tmp_path / "results" / "val_npz_task" / "artifacts.npz"
        assert artifact_path.exists()
        loaded = load_run_artifact(tmp_path / "results" / "val_npz_task", iteration=0, fold=0, name="val_preds")
        assert loaded is not None
        assert not (tmp_path / "results" / "val_npz_task" / "0" / "0" / "val.npz").exists()

    def test_multiple_tasks_create_separate_directories(self, tmp_path):
        tasks = [_make_quick_task(f"task_{i}") for i in range(3)]
        exp = _make_experiment("test_multi_dirs", tasks, tmp_path / "results")
        exp.run_sequential()

        for i in range(3):
            assert (tmp_path / "results" / f"task_{i}").exists()


class TestExperimentSkipLogic:
    def test_skip_if_history_pkl_exists(self, tmp_path):
        """Task should be skipped if history.pkl already exists in its result path."""
        task = _make_quick_task("skip_task")
        results_dir = tmp_path / "results"
        exp = _make_experiment("test_skip", [task], results_dir)

        # Pre-create history.pkl to simulate already-done task
        task_dir = results_dir / "skip_task"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "history.pkl").write_bytes(b"")

        results = exp.run_sequential()
        assert results[0].startswith("SKIPPED")

    def test_already_run_task_not_rerun(self, tmp_path):
        """Running an experiment twice should skip the second run."""
        task = _make_quick_task("idempotent_task")
        exp = _make_experiment("test_idempotent", [task], tmp_path / "results")

        results1 = exp.run_sequential()
        assert results1[0].startswith("DONE")

        # Simulate what happens if history.pkl exists (Experiment marks completion with pkl)
        # The skip check looks for history.pkl specifically
        task_dir = tmp_path / "results" / "idempotent_task"
        if (task_dir / "history.pkl").exists():
            results2 = exp.run_sequential()
            assert results2[0].startswith("SKIPPED")


class TestWriteResultsToFile:
    def test_csv_written_correctly(self, tmp_path):
        import pandas as pd
        from src.experiments.experiment import Experiment

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = str(tmp_path / "test.csv")
        Experiment.write_results_to_file([(path, df)])

        loaded = pd.read_csv(path)
        assert list(loaded["a"]) == [1, 2, 3]

    def test_yaml_written_correctly(self, tmp_path):
        import yaml
        from src.experiments.experiment import Experiment

        yaml_content = yaml.dump({"key": "value", "num": 42})
        path = str(tmp_path / "test.yaml")
        Experiment.write_results_to_file([(path, yaml_content)])

        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["key"] == "value"
        assert loaded["num"] == 42

    def test_npz_written_and_loadable(self, tmp_path):
        import numpy as np
        from src.experiments.experiment import Experiment

        arr = np.array([0.1, 0.9, 0.5], dtype=np.float16)
        path = str(tmp_path / "test.npz")
        Experiment.write_results_to_file([(path, arr)])

        loaded = np.load(path)
        assert "preds" in loaded

    def test_npz_tuple_written_and_loadable(self, tmp_path):
        import numpy as np
        from src.experiments.experiment import Experiment

        labels = np.array([1, 0, 1], dtype=np.int16)
        preds = np.array([0.9, 0.1, 0.8], dtype=np.float16)
        path = str(tmp_path / "test_tuple.npz")
        Experiment.write_results_to_file([(path, (labels, preds))])

        loaded = np.load(path)
        assert "labels" in loaded
        assert "preds" in loaded
        assert np.array_equal(loaded["labels"], labels)
        assert np.array_equal(loaded["preds"], preds)

    def test_npz_mapping_written_and_loadable(self, tmp_path):
        import numpy as np
        from src.experiments.experiment import Experiment

        path = str(tmp_path / "artifact.npz")
        arrays = {
            "test_labels": np.array([1, 0], dtype=np.int16),
            "iter_0/fold_0/val_preds": np.array([0.9, 0.1], dtype=np.float16),
        }

        Experiment.write_results_to_file([(path, arrays)])

        loaded = np.load(path)
        assert "test_labels" in loaded
        assert "iter_0/fold_0/val_preds" in loaded

    def test_creates_parent_directories(self, tmp_path):
        import pandas as pd
        from src.experiments.experiment import Experiment

        df = pd.DataFrame({"x": [1]})
        nested_path = str(tmp_path / "a" / "b" / "c" / "test.csv")
        Experiment.write_results_to_file([(nested_path, df)])
        assert (tmp_path / "a" / "b" / "c" / "test.csv").exists()


class TestExperimentOutputDir:
    def test_new_experiment_uses_output_dir_results_folder(self, tmp_path):
        from src.experiments.experiment import Experiment

        exp = Experiment("my_exp", [], output_dir=str(tmp_path))

        assert str(tmp_path / "results") == exp.results_root
        assert str(tmp_path / "results") in exp.results_path
        assert os.path.isdir(exp.results_path)

    def test_continue_experiment_uses_stable_results_path(self, tmp_path):
        from src.experiments.experiment import Experiment

        exp = Experiment("my_exp", [], continue_experiment=True, output_dir=str(tmp_path))

        assert exp.results_path == str(tmp_path / "results" / "my_exp")
        assert os.path.isdir(exp.results_path)
