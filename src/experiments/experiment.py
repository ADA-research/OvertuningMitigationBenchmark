import os
import time
import gc
from collections.abc import Mapping

from typing import List, Tuple, Any
from datetime import datetime
import traceback

import pandas as pd
import numpy as np

from joblib import Parallel, delayed, parallel_config

from src.experiments.task.task_config import TaskConfig
from src.experiments.task.task import run_task
from src.utils.google_cloud_storage import GCSManager


class Experiment:
    """
    Object taking care of a list of tasks and the structure of running and storing them.
    HPC-safe: supports sequential and parallel execution.
    """

    def __init__(
        self,
        experiment_name: str,
        list_of_tasks: List[TaskConfig],
        continue_experiment: bool = False,
        output_dir: str = ".",
    ):
        self.name = experiment_name
        self.tasks: List[TaskConfig] = list_of_tasks
        self.results_path = None
        self.results_root = os.path.join(output_dir, "results")

        if not continue_experiment:
            self.results_path = os.path.join(
                self.results_root,
                f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            self.results_path = os.path.join(self.results_root, experiment_name)

        os.makedirs(self.results_path, exist_ok=True)

    @staticmethod
    def write_results_to_file(paths_and_objects: List[Tuple[str, Any]]):
        """
        Writes the results to the local filesystem.
        """
        for path, obj in paths_and_objects:

            os.makedirs(os.path.dirname(path), exist_ok=True)

            if isinstance(obj, pd.DataFrame):
                obj.to_csv(path, index=False)

            elif path.endswith(".yaml"):
                with open(path, "w") as f:
                    f.write(obj)

            elif path.endswith(".npz"):
                if isinstance(obj, Mapping):
                    np.savez_compressed(path, **obj)
                elif not isinstance(obj, tuple):
                    np.savez_compressed(path, preds=obj)
                else:
                    np.savez_compressed(path, labels=obj[0], preds=obj[1])

            else:
                raise ValueError("Unknown file type in", path)

    @staticmethod
    def _upload_file_with_manager(
        manager: GCSManager,
        local_path: str,
        upload_vectors: bool = True,
        results_root: str = "results",
    ):
        """
        Uploads a file to GCS using a worker-local storage manager.
        """
        try:
            relative_path = os.path.relpath(local_path, results_root)
        except ValueError:
            relative_path = local_path

        # If upload_vectors, we upload all artifacts to GCS
        # Else, we only upload task-wide config and results
        if upload_vectors or local_path.endswith(".csv") or local_path.endswith("task_config.yaml"):
            manager.upload_blob(
                source_file_name=local_path,
                destination_blob_name=relative_path
            )

    def _run_single_task(self, task: TaskConfig):
        """
        Isolated execution of ONE task.
        Safe for joblib multiprocessing (no unpickleable state).
        """
        # Disable torch lightning logging
        import logging

        logging.disable(logging.CRITICAL)

        remote_storage_manager = None

        if task.store_results_google_cloud:
            remote_storage_manager = GCSManager()

        historical_path = f"{self.results_path}/{task.result_path}/history.pkl"
        if os.path.exists(historical_path):
            return f"SKIPPED {task.result_path}"

        try:
            task_result_path = f"{self.results_path}/{task.result_path}/"
            task_config_path = f"{task_result_path}/task_config.yaml"
            task_history_path = f"{task_result_path}/history.csv"
            time_elapsed_path = f"{task_result_path}/time_elapsed.txt"

            if os.path.exists(task_config_path) and os.path.exists(task_history_path):
                return "SKIPPING - RESULTS ALREADY EXIST"

            # Run task
            time_start = time.perf_counter()
            task_history = run_task(task)
            time_elapsed = time.perf_counter() - time_start

            os.makedirs(task_result_path, exist_ok=False)

            # --- 1. Save Task Configuration ---
            task.to_yaml(task_config_path)

            # --- 2. Save Detailed Results ---
            results = task_history.get_results(root_path=task_result_path)
            self.write_results_to_file(results)

            # Write total task time to file
            with open(time_elapsed_path, "w") as f:
                f.write(str(time_elapsed))

            # After task completion
            gc.collect()

            print(f"Task completed in {round(time_elapsed)}s and saved locally to:", task.result_path)

            # --- 3. Upload to GCS ---
            if task.store_results_google_cloud and remote_storage_manager:
                print(f"Uploading results for {task.result_path} to GCS...")

                results_root = getattr(self, "results_root", "results")

                self._upload_file_with_manager(
                    remote_storage_manager,
                    task_config_path,
                    results_root=results_root,
                )

                for path, _ in results:
                    self._upload_file_with_manager(
                        remote_storage_manager,
                        path,
                        task.store_vectors_google_cloud,
                        results_root,
                    )

            return f"DONE {task.result_path}"

        except Exception:
            print("Task", task.result_path, "failed:")
            traceback.print_exc()
            return f"FAILED {task.result_path}"

    def run_sequential(self):
        """
        Run all tasks strictly sequentially (no multiprocessing).
        """
        print(f"Running {len(self.tasks)} tasks sequentially")

        results = []

        for task in self.tasks:
            result = self._run_single_task(task)
            results.append(result)

        print("Sequential experiment finished:")
        for r in results:
            print(r)

        return results

    def run(self, n_jobs: int = 1):
        """
        - If n_jobs == 1 → sequential execution
        - If n_jobs > 1 → parallel HPC-safe execution
        """

        if n_jobs == 1:
            return self.run_sequential()

        print(f"Running {len(self.tasks)} tasks using {n_jobs} parallel workers")

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._run_single_task)(task)
            for task in self.tasks
        )

        print("Parallel experiment finished:")
        for r in results:
            print(r)

        return results
