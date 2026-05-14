import pickle
from collections import defaultdict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.history.data_classes import Run
from src.history.artifacts import build_run_artifact_archive


class RunHistory:
    def __init__(self):
        self.history: List[Run] = []
        self.incumbent: Run | None = None
        self.incumbent_folds: int = 0

        # Keep track of best previous incumbent
        self.best_test_incumbent: Run | None = None

        # Track number of fold evaluations (For inference)
        self.number_fold_evaluations: int = 0
        self.trajectory: dict[str, List] = {"n_fold_evals": [], "n_configurations": [], "iteration": [],
                                            "is_fully_evaluated": [], "val_score": [], "test_score": [],
                                            "overtuning": [], "overfitting": []}
        self.runs: List[int] = []
        self.fold_savings: float = 0

    def __eq__(self, other):
        # Never equal to a different object
        if not isinstance(other, RunHistory):
            return False

        # Not equal if histories have different number of runs
        if len(self.history) != len(other.history):
            return False

        # Test if all runs are equal
        for run_self, run_other in zip(self.history, other.history):
            if run_self.average_val_score() != run_other.average_val_score():
                print(run_self.average_val_score(), run_other.average_val_score())
                return False
            if run_self.average_test_score() != run_other.average_test_score():
                print(run_self.average_test_score(), run_other.average_test_score())
                return False
            if dict(run_self.config) != dict(run_other.config):
                print("configs")
                return False

            for i in range(len(run_self.folds)):
                fold_self = run_self.folds[i]
                fold_other = run_other.folds[i]

                if fold_self.scores.val != fold_other.scores.val:
                    return False
                if fold_self.scores.test != fold_other.scores.test:
                    return False
                if fold_self.scores.train != fold_other.scores.train:
                    return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def meta_overfitting(self):
        # Compute the meta-overfitting by the incumbent
        return -(self.incumbent.average_val_score() - self.incumbent.average_test_score())

    def overtuning(self):
        # Clip overtuning at 0 if we are not overtuning
        return max(0.0, -(self.best_test_incumbent.average_test_score() - self.incumbent.average_test_score()))

    def update_after_fold_setbased(self, run: Run):
        self.number_fold_evaluations += 1

        if self.incumbent is None:
            self.incumbent = run
            self.best_test_incumbent = run
        else:
            # Only consider updating if run has at least as many folds as incumbent
            if len(run.folds) >= len(self.incumbent.folds):
                if (((run.average_val_score() < self.incumbent.average_val_score() or len(run.folds) > len(
                        self.incumbent.folds)) and not run.early_stopped)):
                    self.incumbent = run

                    # Update best test incumbent if needed
                    if run.average_test_score() < self.best_test_incumbent.average_test_score():
                        self.best_test_incumbent = run

        # Add trajectory point
        self.update_trajectory(self.incumbent, self.number_fold_evaluations)

    def add_run_setbased(self, run):

        if run.iteration in self.runs:
            # Update existing run in history (for elites)
            for i, existing_run in enumerate(self.history):
                if existing_run.iteration == run.iteration:
                    self.history[i] = run
                    break
            return

        # Add new run to history
        self.runs.append(run.iteration)
        self.history.append(run)

    def add_run(self, run):

        self.runs.append(run.iteration)

        self.history.append(run)
        self.number_fold_evaluations += len(run.folds)

        if self.incumbent is not None:
            self.number_fold_evaluations += len(
                self.incumbent.folds) - self.incumbent_folds  # If the incumbent was only partially evaluated we need to add the difference

        # Update incumbent
        if self.incumbent is None or (
                run.average_val_score() < self.incumbent.average_val_score() and not run.early_stopped) or self.incumbent.early_stopped:
            self.incumbent = run

            # If run is the new incumbent we check if it is also better on test, otherwise we are overtuning
            if self.best_test_incumbent is None or run.average_test_score() < self.best_test_incumbent.average_test_score():
                self.best_test_incumbent = run

        self.incumbent_folds = len(self.incumbent.folds)

        self.update_trajectory(self.incumbent, self.number_fold_evaluations)

    def update_trajectory(self, run, num_folds: int = 0):

        self.trajectory["n_fold_evals"].append(num_folds)
        self.trajectory["n_configurations"].append(len(self.history))
        self.trajectory["iteration"].append(run.iteration)
        self.trajectory["is_fully_evaluated"].append(run.fully_evaluated)
        self.trajectory["val_score"].append(run.average_val_score())
        self.trajectory["test_score"].append(run.average_test_score())
        self.trajectory["overtuning"].append(self.overtuning())
        self.trajectory["overfitting"].append(self.meta_overfitting())

    def make_incumbent(self, run: Run):

        self.incumbent = run
        # If run is the new incumbent we check if it is also better on test, otherwise we are overtuning
        if self.best_test_incumbent is None or run.average_test_score() < self.best_test_incumbent.average_test_score():  # Changed the logic to smaller is better
            self.best_test_incumbent = run

    def to_pickle(self):
        return pickle.dumps(self)

    def to_pickle_file(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def get_results(self, root_path=None):
        """
        Returns tuples of relative paths and objects to be stored.

        - For each run
        -- Retrained test preds/labels

        - For each fold
        -- Val preds/labels
        -- Sel preds/labels

        We return labels and preds as tuple for every instance
        """
        root = "" if root_path is None else root_path

        # Add history csv to results
        results: List[Tuple] = [
            (f"{root}history.csv", self.to_dataframe())
        ]

        results += self._list_results(root)

        return results

    def _list_results(self, root):
        """List serialized artifacts using one aggregated archive per task run."""
        if not self.history:
            return []
        
        array_archive, configs = build_run_artifact_archive(self.history)

        results = [
            (f"{root}/artifacts.npz", array_archive),
            ( f"{root}/configs.yaml", yaml.dump(configs, default_flow_style=False, sort_keys=False))
        ]

        return results


    def to_dataframe(self):
        """
        Return a DataFrame with one row per fold evaluation in the HPO process.
        """
        rows = []
        for run in self.history:
            folds = [f for f in run.folds if f.scores]
            avg = lambda arr: sum(arr) / len(arr) if arr else None

            avg_val = avg([f.scores.val for f in folds])
            avg_test = run.retrain.scores.test if run.retrain else avg([f.scores.test for f in folds])

            avg_sel = avg([getattr(f.scores, "selection", None) for f in folds if
                           getattr(f.scores, "selection", None) is not None])

            avg_train = avg([f.scores.train for f in folds])

            early_stopped_makarova = run.early_stopped_makarova

            surrogate_mean = run.surrogate.mean if getattr(run, 'surrogate', None) else None
            surrogate_std = run.surrogate.std if getattr(run, 'surrogate', None) else None
            surrogate_acq = run.surrogate.acquisition if getattr(run, 'surrogate', None) else None

            for i_fold, fold in enumerate(folds):

                row = {
                    "iteration": run.iteration,
                    "fold": i_fold,
                    "train": fold.scores.train,
                    "val": fold.scores.val,
                    "selection": fold.scores.selection,
                    "test": fold.scores.test,
                    "train_time": fold.times.train,
                    "val_inference_time": fold.times.val,
                    "test_inference_time": fold.times.test,
                    "avg_val_score": avg_val,
                    "avg_test_score": avg_test,
                    "avg_selection_score": avg_sel,
                    "avg_train_score": avg_train,
                    "early_stopped_makarova": early_stopped_makarova,
                    "surrogate_mean": surrogate_mean,
                    "surrogate_std": surrogate_std,
                    "surrogate_acq": surrogate_acq,
                    "optimizer_suggest_time": run.optimizer_suggest_time
                }

                rows.append(row)

        return pd.DataFrame(rows)



class MLPlanRunHistory(RunHistory):
    def __init__(self):
        super().__init__()

        self.combined_dataframe_both_phases: pd.DataFrame | None = None
        self.phase_two_history = None

    def add_mlplan_phase_two_history(self,
            run_history_phase2: RunHistory,
            iteration_map_phase_two_to_phase_one: dict,
    ):
        """
        Return a DataFrame with one row per fold evaluation in the HPO process
        """

        self.phase_two_history = run_history_phase2

        # Check if we retrained in phase one
        if all(run.retrain is not None for run in self.history):
            # Copy retrain objects from phase one to phase two
            for i in range(len(run_history_phase2.history)):
                run_history_phase2.history[i].retrain = self.history[
                    iteration_map_phase_two_to_phase_one[run_history_phase2.history[i].iteration]].retrain

        df_phase1 = self.to_dataframe()
        df_phase2 = run_history_phase2.to_dataframe()

        # Compute 75 percentile of scores from phase 2 folds
        df_phase2["75_percentile"] = df_phase2.groupby("iteration")["val"].transform(lambda x: np.percentile(x, 75))

        # Select runs from phase one history corresponding to phase 2 configs and store average val score
        df_phase2["phase1_avg_val_score"] = [self.history[iteration_map_phase_two_to_phase_one[i]].average_val_score()
                                             for i in df_phase2["iteration"]]

        # Compute the MLPlan score, the average of the phase one val score and the 75 percentile of phase two val scores
        df_phase2["mlplan_score"] = (df_phase2["75_percentile"] + df_phase2["phase1_avg_val_score"]) / 2

        # Drop helper columns
        df_phase2 = df_phase2.drop(columns=["75_percentile", "phase1_avg_val_score"])

        # Add column to phase one history for merge
        df_phase1["mlplan_score"] = None

        # Merge dataframes from both phases
        self.combined_dataframe_both_phases = pd.concat([df_phase1, df_phase2]).reset_index(drop=True)

        pass

    def get_results(self, root_path=None):
        """
        Overwrites get_results of RunHistory by uploading
        """
        root = "" if root_path is None else root_path

        # Add history csv to results
        results: List[Tuple] = [
            (f"{root}history_phase_one.csv", self.to_dataframe()),
            (f"{root}history_phase_two.csv", self.phase_two_history.to_dataframe()),
            (f"{root}history.csv", self.combined_dataframe_both_phases)
        ]

        # Store Phase 1 results with phase1/ prefix
        results += self._list_results(f"{root}phase1/")

        # Store Phase 2 results with phase2/ prefix
        if self.phase_two_history is not None:
            results += self.phase_two_history._list_results(f"{root}phase2/")

        return results



