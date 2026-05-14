import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from src.experiments.task.task_config import OuterEvaluationConfig


class TaskDataSplitter:
    def __init__(
            self,
            outer_evaluation_config: OuterEvaluationConfig,
            random_state: int
    ):
        self.outer_evaluation = outer_evaluation_config

        # Data splitting reproducibility is crucial for interpretability
        self.random_state = random_state

    def make_outer_split(self, X: pd.DataFrame, y: pd.Series):
        """
        Create the outer train/test split based on outer evaluation config.

        For holdout: splits data according to train_size
        For CV: generates all folds for all repeats, then selects the specific fold/repeat
               specified in the config to ensure consistent splits across tasks

        Args:
            X: Features (pandas DataFrame)
            y: Labels (pandas Series)

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """

        # Holdout outer split
        if self.outer_evaluation.resampling == "holdout":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                train_size=self.outer_evaluation.train_size,
                random_state=self.random_state,
            )

            return X_train, X_test, y_train, y_test

        # Cross-validation outer split
        elif self.outer_evaluation.resampling == "cv":
            # Use RepeatedKFold to ensure consistent splitting across all tasks
            # All tasks with same n_folds, n_repeats, and random_state will generate
            # the same fold splits, then select their specific fold/repeat combination

            if self.outer_evaluation.n_repeats > 1:
                # Use RepeatedKFold for multiple repeats
                cv_splitter = RepeatedKFold(
                    n_splits=self.outer_evaluation.n_folds,
                    n_repeats=self.outer_evaluation.n_repeats,
                    random_state=self.random_state
                )
            else:
                # Use KFold for single repeat
                cv_splitter = KFold(
                    n_splits=self.outer_evaluation.n_folds,
                    shuffle=True,
                    random_state=self.random_state
                )

            # Generate all splits
            all_splits = list(cv_splitter.split(X, y))

            # Calculate the index for the desired fold/repeat combination
            split_idx = self.outer_evaluation.repeat * self.outer_evaluation.n_folds + self.outer_evaluation.fold

            # Validate split index
            if split_idx >= len(all_splits):
                raise ValueError(
                    f"Invalid fold/repeat combination: fold={self.outer_evaluation.fold}, "
                    f"repeat={self.outer_evaluation.repeat}. "
                    f"Max fold={self.outer_evaluation.n_folds-1}, max repeat={self.outer_evaluation.n_repeats-1}"
                )

            # Get the specific train/test indices for this fold/repeat
            train_indices, test_indices = all_splits[split_idx]

            # Split data based on indices
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]

            return X_train, X_test, y_train, y_test

        else:
            raise ValueError(f"Unknown resampling method: {self.outer_evaluation.resampling}")


        
        