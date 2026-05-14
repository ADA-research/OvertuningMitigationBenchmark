from typing import Iterator, Tuple, List

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

from src.resamplers.base_resampler import BaseResampler


class BaseOnlineResampler(BaseResampler):
    def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame = None,
            y_test: pd.Series = None,
            n_folds: int = 5,
            n_repeats: int = 1,
            reshuffle: bool = False,
            seed: int = 0,
            problem_type: str | None = None,
    ):

        super().__init__(X_train, y_train, X_test, y_test, reshuffle, seed)

        self.n_folds = n_folds
        self.n_repeats = n_repeats

        self.total_folds = n_folds * n_repeats
        self.iteration_count = 0
        # Prefer an explicit task-level problem type when available.
        # This avoids treating integer-valued regression targets as classification.
        if problem_type is not None:
            self.is_regression = problem_type == "regression"
        else:
            self.is_regression = type_of_target(y_train) == "continuous"

        self.folds: List[Tuple[np.ndarray, np.ndarray]] = []

    def resample(self):
        pass

    def get_fold_data(self, fold_idx: int) -> Tuple[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[None, None]
    ]:

        train_idx, val_idx = self.folds[fold_idx]

        X_fold_train = self.X.iloc[train_idx]
        y_fold_train = self.y.iloc[train_idx]
        X_fold_val = self.X.iloc[val_idx]
        y_fold_val = self.y.iloc[val_idx]

        train_data = (X_fold_train, y_fold_train)
        val_data = (X_fold_val, y_fold_val)
        test_data = (self.X_test, self.y_test)

        return train_data, val_data, test_data, (None, None)

    def __iter__(self) -> Iterator[
        Tuple[
            Tuple[pd.DataFrame, pd.Series],
            Tuple[pd.DataFrame, pd.Series],
            Tuple[pd.DataFrame, pd.Series],
            Tuple[None, None]
        ]]:

        self.reset()

        return self

    def __next__(self) -> Tuple[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[None, None]
    ]:

        if self.iteration_count == self.total_folds:
            raise StopIteration

        fold_data = self.get_fold_data(self.iteration_count)

        self.iteration_count += 1

        return fold_data

    def reset(self) -> None:
        """Resetting kfold procedure"""
        self.iteration_count = 0

        if self.reshuffle:
            self.resample()
