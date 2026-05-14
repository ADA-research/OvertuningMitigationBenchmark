from typing import Iterator, Tuple
import numpy as np
import pandas as pd

from src.resamplers.online_resamplers.cv_resampler import CVResampler
from sklearn.model_selection import train_test_split, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.utils.multiclass import type_of_target


class CVSelectionSetResampler(CVResampler):
    def __init__(self,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame = None,
                 y_test: pd.Series = None,
                 seed: int = 0,
                 selection_fraction: float = 0.2,
                 n_folds: int = 5,
                 n_repeats: int = 1,
                 reshuffle: bool = False,
                 problem_type: str | None = None):
        super().__init__(
            X_train,
            y_train,
            X_test,
            y_test,
            n_folds=n_folds,
            n_repeats=n_repeats,
            reshuffle=reshuffle,
            seed=seed,
            problem_type=problem_type,
        )

        self.selection_fraction = selection_fraction

        # Separate selection set from "train" set up front
        # Selection data is not part of the train set used for CV splits
        self.X, self.X_sel, self.y, self.y_sel = train_test_split(
            self.X,
            self.y,
            test_size=selection_fraction,
            random_state=self.seed,
            stratify=None if self.is_regression else self.y
        )

        self.resample()

    def get_fold_data(self, fold_idx: int) -> Tuple[
        Tuple[pd.DataFrame, pd.Series],  # train
        Tuple[pd.DataFrame, pd.Series],  # val
        Tuple[pd.DataFrame, pd.Series],  # test
        Tuple[pd.DataFrame, pd.Series],  # selection
    ]:
        train_idx, val_idx = self.folds[fold_idx]
        X_fold_train = self.X.iloc[train_idx]
        y_fold_train = self.y.iloc[train_idx]
        X_fold_val = self.X.iloc[val_idx]
        y_fold_val = self.y.iloc[val_idx]

        train_data = (X_fold_train, y_fold_train)
        val_data = (X_fold_val, y_fold_val)
        test_data = (self.X_test, self.y_test)
        selection_data = (self.X_sel, self.y_sel)

        return train_data, val_data, test_data, selection_data
