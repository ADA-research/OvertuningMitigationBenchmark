from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.utils.multiclass import type_of_target

from src.history.data_classes import Fold, Split
from src.resamplers.online_resamplers.base_online_resampler import BaseOnlineResampler



class CVResampler(BaseOnlineResampler):
    def __init__(self,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame = None,
                 y_test: pd.Series = None,
                 seed: int = 0,
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

        self.resample()

    def resample(self):
        if self.is_regression:
            kfold = RepeatedKFold(
                n_splits=self.n_folds,
                n_repeats=self.n_repeats,
                random_state=self.seed + self.num_reshuffles
            )
        else:
            kfold = RepeatedStratifiedKFold(
                n_splits=self.n_folds,
                n_repeats=self.n_repeats,
                random_state=self.seed + self.num_reshuffles
            )

        if self.reshuffle:
            self.num_reshuffles += 1

        self.folds = list(kfold.split(self.X, self.y))