from typing import Iterator, Tuple
import numpy as np
import pandas as pd

from src.resamplers.online_resamplers.base_online_resampler import BaseOnlineResampler
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit


class HoldoutResampler(BaseOnlineResampler):
    def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame = None,
            y_test: pd.Series = None,
            holdout_fraction: float = 0.2,
            reshuffle: bool = False,
            n_repeats: int = 1,
            seed: int = 0,
            problem_type: str | None = None,
        ):

        super().__init__(
            X_train,
            y_train,
            X_test,
            y_test,
            n_repeats=n_repeats,
            reshuffle=reshuffle,
            seed=seed,
            problem_type=problem_type,
        )

        self.holdout_fraction = holdout_fraction
        self.total_folds = n_repeats


        # Initial split
        self.resample()

    def resample(self):
        if self.is_regression:
            splits = ShuffleSplit(
                n_splits=self.n_repeats,
                test_size=self.holdout_fraction,
                random_state=self.seed + self.num_reshuffles
            )
        else:
            splits = StratifiedShuffleSplit(
                n_splits=self.n_repeats,
                test_size=self.holdout_fraction,
                random_state=self.seed + self.num_reshuffles
            )

        if self.reshuffle:
            self.num_reshuffles += 1

        self.folds = list(splits.split(self.X, self.y))