from typing import Iterator, Tuple
import numpy as np
import pandas as pd

class BaseResampler:
    def __init__(self,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame = None,
                 y_test: pd.Series = None,
                 reshuffle: bool = False,
                 seed: int = 0):

        self.X = X_train
        self.y = y_train

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = X_test
        self.y_test = y_test
        self.reshuffle = reshuffle
        self.num_reshuffles = 0

        self.seed = seed

    def resample(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass