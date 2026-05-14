from dataclasses import dataclass, field
from typing import Any, List

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Split:
    train: float | np.ndarray | None = None
    val: float | np.ndarray | None = None
    test: float | np.ndarray | None = None
    selection: float | np.ndarray | None = None


@dataclass(frozen=True)
class Surrogate:
    mean: float
    std: float
    acquisition: float | None = None


@dataclass(frozen=True)
class Fold:
    fold_id: int | None = None # Tuple in case of reshuffling/repeated we need to track the seed
    scores: Split | None = None
    preds: Split | None = None
    labels: Split | None = None
    times: Split | None = None


@dataclass(frozen=False)
class Run:
    config: dict | pd.Series | None = None
    optimizer_suggest_time: float | None = None
    iteration: int | None = None
    folds: List[Fold] = field(default_factory=list)
    retrain: Fold | None = None  # If model is retrained
    surrogate: Surrogate | None = None
    early_stopped: bool | None = False # Need an early stopped trigger
    early_stopped_makarova: bool | None = None
    total_folds: int | None = 1 # Add awareness of total number of folds
    evaluated_folds: List[int] = field(default_factory=list) # Track which folds have been evaluated for this run

    def add_fold(self, fold: Fold) -> None:
        # Use Cached evaluations for already added folds
        if fold.fold_id in self.evaluated_folds:
            return
        self.folds.append(fold)
        if fold.fold_id is not None:
            self.evaluated_folds.append(fold.fold_id)

    @property
    def fully_evaluated(self):
        return self.total_folds == len(self.folds)

    def add_retrain_evaluation(self, retrain_fold: Fold) -> None:
        self.retrain = retrain_fold

    def average_val_score(self) -> float:
        return sum([fold.scores.val for fold in self.folds]) / len(self.folds)

    def average_train_score(self) -> float:
        return sum([fold.scores.train for fold in self.folds]) / len(self.folds)

    def average_test_score(self) -> float:
        if self.retrain:
            return self.retrain.scores.test
        else:
            return sum([fold.scores.test for fold in self.folds]) / len(self.folds)