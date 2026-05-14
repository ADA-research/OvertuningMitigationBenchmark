from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.utils.multiclass import type_of_target


class BaseMetric:
    def __init__(self, name: str):
        self.name: str = name

    def score(self, y_true: pd.Series, y_pred: pd.Series):
        pass



class Metric(BaseMetric):
    def __init__(self,
                 metric_name: str,
                 problem_type: str,
                 threshold: float | None = 0.5):
        super().__init__(metric_name)

        # We want a Metric class that handles sklearn scoring metrics
        # Sklearn does not seem to offer a metric superclass
        # We use get_scorer._score_func to get the underlying metric
        # Note: this does NOT include the negation sign for regression metrics starting with "neg_"
        self.scorer = get_scorer(metric_name)
        self.metric_function = self.scorer._score_func
        self.problem_type = problem_type
        self.threshold = threshold
        self.need_proba = ("predict_proba" in self.scorer._response_method)


    def score(self, y_true: pd.Series, y_pred: pd.Series):

        if self.problem_type == "regression":
            # The sklearn get_scorer._score_func does not include the sign
            # So neg_root_mean_squared_error is actually positive RMSE
            # In the framework we minimize the metric
            # So to minimize neg_root_mean_squared_error, which is actually RMSE, we return the score as-is
            # TODO: Extend for postive regression metrics like r2
            return self.metric_function(y_true, y_pred)

        elif self.problem_type == "binary":
            # We take the probabilities for the positive class
            if not self.need_proba:
                y_pred = (y_pred[:, 1] > self.threshold).astype(int)

            else:
                y_pred = np.clip(y_pred[:, 1], 0.0, 1.0)

            return -self.metric_function(y_true, y_pred)

        # Multiclass metrics
        else:
            if not self.need_proba:
                # Here we extract the highest predicted class
                y_pred = y_pred.argmax(axis=1)
            else:
                # Clip to [0, 1] to guard against floating-point rounding (e.g. 1.0000000000000002)
                y_pred = np.clip(y_pred, 0.0, 1.0)

            if self.name == "roc_auc":
                return -self.metric_function(y_true, y_pred, multi_class="ovr")

            return self.metric_function(y_true, y_pred)
