import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.history.data_classes import Fold, Split
from src.metrics.metric import BaseMetric


class Evaluator:

    def __init__(self, metric: BaseMetric, prediction_storage_precision: str = "float16"):
        self.metric: BaseMetric = metric
        self.prediction_storage_precision = prediction_storage_precision

    def evaluate(self,
                 estimator: any,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_val: pd.DataFrame,
                 y_val: pd.Series,
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 X_sel: Optional[pd.DataFrame] = None,
                 y_sel: Optional[pd.Series] = None,
                 fold_id: int = None) -> Fold:
        
        use_selection_set = (X_sel is not None and y_sel is not None)

        estimator = clone(estimator)
        prediction_function = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict

        train_start = time.time()
        estimator.fit(X_train, y_train)
        train_time = time.time() - train_start

        train_preds = prediction_function(X_train)

        val_inference_start = time.time()
        val_preds = prediction_function(X_val)
        val_inference_time = time.time() - val_inference_start

        test_inference_start = time.time()
        test_preds = prediction_function(X_test)
        test_inference_time = time.time() - test_inference_start

        train_score = self.metric.score(y_train, train_preds)
        val_score = self.metric.score(y_val, val_preds)
        test_score = self.metric.score(y_test, test_preds)

        selection_inference_time = None
        sel_score = None
        sel_preds = None
        if use_selection_set:
            selection_inference_start = time.time()
            sel_preds = prediction_function(X_sel)
            selection_inference_time = time.time() - selection_inference_start
            sel_score = self.metric.score(y_sel, sel_preds)

        return Fold(
            fold_id=fold_id,
            scores=Split(
                train=train_score,
                val=val_score,
                test=test_score,
                selection=sel_score,
            ),
            preds=Split(
                train=None,
                val=self.series_to_numpy(val_preds),
                test=self.series_to_numpy(test_preds),
                selection=None if not use_selection_set else self.series_to_numpy(sel_preds),
            ),
            labels=Split(
                train=None,
                val=self.series_to_numpy(y_val),
                test=self.series_to_numpy(y_test),
                selection=None if not use_selection_set else self.series_to_numpy(y_sel),
            ),
            times=Split(
                train=train_time,
                val=val_inference_time,
                test=test_inference_time,
                selection=selection_inference_time,
            ),
        )


    def retrain_and_evaluate(self,
                             estimator: any,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_val: pd.DataFrame,
                             y_val: pd.Series,
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             X_sel: Optional[pd.DataFrame] = None,
                             y_sel: Optional[pd.Series] = None,
                             fold_id: int = None) -> Fold:

        use_selection_set = (X_sel is not None and y_sel is not None)

        estimator = clone(estimator)
        prediction_function = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict

        if use_selection_set:
            X_retrain = pd.concat([X_train, X_val, X_sel], axis=0)
            y_retrain = pd.concat([y_train, y_val, y_sel], axis=0)
        else:
            X_retrain = pd.concat([X_train, X_val], axis=0)
            y_retrain = pd.concat([y_train, y_val], axis=0)

        retrain_start = time.time()
        estimator.fit(X_retrain, y_retrain)
        retrain_time = time.time() - retrain_start

        retrained_train_preds = prediction_function(X_retrain)

        test_inference_start = time.time()
        test_preds = prediction_function(X_test)
        test_inference_time = time.time() - test_inference_start

        retrained_train_score = self.metric.score(y_retrain, retrained_train_preds)
        retrained_test_score = self.metric.score(y_test, test_preds)

        return Fold(
            fold_id=fold_id,
            scores=Split(
                train=retrained_train_score,
                val=None,
                test=retrained_test_score,
                selection=None,
            ),
            preds=Split(
                train=None,
                val=None,
                test=self.series_to_numpy(test_preds),
                selection=None,
            ),
            labels=Split(
                train=None,
                val=None,
                test=self.series_to_numpy(y_test),
                selection=None,
            ),
            times=Split(
                train=retrain_time,
                val=None,
                test=test_inference_time,
            ),
        )

    def series_to_numpy(self, series: pd.Series) -> np.ndarray:
        series_dtype = getattr(series, "dtype", None)
        if series_dtype is not None and np.issubdtype(series_dtype, np.integer):
            return series.to_numpy(dtype=np.int16) if hasattr(series, "to_numpy") else series.astype(np.int16)

        precision_map = {
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
        }

        target_dtype = precision_map[self.prediction_storage_precision]
        array = series.to_numpy(dtype=target_dtype) if hasattr(series, "to_numpy") else series.astype(target_dtype)

        if array.ndim == 2 and array.shape[1] == 2:
            array = array[:, 1]

        return array
