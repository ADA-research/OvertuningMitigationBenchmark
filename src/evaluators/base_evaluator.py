import time
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import sklearn
from sklearn.base import clone
from joblib import Parallel, delayed, parallel_config

from src.history.data_classes import Fold, Split
from src.metrics.metric import BaseMetric



def _evaluate_worker(metric: BaseMetric,
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

    # We predict probabilities unless the estimator does not support that
    prediction_function = estimator.predict_proba if hasattr(estimator, 'predict_proba') else estimator.predict

    # Normal fit of estimator on train data
    train_start = time.time()
    estimator.fit(X_train, y_train)
    train_time = time.time() - train_start

    # Make predictions on train data
    train_inference_start = time.time()
    train_preds = prediction_function(X_train)
    train_inference_time = time.time() - train_inference_start

    # Make predictions on validation data
    val_inference_start = time.time()
    val_preds = prediction_function(X_val)
    val_inference_time = time.time() - val_inference_start

    # Make predictions on test data
    test_inference_start = time.time()
    test_preds = prediction_function(X_test)
    test_inference_time = time.time() - test_inference_start

    # Calculate scores
    train_score = metric.score(y_train, train_preds)
    val_score = metric.score(y_val, val_preds)
    test_score = metric.score(y_test, test_preds)

    if use_selection_set:
        selection_inference_start = time.time()
        sel_preds = prediction_function(X_sel)
        selection_inference_time = time.time() - selection_inference_start
        sel_score = metric.score(y_sel, sel_preds)

    # Create the main fold result
    main_fold = Fold(
        fold_id=fold_id,
        scores=Split(
            train=train_score,
            val=val_score,
            test=test_score,
            selection=None if not use_selection_set else sel_score
        ),
        preds=Split(
            train=None, #train_preds.tolist() if hasattr(train_preds, 'tolist') else train_preds, Question: Do we need train preds?
            val=series_to_numpy(val_preds),
            test=series_to_numpy(test_preds),
            selection=None if not use_selection_set else series_to_numpy(sel_preds)
        ),
        labels=Split(
            train=None, # y_train.tolist() if hasattr(y_train, 'tolist') else y_train,
            val=series_to_numpy(y_val),
            test=series_to_numpy(y_test),
            selection=None if not use_selection_set else series_to_numpy(y_sel)
        ),
        times=Split(
            train=train_time,
            val=val_inference_time,
            test=test_inference_time,
            selection=None if not use_selection_set else selection_inference_time
        )
    )

    return main_fold


def _retrain_and_evaluate_worker(metric: BaseMetric,
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

    # We predict probabilities unless the estimator does not support that
    prediction_function = estimator.predict_proba if hasattr(estimator, 'predict_proba') else estimator.predict

    if use_selection_set:
        # Concatenate train, validation, and selection data for retraining
        X_retrain = pd.concat([X_train, X_val, X_sel], axis=0)
        y_retrain = pd.concat([y_train, y_val, y_sel], axis=0)
    else:
        # Concatenate train and validation data for retraining
        X_retrain = pd.concat([X_train, X_val], axis=0)
        y_retrain = pd.concat([y_train, y_val], axis=0)

    # Retrain estimator
    retrain_start = time.time()
    estimator.fit(X_retrain, y_retrain)
    retrain_time = time.time() - retrain_start

    # Make predictions on train data
    retrained_inference_start = time.time()
    retrained_train_preds = prediction_function(X_retrain)
    retrained_inference_time = time.time() - retrained_inference_start

    # Make predictions on test data
    test_inference_start = time.time()
    test_preds = prediction_function(X_test)
    test_inference_time = time.time() - test_inference_start

    retrained_train_score = metric.score(y_retrain, retrained_train_preds)
    retrained_test_score = metric.score(y_test, test_preds)

    # Create retrained fold object
    retrained_fold = Fold(
        fold_id=fold_id,
        scores=Split(
            train=retrained_train_score,
            val=None,
            test=retrained_test_score,
            selection=None
        ),
        preds=Split(
            train=None, # retrained_train_preds, Question: Do we need train preds?
            val=None,
            test=series_to_numpy(test_preds),
            selection=None
        ),
        labels=Split(
            train=None, # y_retrain.tolist() if hasattr(y_retrain, 'tolist') else y_retrain, Question: Do we need train preds?
            val=None,
            test=series_to_numpy(y_test),
            selection=None

        ),
        times=Split(
            train=retrain_time,
            val=None,
            test=test_inference_time
        )
    )

    return retrained_fold




def series_to_numpy(series: pd.Series) -> np.ndarray:
    # For integer labels, we can keep binary format
    if series.dtype == int:
        return series.to_numpy(dtype=np.int16) if hasattr(series, 'to_numpy') else series.astype(np.int16)

    # For non-integer values (probabilities, regression targets or outputs), we use float32
    array = series.to_numpy(dtype=np.float32) if hasattr(series, 'to_numpy') else series.astype(np.float64)

    if array.ndim == 2 and array.shape[1] == 2:
        array = array[:, 1]  # Keep only the positive class probabilities for binary classification

    return array


class Evaluator:

    def __init__(self, metric: BaseMetric):
        self.metric: BaseMetric = metric

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
        return _evaluate_worker(
            self.metric,
            estimator,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X_sel,
            y_sel,
            fold_id
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
        return _retrain_and_evaluate_worker(
            self.metric,
            estimator,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X_sel,
            y_sel,
            fold_id
        )