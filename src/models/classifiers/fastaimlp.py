import numpy as np
import tempfile
import pandas as pd
from sklearn.base import BaseEstimator
from autogluon.tabular.models import NNFastAiTabularModel
from src.models.base_component import BaseComponent

LAYERS_OPTIONS = {
    0: [200],
    1: [400],
    2: [200, 100],
    3: [400, 200],
    4: [800, 400],
    5: [200, 100, 50],
    6: [400, 200, 100],
}


class FastAIMLPModel(BaseComponent, BaseEstimator):
    """
    To use NNFastAiTabularModel, we need to pass hyperparameters as dict. 
    This model is not aslready an sklearn estimator, so we implement fit, predict_proba, and predict
    """
    def __init__(self, problem_type="binary", hyperparameters=None, num_cpus=1):
        BaseComponent.__init__(self, "FastAIMLP")
        self.problem_type = problem_type
        self.hyperparameters = {} if hyperparameters is None else hyperparameters
        self.num_cpus = num_cpus

    def construct(self, config):
        # Construct hyperparamters to be passed to NNFastAiTabularModel
        hp = {
            k.replace(f"{self.prefix}_", ""): v for k, v in config.items() if k.startswith(f"{self.prefix}_")
        }

        if "random_state" in config:
            hp["random_seed"] = int(config["random_state"])

        if "layers" in hp:
            hp["layers"] = LAYERS_OPTIONS[int(hp["layers"])]
        
        return FastAIMLPModel(
            problem_type=config.get("_problem_type", "binary"),
            hyperparameters=hp,
            num_cpus=max(1, int(config.get("_model_threads", 1))),
        )

    def fit(self, X, y):
        # Initialize model
        self.model_ = NNFastAiTabularModel(
            path=tempfile.gettempdir(),
            problem_type=self.problem_type, 
            hyperparameters=self.hyperparameters
        
        )
        
        # Fit model
        self.model_.fit(
            X=pd.DataFrame(X), 
            y=pd.Series(y), 
            num_gpus=0, 
            num_cpus=self.num_cpus
        )

        return self

    def predict(self, X):
        return self.model_.predict(X=pd.DataFrame(X))

    def predict_proba(self, X):
        proba = self.model_.predict_proba(X=pd.DataFrame(X))

        if hasattr(proba, "ndim") and proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])

        return proba


FastAIMLPClassifier = FastAIMLPModel