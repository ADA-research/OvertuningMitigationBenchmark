import pandas as pd
import tempfile
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


class FastAIMLPRegressorModel(BaseComponent, BaseEstimator):
    def __init__(self, hyperparameters=None, num_cpus=1):
        BaseComponent.__init__(self, "FastAIMLP")
        self.hyperparameters = {} if hyperparameters is None else hyperparameters
        self.num_cpus = num_cpus


    def construct(self, config):
        hp = {k.replace(f"{self.prefix}_", ""): v for k, v in config.items() if k.startswith(f"{self.prefix}_")}
        
        if "random_state" in config:
            hp["random_seed"] = int(config["random_state"])

        if "layers" in hp:
            hp["layers"] = LAYERS_OPTIONS[int(hp["layers"])]

        return FastAIMLPRegressorModel(
            hyperparameters=hp,
            num_cpus=max(1, int(config.get("_model_threads", 1))),
        )

    def fit(self, X, y):
        self.model_ = NNFastAiTabularModel(
            path=tempfile.gettempdir(),
            problem_type="regression", 
            hyperparameters=self.hyperparameters
        )

        self.model_.fit(
            X=pd.DataFrame(X), 
            y=pd.Series(y),
            num_gpus=0, 
            num_cpus=self.num_cpus
        )
        
        return self

    def predict(self, X):
        return self.model_.predict(X=pd.DataFrame(X))
