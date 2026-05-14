from src.models.base_component import BaseComponent
from src.models.preprocessors.dim_reducers.dynamic_n_components import DynamicDimensionReducer

from sklearn.decomposition import PCA


class PCAComponent(BaseComponent):
    def __init__(self):
        super().__init__("PCA")
        self.required_hyperparameters = [
            f"{self.prefix}_n_components",
            f"{self.prefix}_whiten"
        ]

    def construct(self, config):
        """
        Construct PCA reducer.
        Default configuration: `n_components` must be specified.
        """
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }

        self.check(self.config)

        return DynamicDimensionReducer(
            PCA(
                n_components=self.config[f"{self.prefix}_n_components"],
                whiten=self.config[f"{self.prefix}_whiten"]
        ))
