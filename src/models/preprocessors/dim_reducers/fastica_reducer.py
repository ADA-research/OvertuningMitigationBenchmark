from src.models.base_component import BaseComponent
from src.models.preprocessors.dim_reducers.dynamic_n_components import DynamicDimensionReducer
from sklearn.decomposition import FastICA


class FastICAComponent(BaseComponent):
    def __init__(self):
        super().__init__("FastICA")

        self.required_hyperparameters = [
            f"{self.prefix}_n_components",
            f"{self.prefix}_algorithm",
            f"{self.prefix}_fun",
            f"{self.prefix}_max_iter"
        ]

    def construct(self, config):
        """Construct FastICA reducer.
        Add configuration based on dataset, like number of columns? This could avoid the dynamic reducer class.
        """
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }

        self.check(self.config)

        # Prevent n_components to be larger than n_features
        return DynamicDimensionReducer(
            FastICA(
                n_components=self.config[f"{self.prefix}_n_components"],
                algorithm=self.config[f"{self.prefix}_algorithm"],
                fun=self.config[f"{self.prefix}_fun"],
                max_iter=self.config[f"{self.prefix}_max_iter"]
        ))