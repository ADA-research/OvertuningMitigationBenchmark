from src.models.base_component import BaseComponent
from sklearn.tree import ExtraTreeRegressor


class ExtraTreeRegressorModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("ExtraTree")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_max_depth",
            f"{self.prefix}_min_samples_split",
            f"{self.prefix}_min_samples_leaf",
            f"{self.prefix}_splitter",
            f"{self.prefix}_criterion"
        ]

    def construct(self, config):
        """Construct Extra Tree regressor with hyperparameters
        """
        # Store hyperparameters relevant for Extra Tree in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return ExtraTreeRegressor(
            max_depth=self.config[f"{self.prefix}_max_depth"],
            min_samples_split=self.config[f"{self.prefix}_min_samples_split"],
            min_samples_leaf=self.config[f"{self.prefix}_min_samples_leaf"],
            splitter=self.config[f"{self.prefix}_splitter"],
            criterion=self.config[f"{self.prefix}_criterion"],
            random_state=self.config["random_state"]
        )
