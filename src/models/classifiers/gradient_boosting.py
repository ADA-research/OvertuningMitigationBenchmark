from src.models.base_component import BaseComponent
from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostingModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("GradientBoosting")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_n_estimators",
            f"{self.prefix}_learning_rate",
            f"{self.prefix}_max_depth",
            f"{self.prefix}_min_samples_split",
            f"{self.prefix}_min_samples_leaf",
            f"{self.prefix}_loss",
            f"{self.prefix}_subsample"
        ]

    def construct(self, config):
        """Construct Gradient Boosting classifier with hyperparameters
        """
        # Store hyperparameters relevant for Gradient Boosting in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return GradientBoostingClassifier(
            n_estimators=self.config[f"{self.prefix}_n_estimators"],
            learning_rate=self.config[f"{self.prefix}_learning_rate"],
            max_depth=self.config[f"{self.prefix}_max_depth"],
            min_samples_split=self.config[f"{self.prefix}_min_samples_split"],
            min_samples_leaf=self.config[f"{self.prefix}_min_samples_leaf"],
            loss=self.config[f"{self.prefix}_loss"],
            subsample=self.config[f"{self.prefix}_subsample"],
            random_state=self.config["random_state"]
        )