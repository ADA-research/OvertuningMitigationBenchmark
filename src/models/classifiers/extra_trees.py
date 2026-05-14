from src.models.base_component import BaseComponent
from sklearn.ensemble import ExtraTreesClassifier


class ExtraTreesModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("ExtraTrees")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_n_estimators",
            f"{self.prefix}_max_features",
            f"{self.prefix}_min_samples_split",
            f"{self.prefix}_min_impurity_decrease",
            f"{self.prefix}_bootstrap"
        ]

    def construct(self, config):
        """Construct Extra Trees classifier with hyperparameters
        """
        # Store hyperparameters relevant for Extra Trees in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return ExtraTreesClassifier(
            n_estimators=self.config[f"{self.prefix}_n_estimators"],
            max_features=self.config[f"{self.prefix}_max_features"],
            min_samples_split=self.config[f"{self.prefix}_min_samples_split"],
            min_impurity_decrease=self.config[f"{self.prefix}_min_impurity_decrease"],
            bootstrap=self.config[f"{self.prefix}_bootstrap"],
            random_state=self.config["random_state"],
            n_jobs=1
        )