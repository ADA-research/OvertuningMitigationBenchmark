from src.models.base_component import BaseComponent
from sklearn.ensemble import BaggingClassifier


class BaggingModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("Bagging")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_n_estimators",
            f"{self.prefix}_max_samples",
            f"{self.prefix}_max_features",
            f"{self.prefix}_bootstrap"
        ]

    def construct(self, config):
        """Construct Bagging classifier with hyperparameters
        """
        # Store hyperparameters relevant for Bagging in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return BaggingClassifier(
            n_estimators=self.config[f"{self.prefix}_n_estimators"],
            max_samples=self.config[f"{self.prefix}_max_samples"],
            max_features=self.config[f"{self.prefix}_max_features"],
            bootstrap=self.config[f"{self.prefix}_bootstrap"],
            random_state=self.config["random_state"],
            n_jobs=-1
        )