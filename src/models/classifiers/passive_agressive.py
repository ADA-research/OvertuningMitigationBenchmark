from src.models.base_component import BaseComponent
from sklearn.linear_model import PassiveAggressiveClassifier


class PassiveAggressiveModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("PassiveAggressive")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_C",
            f"{self.prefix}_max_iter",
            f"{self.prefix}_tol"
        ]

    def construct(self, config):
        """Construct Passive Aggressive Classifier with hyperparameters
        """
        # Store hyperparameters relevant for Passive Aggressive Classifier in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return PassiveAggressiveClassifier(
            C=self.config[f"{self.prefix}_C"],
            max_iter=self.config[f"{self.prefix}_max_iter"],
            tol=self.config[f"{self.prefix}_tol"],
            random_state=self.config["random_state"],
            n_jobs=-1
        )