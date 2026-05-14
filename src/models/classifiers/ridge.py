from src.models.base_component import BaseComponent
from sklearn.linear_model import RidgeClassifier


class RidgeModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("Ridge")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_alpha",
            f"{self.prefix}_solver"
        ]

    def construct(self, config):
        """Construct Ridge Classifier with hyperparameters
        """
        # Store hyperparameters relevant for Ridge Classifier in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return RidgeClassifier(
            alpha=self.config[f"{self.prefix}_alpha"],
            solver=self.config[f"{self.prefix}_solver"],
            random_state=self.config["random_state"]
        )