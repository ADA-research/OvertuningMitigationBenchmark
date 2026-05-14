from src.models.base_component import BaseComponent
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("LogisticRegression")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_C",
            f"{self.prefix}_solver"
        ]

    def construct(self, config):
        """Construct Logistic Regression classifier with hyperparameters
        """
        # Store hyperparameters relevant for Logistic Regression in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return LogisticRegression(
            C=self.config[f"{self.prefix}_C"],
            solver=self.config[f"{self.prefix}_solver"],
            random_state=self.config["random_state"],
            n_jobs=-1
        )