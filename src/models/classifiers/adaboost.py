from src.models.base_component import BaseComponent
from sklearn.ensemble import AdaBoostClassifier


class AdaBoostModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("AdaBoost")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_n_estimators",
            f"{self.prefix}_learning_rate"
        ]

    def construct(self, config):
        """Construct AdaBoost classifier with hyperparameters
        """
        # Store hyperparameters relevant for AdaBoost in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return AdaBoostClassifier(
            n_estimators=self.config[f"{self.prefix}_n_estimators"],
            learning_rate=self.config[f"{self.prefix}_learning_rate"],
            random_state=self.config["random_state"]
        )