from src.models.base_component import BaseComponent
from sklearn.linear_model import SGDClassifier


class SGDModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("SGD")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_loss",
            f"{self.prefix}_penalty",
            f"{self.prefix}_alpha",
            f"{self.prefix}_learning_rate",
            f"{self.prefix}_eta0",
            f"{self.prefix}_l1_ratio",
            f"{self.prefix}_power_t"
        ]

    def construct(self, config):
        """Construct SGD Classifier with hyperparameters
        """
        # Store hyperparameters relevant for SGD Classifier in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return SGDClassifier(
            loss=self.config[f"{self.prefix}_loss"],
            penalty=self.config[f"{self.prefix}_penalty"],
            alpha=self.config[f"{self.prefix}_alpha"],
            learning_rate=self.config[f"{self.prefix}_learning_rate"],
            eta0=self.config[f"{self.prefix}_eta0"],
            l1_ratio=self.config[f"{self.prefix}_l1_ratio"],
            power_t=self.config[f"{self.prefix}_power_t"],
            random_state=self.config["random_state"],
            n_jobs=-1
        )
