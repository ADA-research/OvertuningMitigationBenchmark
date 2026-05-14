from src.models.base_component import BaseComponent
from sklearn.naive_bayes import GaussianNB


class GaussianNBModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("GaussianNB")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_var_smoothing"
        ]

    def construct(self, config):
        """Construct GaussianNB model with hyperparameters
        """
        # Store hyperparameters relevant for GaussianNB in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }

        # Check hyperparameters
        self.check(self.config)

        return GaussianNB(
            var_smoothing=self.config[f"{self.prefix}_var_smoothing"]
        )