from src.models.base_component import BaseComponent
from sklearn.linear_model import Lasso


class LassoModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("Lasso")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_alpha"
        ]

    def construct(self, config):
        """Construct Lasso model with hyperparameters
        """
        # Store hyperparameters relevant for Lasso in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return Lasso(
            alpha=self.config[f"{self.prefix}_alpha"],
            random_state=self.config["random_state"]
        )