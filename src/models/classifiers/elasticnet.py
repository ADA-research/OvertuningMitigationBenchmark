from src.models.base_component import BaseComponent
from sklearn.linear_model import ElasticNet


class ElasticNetModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("ElasticNet")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_alpha",
            f"{self.prefix}_l1_ratio"
        ]

    def construct(self, config):
        """Construct ElasticNet model with hyperparameters
        """
        # Store hyperparameters relevant for ElasticNet in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return ElasticNet(
            alpha=self.config[f"{self.prefix}_alpha"],
            l1_ratio=self.config[f"{self.prefix}_l1_ratio"],
            random_state=self.config["random_state"]
        )