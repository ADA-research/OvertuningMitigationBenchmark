from src.models.base_component import BaseComponent
from sklearn.impute import KNNImputer


class KNNImputerComponent(BaseComponent):
    def __init__(self):
        super().__init__("KNNImputer")

        self.required_hyperparameters = [
            f"{self.prefix}_n_neighbors",
            f"{self.prefix}_weights"
        ]

    def construct(self, config):
        """
        Construct KNNImputer for numerical data.
        Default number of neighbors: 5, and weights: 'uniform'.
        """
        self.config = {k: v for k, v in config.items() if k.startswith(self.prefix)}
        self.check(self.config)

        return KNNImputer(
            n_neighbors=self.config[f"{self.prefix}_n_neighbors"],
            weights=self.config[f"{self.prefix}_weights"]
        )