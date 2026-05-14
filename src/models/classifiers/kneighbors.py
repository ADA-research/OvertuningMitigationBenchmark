from src.models.base_component import BaseComponent
from sklearn.neighbors import KNeighborsClassifier


class KNeighborsModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("KNeighbors")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_n_neighbors",
            f"{self.prefix}_weights",
            f"{self.prefix}_algorithm",
            f"{self.prefix}_leaf_size",
            f"{self.prefix}_p"
        ]

    def construct(self, config):
        """Construct KNeighborsClassifier with hyperparameters
        """
        # Store hyperparameters relevant for KNeighborsClassifier in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }

        # Check hyperparameters
        self.check(self.config)

        return KNeighborsClassifier(
            n_neighbors=self.config[f"{self.prefix}_n_neighbors"],
            weights=self.config[f"{self.prefix}_weights"],
            algorithm=self.config[f"{self.prefix}_algorithm"],
            leaf_size=self.config[f"{self.prefix}_leaf_size"],
            p=self.config[f"{self.prefix}_p"],
            n_jobs=-1
        )