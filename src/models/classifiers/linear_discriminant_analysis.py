from src.models.base_component import BaseComponent
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LinearDiscriminantAnalysisModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("LinearDiscriminantAnalysis")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_solver",
            # f"{self.prefix}_shrinkage"
        ]

    def construct(self, config):
        """Construct LinearDiscriminantAnalysis model with hyperparameters
        """
        # Store hyperparameters relevant for LinearDiscriminantAnalysis in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }

        # Check hyperparameters
        self.check(self.config)

        return LinearDiscriminantAnalysis(
            solver=self.config[f"{self.prefix}_solver"],
            # shrinkage=self.config[f"{self.prefix}_shrinkage"]
        )