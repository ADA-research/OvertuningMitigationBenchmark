from src.models.base_component import BaseComponent
from sklearn.preprocessing import Normalizer


class NormalizerComponent(BaseComponent):
    def __init__(self):
        super().__init__("Normalizer")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_norm"
        ]

    def construct(self, config):
        """Construct Normalizer with normalization mode."""
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }

        self.check(self.config)

        return Normalizer(
            norm=self.config[f"{self.prefix}_norm"]
        )