from src.models.base_component import BaseComponent
from sklearn.preprocessing import QuantileTransformer


class QuantileTransformerComponent(BaseComponent):
    def __init__(self):
        super().__init__("QuantileTransformer")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_n_quantiles",
            f"{self.prefix}_output_distribution"
        ]

    def construct(self, config):
        """Construct QuantileTransformer with specific parameters."""
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }

        self.check(self.config)

        return QuantileTransformer(
            n_quantiles=self.config[f"{self.prefix}_n_quantiles"],
            output_distribution=self.config[f"{self.prefix}_output_distribution"]
        )