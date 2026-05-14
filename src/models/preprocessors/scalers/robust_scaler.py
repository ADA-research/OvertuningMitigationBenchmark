from src.models.base_component import BaseComponent
from sklearn.preprocessing import RobustScaler


class RobustScalerComponent(BaseComponent):
    def __init__(self):
        super().__init__("RobustScaler")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_quantile_range_lower",
            f"{self.prefix}_quantile_range_upper"
        ]

    def construct(self, config):
        """Construct RobustScaler with quantile range."""
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }

        self.check(self.config)

        return RobustScaler(
            quantile_range=(self.config[f"{self.prefix}_quantile_range_lower"],
                            self.config[f"{self.prefix}_quantile_range_upper"])
        )