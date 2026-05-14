from src.models.base_component import BaseComponent
from sklearn.preprocessing import MinMaxScaler


class MinMaxScalerComponent(BaseComponent):
    def __init__(self):
        super().__init__("MinMaxScaler")

        # Required hyperparameters
        self.required_hyperparameters = []

    def construct(self, config):
        """Construct MinMaxScaler with feature range."""
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }

        self.check(self.config)

        return MinMaxScaler()
