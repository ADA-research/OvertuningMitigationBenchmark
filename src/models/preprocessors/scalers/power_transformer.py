from src.models.base_component import BaseComponent
from sklearn.preprocessing import PowerTransformer


class PowerTransformerComponent(BaseComponent):
    def __init__(self):
        super().__init__("PowerTransformer")

        # Required hyperparameters
        self.required_hyperparameters = []

    def construct(self, config):
        """Construct PowerTransformer with specific method."""
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }

        self.check(self.config)

        return PowerTransformer()