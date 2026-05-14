from src.models.base_component import BaseComponent
from sklearn.preprocessing import StandardScaler


class StandardScalerComponent(BaseComponent):
    def __init__(self):
        super().__init__("StandardScaler")

        # No required hyperparameters for StandardScaler
        self.required_hyperparameters = []

    def construct(self, config):
        """Construct StandardScaler without any specific hyperparameters."""

        self.check(config)

        return StandardScaler()