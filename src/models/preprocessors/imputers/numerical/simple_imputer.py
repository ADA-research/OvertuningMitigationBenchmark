from src.models.base_component import BaseComponent
from sklearn.impute import SimpleImputer


class NumericalSimpleImputerComponent(BaseComponent):
    def __init__(self):
        super().__init__("NumericalSimpleImputer")
        self.required_hyperparameters = [
            f"{self.prefix}_strategy"
        ]

    def construct(self, config):
        """
        Construct SimpleImputer for numerical data.
        Default strategy: 'mean'.
        """
        self.config = {k: v for k, v in config.items() if k.startswith(self.prefix)}
        self.check(self.config)

        return SimpleImputer(
            strategy=self.config[f"{self.prefix}_strategy"]
        )