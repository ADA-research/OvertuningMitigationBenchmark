from src.models.base_component import BaseComponent
from sklearn.impute import SimpleImputer


class CategoricalSimpleImputerComponent(BaseComponent):
    def __init__(self):
        super().__init__("CategoricalSimpleImputer")
        self.required_hyperparameters = [
            f"{self.prefix}_strategy",
            f"{self.prefix}_fill_value"
        ]

    def construct(self, config):
        """
        Construct SimpleImputer for categorical data.
        Default strategy: 'most_frequent'. Supports 'constant' for filling specific values.
        """
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }

        self.check(self.config)

        return SimpleImputer(
            strategy=self.config[f"{self.prefix}_strategy"],
            fill_value=self.config[f"{self.prefix}_fill_value"]
        )