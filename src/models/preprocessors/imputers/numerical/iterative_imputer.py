from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from src.models.base_component import BaseComponent


class IterativeImputerComponent(BaseComponent):
    def __init__(self):
        super().__init__("IterativeImputer")
        self.required_hyperparameters = [
            f"{self.prefix}_max_iter",
            f"{self.prefix}_imputation_order"
        ]

    def construct(self, config):
        """
        Construct IterativeImputer for numerical data.
        Default settings: max_iter=10, imputation_order='ascending'.
        """
        self.config = {k: v for k, v in config.items() if k.startswith(self.prefix)}
        self.check(self.config)

        return IterativeImputer(
            max_iter=self.config[f"{self.prefix}_max_iter"],
            imputation_order=self.config[f"{self.prefix}_imputation_order"]
        )
