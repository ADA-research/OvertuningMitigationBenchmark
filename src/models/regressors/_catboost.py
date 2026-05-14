from src.models.base_component import BaseComponent
from catboost import CatBoostRegressor

class CatBoostRegressorModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("CatBoost")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_iterations",
            f"{self.prefix}_depth",
            f"{self.prefix}_learning_rate",
            f"{self.prefix}_l2_leaf_reg",
            f"{self.prefix}_border_count",
            f"{self.prefix}_bagging_temperature",
            f"{self.prefix}_random_strength"
        ]

    def construct(self, config):
        """Construct CatBoost regressor with hyperparameters
        """
        # Store hyperparameters relevant for CatBoost in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return CatBoostRegressor(
            iterations=self.config[f"{self.prefix}_iterations"],
            depth=self.config[f"{self.prefix}_depth"],
            learning_rate=self.config[f"{self.prefix}_learning_rate"],
            l2_leaf_reg=self.config[f"{self.prefix}_l2_leaf_reg"],
            border_count=self.config[f"{self.prefix}_border_count"],
            bagging_temperature=self.config[f"{self.prefix}_bagging_temperature"],
            random_strength=self.config[f"{self.prefix}_random_strength"],
            random_seed=self.config['random_state'],
            verbose=False,
            thread_count=-1,
        )