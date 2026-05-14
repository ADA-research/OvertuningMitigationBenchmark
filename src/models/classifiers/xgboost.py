from src.models.base_component import BaseComponent
from xgboost import XGBClassifier


class XGBoostModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("XGBoost")

        # Required hyperparameters - most important ones to tune
        self.required_hyperparameters = [
            f"{self.prefix}_n_estimators",
            f"{self.prefix}_learning_rate",
            f"{self.prefix}_max_depth",
            f"{self.prefix}_subsample",
            f"{self.prefix}_colsample_bytree",
            f"{self.prefix}_reg_alpha",
            f"{self.prefix}_reg_lambda"
        ]

    def construct(self, config):
        """Construct XGBoost classifier with hyperparameters
        """
        # Store hyperparameters relevant for XGBoost in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return XGBClassifier(
            n_estimators=self.config[f"{self.prefix}_n_estimators"],
            learning_rate=self.config[f"{self.prefix}_learning_rate"],
            max_depth=self.config[f"{self.prefix}_max_depth"],
            subsample=self.config[f"{self.prefix}_subsample"],
            colsample_bytree=self.config[f"{self.prefix}_colsample_bytree"],
            reg_alpha=self.config[f"{self.prefix}_reg_alpha"],
            reg_lambda=self.config[f"{self.prefix}_reg_lambda"],
            random_state=self.config["random_state"],
        )
