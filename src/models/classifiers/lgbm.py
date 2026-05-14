from src.models.base_component import BaseComponent
from lightgbm import LGBMClassifier


class LGBMModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("LGBM")

        # Required hyperparameters - most important ones to tune for LGBM
        self.required_hyperparameters = [
            f"{self.prefix}_n_estimators",
            # f"{self.prefix}_early_stopping_rounds",
            f"{self.prefix}_learning_rate",
            f"{self.prefix}_num_leaves",
            f"{self.prefix}_feature_fraction",
            f"{self.prefix}_bagging_fraction",
            f"{self.prefix}_bagging_freq",
            f"{self.prefix}_min_data_in_leaf",
            f"{self.prefix}_min_sum_hessian_in_leaf",
            f"{self.prefix}_lambda_l1_use",
            f"{self.prefix}_lambda_l2_use",
        ]

    def construct(self, config):
        """Construct LightGBM classifier with hyperparameters
        """
        model_threads = max(1, int(config.get("_model_threads", 1)))

        # Store hyperparameters relevant for LGBM in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        # Determine lambda_l1 and lambda_l2 values based on use flags
        lambda_l1 = (
            self.config[f"{self.prefix}_lambda_l1_value"]
            if self.config[f"{self.prefix}_lambda_l1_use"]
            else 0.0
        )
        lambda_l2 = (
            self.config[f"{self.prefix}_lambda_l2_value"]
            if self.config[f"{self.prefix}_lambda_l2_use"]
            else 0.0
        )

        return LGBMClassifier(
            n_estimators=self.config[f"{self.prefix}_n_estimators"],
            # early_stopping_rounds=self.config[f"{self.prefix}_early_stopping_rounds"],
            learning_rate=self.config[f"{self.prefix}_learning_rate"],
            num_leaves=self.config[f"{self.prefix}_num_leaves"],
            feature_fraction=self.config[f"{self.prefix}_feature_fraction"],
            bagging_fraction=self.config[f"{self.prefix}_bagging_fraction"],
            bagging_freq=self.config[f"{self.prefix}_bagging_freq"],
            min_data_in_leaf=self.config[f"{self.prefix}_min_data_in_leaf"],
            min_sum_hessian_in_leaf=self.config[f"{self.prefix}_min_sum_hessian_in_leaf"],
            reg_alpha=lambda_l1,
            reg_lambda=lambda_l2,
            random_state=self.config["random_state"],
            num_threads=model_threads,
            verbose=-1  # Suppress training output
        )