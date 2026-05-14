from src.models.base_component import BaseComponent
from sklearn.ensemble import RandomForestRegressor


class RandomForestRegressorModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("RandomForest")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_n_estimators",
            f"{self.prefix}_min_samples_split",
            f"{self.prefix}_max_features",
            f"{self.prefix}_bootstrap",
            # f"{self.prefix}_max_samples",
            f"{self.prefix}_min_impurity_decrease",
        ]

    def construct(self, config):
        """Construct random forest with hyperparameters
        """
        # Store hyperparameters relevant for the random forest in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        max_samples = None if not self.config[f"{self.prefix}_bootstrap"] else self.config[f"{self.prefix}_max_samples"]

        return RandomForestRegressor(
            n_estimators=self.config[f"{self.prefix}_n_estimators"],
            min_samples_split=self.config[f"{self.prefix}_min_samples_split"],
            max_features=self.config[f"{self.prefix}_max_features"],
            bootstrap=self.config[f"{self.prefix}_bootstrap"],
            max_samples=max_samples,
            min_impurity_decrease=self.config[f"{self.prefix}_min_impurity_decrease"],
            random_state=self.config['random_state'],
            n_jobs=1,
        )