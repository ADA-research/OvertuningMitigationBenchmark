from src.models.base_component import BaseComponent
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif, f_regression, mutual_info_regression


class SelectPercentileComponent(BaseComponent):
    def __init__(self):
        super().__init__("SelectPercentile")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_score_func",
            f"{self.prefix}_percentile"
        ]

    def construct(self, config):
        """Construct SelectPercentile feature selector."""
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }
        self.check(self.config)

        # Map score functions
        score_func_mapping = {
            "f_classif": f_classif,
            "mutual_info_classif": mutual_info_classif,
            "f_regression": f_regression,
            "mutual_info_regression": mutual_info_regression,
        }

        score_func = score_func_mapping.get(self.config[f"{self.prefix}_score_func"])

        return SelectPercentile(
            score_func=score_func,
            percentile=self.config[f"{self.prefix}_percentile"]
        )