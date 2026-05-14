from src.models.base_component import BaseComponent
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, mutual_info_regression


class SelectKBestComponent(BaseComponent):
    def __init__(self):
        super().__init__("SelectKBest")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_score_func",
            f"{self.prefix}_k"
        ]

    def construct(self, config):
        """Construct SelectKBest feature selector."""
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

        return SelectKBest(
            score_func=score_func,
            k=self.config[f"{self.prefix}_k"]
        )