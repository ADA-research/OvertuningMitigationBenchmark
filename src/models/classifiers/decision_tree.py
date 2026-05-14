from src.models.base_component import BaseComponent
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("DecisionTree")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_max_depth",
            f"{self.prefix}_min_samples_split",
            f"{self.prefix}_min_samples_leaf"
        ]

    def construct(self, config):
        """Construct Decision Tree classifier with hyperparameters
        """
        # Store hyperparameters relevant for Decision Tree in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return DecisionTreeClassifier(
            max_depth=self.config[f"{self.prefix}_max_depth"],
            min_samples_split=self.config[f"{self.prefix}_min_samples_split"],
            min_samples_leaf=self.config[f"{self.prefix}_min_samples_leaf"],
            random_state=self.config["random_state"]
        )