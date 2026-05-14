from src.models.base_component import BaseComponent
from sklearn.neural_network import MLPClassifier


class MLPModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("MLP")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_n_hidden_layers",
            f"{self.prefix}_hidden_layer_sizes",
            f"{self.prefix}_activation",
            f"{self.prefix}_solver",
            f"{self.prefix}_alpha",
            f"{self.prefix}_learning_rate_init",
            f"{self.prefix}_max_iter",
            f"{self.prefix}_early_stopping"
        ]

    def construct(self, config):
        """Construct MLPClassifier with hyperparameters
        """
        # Store hyperparameters relevant for MLPClassifier in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        return MLPClassifier(
            hidden_layer_sizes=tuple(
                [self.config[f"{self.prefix}_hidden_layer_sizes"]] * self.config[f"{self.prefix}_n_hidden_layers"]),
            activation=self.config[f"{self.prefix}_activation"],
            solver=self.config[f"{self.prefix}_solver"],
            alpha=self.config[f"{self.prefix}_alpha"],
            learning_rate_init=self.config[f"{self.prefix}_learning_rate_init"],
            max_iter=self.config[f"{self.prefix}_max_iter"],
            early_stopping=self.config[f"{self.prefix}_early_stopping"],
            random_state=self.config["random_state"]
        )
