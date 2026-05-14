"""
Multi-Layer Perceptron Space Configuration.

This module defines the `MLPSpace` class, which is responsible
for constructing the hyperparameter search space for the MLPClassifier
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'MLPClassifier'.

Classes:
    - MLPSpace: Constructs the hyperparameter space for MLPClassifier.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class MLPSpace(BaseComponentSpace):
    """
    Defines the configuration space for the MLPClassifier model.

    This class builds the search space for fitting an MLPClassifier model
    via ConfigSpace. It includes standard hyperparameters like hidden layer
    size, activation function, and solver type, as well as other hyperparameters
    relevant to neural networks, such as learning rate initialization and maximum
    iterations.

    Inherits:
        BaseComponentSpace: Handles conditional addition of hyperparameters
        dependent on the model choice.

    Attributes:
        component_hp (ConfigSpace.hyperparameter.Hyperparameter):
            Reference to the top-level component hyperparameter, which determines
            the model type in the pipeline.
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the MLPClassifier hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'MLPClassifier').
        """
        super().__init__(component_hp, name="MLP", seed=seed)

        # Add hyperparameters specific to MLP
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_n_hidden_layers", 1, 3, default_value=1
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_hidden_layer_sizes", 10, 500, default_value=100
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_activation", choices=["identity", "logistic", "tanh", "relu"], default_value="relu"
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_solver", choices=["lbfgs", "sgd", "adam"], default_value="adam"
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_learning_rate_init", 0.0001, 0.1, default_value=0.001, log=True
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_max_iter", 10, 1000, default_value=200
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_early_stopping", choices=[True, False], default_value=False
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_alpha", 1e-5, 1.0, default_value=0.0001, log=True
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_learning_rate", choices=["constant", "invscaling", "adaptive"], default_value="constant"
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_batch_size", 1, 256, default_value=16
            )
        ))