"""
SGD Classifier Space Configuration.

This module defines the `SGDClassifierSpace` class, which is responsible
for constructing the hyperparameter search space for the SGDClassifier
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'SGDClassifier'.

Classes:
    - SGDClassifierSpace: Constructs the hyperparameter space for SGDClassifier.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class SGDSpace(BaseComponentSpace):
    """
    Defines the configuration space for the SGDClassifier model.

    This class builds the search space for fitting a SGDClassifier model
    via ConfigSpace. It includes hyperparameters for controlling the optimization
    process such as the loss function, regularization type, and learning rate scheme.

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
        Initializes the SGDClassifier hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'SGD').
        """
        super().__init__(component_hp, name="SGD", seed=seed)

        # Add hyperparameters specific to SGD
        self.add_hyperparameters((
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_loss",
                choices=[
                    "huber", "squared_epsilon_insensitive", "squared_error",
                    "squared_hinge", "perceptron", "hinge", "log_loss",
                    "modified_huber", "epsilon_insensitive"
                ],
                default_value="hinge"
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_penalty", choices=["l2", "l1", "elasticnet"], default_value="l2"
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_alpha", 1e-6, 1e-1, default_value=1e-4, log=True
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_learning_rate",
                choices=["constant", "optimal", "invscaling", "adaptive"],
                default_value="optimal"
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_l1_ratio", 0.0, 1.0, default_value=0.15
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_power_t", 0.0, 50.0, default_value=0.5
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_eta0", 1e-7, 1e-2, default_value=1e-3, log=True
            ),
        ))