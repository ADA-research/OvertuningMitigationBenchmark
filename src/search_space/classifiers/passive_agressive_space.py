"""
Passive Aggressive Classifier Space Configuration.

This module defines the `PassiveAggressiveSpace` class, which is responsible
for constructing the hyperparameter search space for the PassiveAggressiveClassifier
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'PassiveAggressiveClassifier'.

Classes:
    - PassiveAggressiveSpace: Constructs the hyperparameter space for PassiveAggressiveClassifier.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class PassiveAggressiveSpace(BaseComponentSpace):
    """
    Defines the configuration space for the PassiveAggressiveClassifier model.

    This class builds the search space for fitting a PassiveAggressiveClassifier model
    via ConfigSpace. It includes hyperparameters such as regularization strength,
    maximum iterations, and tolerance. Additional hyperparameters are included for
    full control over the model behavior.

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
        Initializes the PassiveAggressiveClassifier hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'PassiveAggressiveClassifier').
        """
        super().__init__(component_hp, name="PassiveAggressive", seed=seed)

        # Add hyperparameters specific to PassiveAggressiveClassifier
        self.add_hyperparameters((
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_C", 0.01, 10.0, default_value=1.0, log=True
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_max_iter", 1, 3000, default_value=1000, log=True
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_tol", 1e-4, 1e-1, default_value=1e-3, log=True
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_fit_intercept", choices=[True, False], default_value=True
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_shuffle", choices=[True, False], default_value=True
            )
        ))