"""
AdaBoost Space Configuration.

This module defines the `AdaBoostSpace` class, which is responsible
for constructing the hyperparameter search space for the AdaBoost
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'AdaBoost'.

Classes:
    - AdaBoostSpace: Constructs the hyperparameter space for AdaBoost.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class AdaBoostSpace(BaseComponentSpace):
    """
    Defines the configuration space for the AdaBoost model.

    This class builds the search space for fitting an AdaBoost model
    via ConfigSpace. It includes hyperparameters such as the number of
    estimators and learning rate, which are constrained by the model
    being AdaBoost in the specified pipeline.

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
        Initializes the AdaBoost hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'AdaBoost').
        """
        super().__init__(component_hp, name="AdaBoost", seed=seed)

        # Add hyperparameters specific to AdaBoost
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_n_estimators", 10, 250, default_value=50
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_learning_rate", 0.001, 2.0, default_value=0.1, log=True
            )
        ))
