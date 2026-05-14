"""
Lasso Classifier Space Configuration.

This module defines the `LassoSpace` class, which is responsible
for constructing the hyperparameter search space for the Lasso regression
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'Lasso'.

Classes:
    - LassoSpace: Constructs the hyperparameter space for Lasso regression.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class LassoSpace(BaseComponentSpace):
    """
    Defines the configuration space for the Lasso model.

    This class builds the search space for fitting a Lasso model
    via ConfigSpace. It includes the alpha (regularization strength) hyperparameter
    to control model fitting and regularization.

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
        Initializes the Lasso hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'Lasso').
        """
        super().__init__(component_hp, name="Lasso", seed=seed)

        # Add hyperparameters specific to Lasso
        self.add_hyperparameters((
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_alpha", 0.0001, 1.0, default_value=0.01, log=True
            ),
        ))