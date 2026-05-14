"""
Ridge Regressor Space Configuration.

This module defines the `RidgeSpace` class, which is responsible
for constructing the hyperparameter search space for the Ridge
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'Ridge'.

Classes:
    - RidgeSpace: Constructs the hyperparameter space for Ridge.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class RidgeSpace(BaseComponentSpace):
    """
    Defines the configuration space for the Ridge model.

    This class builds the search space for fitting a Ridge model
    via ConfigSpace. It includes hyperparameters such as regularization strength
    and solver choice. Additional control is provided for fine-tuning the regressor.

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
        Initializes the Ridge hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'Ridge').
        """
        super().__init__(component_hp, name="Ridge", seed=seed)

        # Add hyperparameters specific to Ridge
        self.add_hyperparameters((
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_alpha", 0.01, 10.0, default_value=1.0, log=True
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_solver",
                choices=["auto", "svd", "cholesky", "lsqr", "sag", "saga"],
                default_value="auto"
            )
        ))
