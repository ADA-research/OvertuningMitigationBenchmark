"""
GaussianNB Classifier Space Configuration.

This module defines the `GaussianNBSpace` class, which is responsible
for constructing the hyperparameter search space for the GaussianNB
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'GaussianNB'.

Classes:
    - GaussianNBSpace: Constructs the hyperparameter space for GaussianNB.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class GaussianNBSpace(BaseComponentSpace):
    """
    Defines the configuration space for the GaussianNB model.

    This class builds the search space for fitting a GaussianNB model
    via ConfigSpace. It includes hyperparameters such as `var_smoothing`
    which adjusts variances to account for numerical stability.

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
        Initializes the GaussianNB hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'GaussianNB').
        """
        super().__init__(component_hp, name="GaussianNB", seed=seed)

        # Add hyperparameters specific to GaussianNB
        self.add_hyperparameters((
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_var_smoothing", 1e-12, 1e-6, default_value=1e-9, log=True
            ),
        ))