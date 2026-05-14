"""
Bagging Regressor Space Configuration.

This module defines the `BaggingSpace` class, which is responsible
for constructing the hyperparameter search space for the BaggingRegressor
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'BaggingRegressor'.

Classes:
    - BaggingSpace: Constructs the hyperparameter space for BaggingRegressor.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class BaggingSpace(BaseComponentSpace):
    """
    Defines the configuration space for the BaggingRegressor model.

    This class builds the search space for fitting a BaggingRegressor model
    via ConfigSpace. It includes the number of base estimators, the fraction
    of samples and features used for training base estimators, and bootstrapping
    options. Additional hyperparameters relevant to bagging are included for
    enhanced flexibility in optimization.

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
        Initializes the BaggingRegressor hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'BaggingRegressor').
        """
        super().__init__(component_hp, name="Bagging", seed=seed)

        # Add hyperparameters specific to BaggingRegressor
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_n_estimators", 10, 50, default_value=10
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_max_samples", 0.01, 1.0, default_value=1.0
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_max_features", 0.01, 1.0, default_value=1.0
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_bootstrap", choices=[True, False], default_value=True
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_bootstrap_features", choices=[True, False], default_value=False
            ),
        ))
