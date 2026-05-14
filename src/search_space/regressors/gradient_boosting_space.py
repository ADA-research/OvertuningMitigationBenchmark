"""
Gradient Boosting Regressor Space Configuration.

This module defines the `GradientBoostingSpace` class, which is responsible
for constructing the hyperparameter search space for the GradientBoostingRegressor
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'GradientBoostingRegressor'.

Classes:
    - GradientBoostingSpace: Constructs the hyperparameter space for GradientBoostingRegressor.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class GradientBoostingSpace(BaseComponentSpace):
    """
    Defines the configuration space for the GradientBoostingRegressor model.

    This class builds the search space for fitting a GradientBoostingRegressor model
    via ConfigSpace. It includes hyperparameters such as the number of estimators, learning
    rate, max depth, and others. Additional hyperparameters relevant to gradient boosting
    are also included to allow for a thorough exploration of the parameter space.

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
        Initializes the GradientBoostingRegressor hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'GradientBoostingRegressor').
        """
        super().__init__(component_hp, name="GradientBoosting", seed=seed)

        # Add hyperparameters specific to GradientBoostingRegressor
        self.add_hyperparameters((
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_loss",
                choices=["squared_error", "absolute_error", "huber", "quantile"],
                default_value="squared_error"
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_n_estimators", 10, 200, default_value=100
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_learning_rate", 0.001, 1.0, default_value=0.1, log=True
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_max_depth", 1, 15, default_value=3
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_subsample", 0.05, 1.0, default_value=1.0
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_min_samples_split", 2, 20, default_value=2
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_min_samples_leaf", 1, 20, default_value=1
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_max_features", 1, 100, default_value=10
            )
        ))
