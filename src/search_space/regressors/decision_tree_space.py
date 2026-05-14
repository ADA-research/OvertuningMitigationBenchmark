"""
Decision Tree Regressor Space Configuration.

This module defines the `DecisionTreeSpace` class, which is responsible
for constructing the hyperparameter search space for the DecisionTreeRegressor
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'DecisionTreeRegressor'.

Classes:
    - DecisionTreeSpace: Constructs the hyperparameter space for DecisionTreeRegressor.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class DecisionTreeSpace(BaseComponentSpace):
    """
    Defines the configuration space for the DecisionTreeRegressor model.

    This class builds the search space for fitting a DecisionTreeRegressor model
    via ConfigSpace. It includes hyperparameters such as maximum depth, minimum
    samples required for splits, and the criterion for measuring split quality.
    Additional relevant hyperparameters are included for enhanced flexibility in optimization.

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
        Initializes the DecisionTreeRegressor hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'DecisionTreeRegressor').
        """
        super().__init__(component_hp, name="DecisionTree", seed=seed)

        # Add hyperparameters specific to DecisionTreeRegressor
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_max_depth", 1, 25, default_value=10
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_min_samples_split", 2, 20, default_value=2
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_min_samples_leaf", 1, 20, default_value=1
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_criterion", choices=["squared_error", "absolute_error", "friedman_mse", "poisson"], default_value="squared_error"
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_splitter", choices=["best", "random"], default_value="best"
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_min_weight_fraction_leaf", 0.0, 0.5, default_value=0.0
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_max_features", 1, 100, default_value=10
            )
        ))
