"""
Extra Tree Regressor Space Configuration.

This module defines the `ExtraTreeSpace` class, which is responsible
for constructing the hyperparameter search space for the ExtraTreeRegressor
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'ExtraTreeRegressor'.

Classes:
    - ExtraTreeSpace: Constructs the hyperparameter space for ExtraTreeRegressor.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace


class ExtraTreeSpace(BaseComponentSpace):
    """
    Defines the configuration space for the ExtraTreeRegressor model.

    This class builds the search space for fitting an ExtraTreeRegressor model
    via ConfigSpace. It includes hyperparameters such as maximum depth, the
    splitting criterion, and sample thresholds. Additional hyperparameters
    are provided to enable fine-grained control.

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
        Initializes the ExtraTreeRegressor hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'ExtraTreeRegressor').
        """
        super().__init__(component_hp, name="ExtraTree", seed=seed)

        # Add hyperparameters specific to ExtraTreeRegressor
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_max_depth", 1, 150, default_value=None
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_criterion", choices=["squared_error", "absolute_error", "friedman_mse"], default_value="squared_error"
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_splitter", choices=["random", "best"], default_value="best"
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_min_samples_split", 2, 20, default_value=2
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_min_samples_leaf", 1, 20, default_value=1
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_min_weight_fraction_leaf", 0.0, 0.5, default_value=0.0
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_max_features", 1, 20, default_value=10
            )
        ))
