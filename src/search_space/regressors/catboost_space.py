"""
CatBoost Space Configuration.

This module defines the `CatBoostSpace` class, which is responsible
for constructing the hyperparameter search space for the CatBoost
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'CatBoost'.

Classes:
    - CatBoostSpace: Constructs the hyperparameter space for CatBoost.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class CatBoostSpace(BaseComponentSpace):
    """
    Defines the configuration space for the CatBoost model.

    This class builds the search space for fitting a CatBoost model
    via ConfigSpace. It includes hyperparameters such as the number of
    iterations, tree depth, learning rate, and regularization parameters,
    which are constrained by the model being CatBoost in the specified pipeline.

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
        Initializes the CatBoost hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'CatBoost').
        """
        super().__init__(component_hp, name="CatBoost", seed=seed)

        # Add hyperparameters specific to CatBoost
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_iterations", 100, 1000, default_value=500
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_depth", 4, 10, default_value=6
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_learning_rate", 0.01, 0.3, default_value=0.1, log=True
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_l2_leaf_reg", 1, 10, default_value=3
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_border_count", 32, 255, default_value=128
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_bagging_temperature", 0, 1, default_value=1
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_random_strength", 0, 10, default_value=1
            )
        ))
