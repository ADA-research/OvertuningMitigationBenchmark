"""
KNeighbors Regressor Space Configuration.

This module defines the `KNeighborsSpace` class, which is responsible
for constructing the hyperparameter search space for the KNeighborsRegressor
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'KNeighborsRegressor'.

Classes:
    - KNeighborsSpace: Constructs the hyperparameter space for KNeighborsRegressor.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class KNeighborsSpace(BaseComponentSpace):
    """
    Defines the configuration space for the KNeighborsRegressor model.

    This class builds the search space for fitting a KNeighborsRegressor model
    via ConfigSpace. It includes hyperparameters such as the number of neighbors,
    weight strategy, distance metric, and more.

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
        Initializes the KNeighborsRegressor hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'KNeighborsRegressor').
        """
        super().__init__(component_hp, name="KNeighbors", seed=seed)

        # Add hyperparameters specific to KNeighborsRegressor
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_n_neighbors", 1, 100, default_value=5
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_weights", choices=["uniform", "distance"], default_value="uniform"
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_algorithm", choices=["ball_tree", "kd_tree", "auto"], default_value="auto"
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_leaf_size", 10, 50, default_value=30
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_p", 1, 3, default_value=2
            ),
        ))
