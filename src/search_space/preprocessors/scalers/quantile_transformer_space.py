"""
QuantileTransformer Space Configuration.

This module defines the `QuantileTransformerSpace` class, which is responsible
for constructing the hyperparameter search space for the QuantileTransformer
preprocessor using ConfigSpace. All hyperparameters added to this space
are conditional on the component being set to 'QuantileTransformer'.

Classes:
    - QuantileTransformerSpace: Constructs the hyperparameter space for QuantileTransformer.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class QuantileTransformerSpace(BaseComponentSpace):
    """
    Defines the configuration space for the QuantileTransformer preprocessor.

    This class builds the search space for the QuantileTransformer preprocessor
    using ConfigSpace. It focuses on two main configurable parameters,
    `n_quantiles` and `output_distribution`, to allow customization of feature
    scaling and distribution transformations.

    Inherits:
        BaseComponentSpace: Handles conditional addition of hyperparameters
        dependent on the preprocessor choice.

    Attributes:
        component_hp (ConfigSpace.hyperparameter.Hyperparameter):
            Reference to the top-level component hyperparameter, which determines
            the preprocessor type in the pipeline.
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the QuantileTransformer hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the preprocessor
                type (e.g., 'QuantileTransformer').
        """
        super().__init__(component_hp, name="QuantileTransformer", seed=seed)

        # Add hyperparameters specific to QuantileTransformer
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_n_quantiles", 10, 1000, default_value=1000
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_output_distribution", choices=["uniform", "normal"], default_value="uniform"
            ),
        ))