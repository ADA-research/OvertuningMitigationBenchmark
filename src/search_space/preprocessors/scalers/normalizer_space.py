"""
Normalizer Space Configuration.

This module defines the `NormalizerSpace` class, which constructs
the hyperparameter search space for the Normalizer preprocessor
using ConfigSpace. The Normalizer preprocessor exposes a single
hyperparameter, 'norm', which determines the normalization method.

Classes:
    - NormalizerSpace: Configuration space for the Normalizer preprocessor.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class NormalizerSpace(BaseComponentSpace):
    """
    Defines the configuration space for the Normalizer preprocessor.

    This class builds the configuration space for the Normalizer preprocessor,
    introducing the 'norm' hyperparameter which controls the type of
    normalization to apply.

    Inherits:
        BaseComponentSpace: Aligns preprocessor configuration handling in a consistent way.

    Attributes:
        component_hp (ConfigSpace.hyperparameter.Hyperparameter):
            Reference to the top-level component hyperparameter, which determines
            the preprocessor type in the pipeline.
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the Normalizer configuration space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the preprocessor
                type (e.g., 'Normalizer').
        """
        super().__init__(component_hp, name="Normalizer", seed=seed)

        self.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter(
                name=f"{self.name}_norm",
                choices=["l1", "l2", "max"],
                default_value="l2"
            )
        )