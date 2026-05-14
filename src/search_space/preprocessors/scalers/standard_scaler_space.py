"""
StandardScaler Space Configuration.

This module defines the `StandardScalerSpace` class, which constructs
the hyperparameter search space for the StandardScaler preprocessor
using ConfigSpace. Since StandardScaler does not have configurable
hyperparameters, this class serves to integrate it seamlessly into the
pipeline.

Classes:
    - StandardScalerSpace: Skeleton for the StandardScaler configuration space.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class StandardScalerSpace(BaseComponentSpace):
    """
    Defines the configuration space for the StandardScaler preprocessor.

    This class builds a placeholder configuration space for the StandardScaler
    preprocessor. Since no hyperparameters are configurable for StandardScaler,
    this class acts as a skeleton for maintaining consistency in the pipeline's
    preprocessing steps.

    Inherits:
        BaseComponentSpace: Aligns preprocessor configuration handling in a consistent way.

    Attributes:
        component_hp (ConfigSpace.hyperparameter.Hyperparameter):
            Reference to the top-level component hyperparameter, which determines
            the preprocessor type in the pipeline.
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the StandardScaler configuration space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the preprocessor
                type (e.g., 'StandardScaler').
        """
        super().__init__(component_hp, name="StandardScaler", seed=seed)
