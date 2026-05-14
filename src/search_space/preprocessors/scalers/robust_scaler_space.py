"""
RobustScaler Space Configuration.

This module defines the `RobustScalerSpace` class, which is responsible
for constructing the hyperparameter search space for the RobustScaler
preprocessor using ConfigSpace. All hyperparameters added to this space
are conditional on the component being set to 'RobustScaler'.

Classes:
    - RobustScalerSpace: Constructs the hyperparameter space for RobustScaler.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class RobustScalerSpace(BaseComponentSpace):
    """
    Defines the configuration space for the RobustScaler preprocessor.

    This class builds the search space for the RobustScaler preprocessor
    using ConfigSpace. It focuses on the configurable parameter `quantile_range`,
    used for scaling features robustly to outliers.

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
        Initializes the RobustScaler hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the preprocessor
                type (e.g., 'RobustScaler').
        """
        super().__init__(component_hp, name="RobustScaler", seed=seed)

        # Add hyperparameters specific to RobustScaler
        self.add_hyperparameters((
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_quantile_range_lower", 0.0, 100.0, default_value=25.0
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_quantile_range_upper", 0.0, 100.0, default_value=75.0
            ),
        ))