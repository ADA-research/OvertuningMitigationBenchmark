import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class VarianceThresholdSpace(BaseComponentSpace):
    """
    Defines the configuration space for the VarianceThreshold feature selector.

    This creates a search space for:
    - variance_threshold: The threshold below which features are removed.
    """

    def __init__(self, component_hp, problem_type, seed=0):
        """
        Initializes the VarianceThreshold hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the feature
                selector type (e.g., 'VarianceThreshold').
        """
        super().__init__(component_hp, name="VarianceThreshold", seed=seed)

        # Add VarianceThreshold-specific hyperparameter
        self.add_hyperparameter(
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_threshold", 0.0, 0.05, default_value=0.0
            )
        )