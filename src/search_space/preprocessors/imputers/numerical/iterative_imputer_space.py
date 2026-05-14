import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class IterativeImputerSpace(BaseComponentSpace):
    """
    Defines the configuration space for the IterativeImputer.

    This creates a search space for:
    - Maximum iterations
    - Imputation order
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the IterativeImputer hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the imputer type
                (e.g., 'IterativeImputer').
        """
        super().__init__(component_hp, name="IterativeImputer", seed=seed)

        # Add IterativeImputer-specific hyperparameters
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_max_iter", 10, 100, default_value=50  # Maximum number of iterations
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_imputation_order",
                ["ascending", "descending", "roman", "arabic"],
                default_value="ascending"  # The order in which features should be imputed
            )
        ))