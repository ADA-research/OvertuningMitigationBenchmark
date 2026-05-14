import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class CategoricalSimpleImputerSpace(BaseComponentSpace):
    """
    Defines the configuration space for the CategoricalSimpleImputer.

    This creates a search space for:
    - Strategy: Options include 'constant' and 'most_frequent'.
    - Fill value: Custom value for the 'constant' strategy.
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the categorical SimpleImputer hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the imputer type
                (e.g., 'CategoricalSimpleImputer').
        """
        super().__init__(component_hp, name="CategoricalSimpleImputer", seed=seed)

        # Add CategoricalSimpleImputer-specific hyperparameters
        self.add_hyperparameters((
            # ConfigSpace.CategoricalHyperparameter(
            #     f"{self.name}_strategy", ["constant", "most_frequent"], default_value="most_frequent"
            # ),
            # For CASH we can tune preprocessing hyperparameters, for tuning one model this makes less sense
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_strategy", ["constant"]
            ),

            ConfigSpace.Constant(
                f"{self.name}_fill_value", value="missing_value"  # Default fill value for 'constant' strategy
            ),
        ))