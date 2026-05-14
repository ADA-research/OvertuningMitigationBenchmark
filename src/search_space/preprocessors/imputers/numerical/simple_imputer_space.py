import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class NumericalSimpleImputerSpace(BaseComponentSpace):
    """
    Defines the configuration space for the SimpleImputer.

    This creates a search space for:
    - Strategy: Defines how missing values are imputed. Options include 'mean', 'median', and 'constant'.
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the SimpleImputer hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the imputer type
                (e.g., 'SimpleImputer').
        """
        super().__init__(component_hp, name="NumericalSimpleImputer", seed=seed)

        # Add SimpleImputer-specific hyperparameters
        self.add_hyperparameters((
            # ConfigSpace.CategoricalHyperparameter(
            #     f"{self.name}_strategy", ["mean", "median", "constant"], default_value="mean"
            # ),
            ConfigSpace.Constant(
                f"{self.name}_strategy", value="mean"
            ),
        ))