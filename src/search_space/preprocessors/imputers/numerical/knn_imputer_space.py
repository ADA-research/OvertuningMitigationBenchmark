import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class KNNImputerSpace(BaseComponentSpace):
    """
    Defines the configuration space for the KNNImputer.

    This creates a search space for:
    - Number of neighbors
    - Weights for distance calculation
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the KNNImputer hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the imputer type
                (e.g., 'KNNImputer').
        """
        super().__init__(component_hp, name="KNNImputer", seed=seed)

        # Add KNNImputer-specific hyperparameters
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_n_neighbors", 1, 10, default_value=5  # Number of neighbors
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_weights", ["uniform", "distance"], default_value="uniform"  # Weight function
            ),
        ))