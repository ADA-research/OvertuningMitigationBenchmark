import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class PCASpace(BaseComponentSpace):
    """
    Defines the configuration space for the PCA dimensionality reduction method.

    This creates search space for two hyperparameters:
    - Number of components
    - Whiten option
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the PCA hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the reducer
                type (e.g., 'PCA').
        """
        super().__init__(component_hp, name="PCA", seed=seed)

        # Add PCA-specific hyperparameters
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_n_components", 1, 50  # Number of PCA components
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_whiten", [True, False], default_value=False  # Whitening option
            ),
        ))