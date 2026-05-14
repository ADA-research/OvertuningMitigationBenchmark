import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class FastICASpace(BaseComponentSpace):
    """
    Defines the configuration space for the FastICA dimensionality reduction method.

    This creates search spaces for the following hyperparameters:
    - Number of components
    - Algorithm type
    - Maximum number of iterations
    - Function used in the FastICA computation
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the FastICA hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the reducer type
                (e.g., 'FastICA').
        """
        super().__init__(component_hp, name="FastICA", seed=seed)

        # Add FastICA-specific hyperparameters
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_n_components", 1, 50
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_algorithm", ["parallel", "deflation"], default_value="parallel"  # Algorithm type
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_max_iter", 100, 200, default_value=200  # Maximum iterations
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_fun", ["logcosh", "exp"], default_value="logcosh"  # FastICA function
            ),
        ))