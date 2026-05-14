import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class SelectPercentileSpace(BaseComponentSpace):
    """
    Defines the configuration space for the SelectPercentile feature selector.

    This creates a search space for:
    - percentile: Percentage of features to select.
    - score_func: The scoring function to rank features.
    """

    def __init__(self, component_hp, problem_type=None, seed=0):
        """
        Initializes the SelectPercentile hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the feature
                selector type (e.g., 'SelectPercentile').

            problem_type (str):
                The problem type ('regression' or otherwise). Determines the score_func.
        """
        super().__init__(component_hp, name="SelectPercentile", seed=seed)

        # Define score functions based on the problem type
        if problem_type == "regression":
            score_func_choices = ["f_regression", "mutual_info_regression"]
        else:
            score_func_choices = ["f_classif", "mutual_info_classif"]

        # Add SelectPercentile-specific hyperparameters
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_percentile", 10, 100, default_value=50
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_score_func", choices=score_func_choices, default_value=score_func_choices[0]
            ),
        ))