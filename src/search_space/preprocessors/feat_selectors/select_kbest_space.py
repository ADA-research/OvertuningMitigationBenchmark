import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class SelectKBestSpace(BaseComponentSpace):
    """
    Defines the configuration space for the SelectKBest feature selector.

    This creates a search space for:
    - k: Number of top features to select.
    - score_func: The scoring function to rank features.
    """

    def __init__(self, component_hp, problem_type, seed=0):
        """
        Initializes the SelectKBest hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the feature
                selector type (e.g., 'SelectKBest').

            problem_type (str):
                The problem type ('regression' or otherwise). Determines the score_func.
        """
        super().__init__(component_hp, name="SelectKBest", seed=seed)

        # Define score functions based on the problem type
        if problem_type == "regression":
            score_func_choices = ["f_regression", "mutual_info_regression"]
        else:
            score_func_choices = ["f_classif", "mutual_info_classif"]

        # Add SelectKBest-specific hyperparameters
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_k", 5, 50, default_value=10
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_score_func", choices=score_func_choices, default_value=score_func_choices[0]
            ),
        ))