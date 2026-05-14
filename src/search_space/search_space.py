import ConfigSpace
from ConfigSpace.hyperparameters import Constant
from src.search_space.preprocessing_search_space import PreprocessingSearchSpace
from src.search_space.binary_search_space import BinarySearchSpace
from src.search_space.multiclass_search_space import MulticlassSearchSpace
from src.search_space.regression_search_space import RegressionSearchSpace
from src.search_space import forbidden
from src.utils.config import config_dict_to_namespace


class SearchSpace:
    def __init__(self, config):
        if isinstance(config, dict):
            config = config_dict_to_namespace(config)

        self.preprocessor_space = PreprocessingSearchSpace(config)

        if config.problem_type == 'binary':
            self.predictor_space = BinarySearchSpace(config)
        elif config.problem_type == 'multiclass':
            self.predictor_space = MulticlassSearchSpace(config)
        elif config.problem_type == 'regression':
            self.predictor_space = RegressionSearchSpace(config)
        else:
            raise ValueError(f"Problem type {config.problem_type} is not supported.")

        self.space = ConfigSpace.ConfigurationSpace(name="SearchSpace", seed=config.random_state)
        self.space.add(Constant("random_state", value=config.random_state))

        self.space.add(self.preprocessor_space.space.values())
        self.space.add(self.preprocessor_space.space.conditions)

        self.space.add(self.predictor_space.space.values())
        self.space.add(self.predictor_space.space.conditions)

        forbidden_clauses = forbidden.get_forbidden_clauses(self.space, config.problem_type)

        self.space.add(forbidden_clauses)


    def get_space(self):
        """Returns the search space."""
        return self.space
