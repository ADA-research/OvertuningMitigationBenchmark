from src.optimizers.base_optimizer import BaseOptimizer
from ConfigSpace import ConfigurationSpace


class RandomSearch(BaseOptimizer):
    def __init__(
            self,
            search_space: ConfigurationSpace,
            random_state: int = None
    ):
        super().__init__(
            search_space,
            random_state
        )

    def generate_configuration(self):
        config = dict(self.search_space.sample_configuration())
        config["random_state"] = self.random_state
        return config, 0.0
