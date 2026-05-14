from ConfigSpace import ConfigurationSpace

class BaseOptimizer:
    def __init__(self,
                 search_space: ConfigurationSpace = None,
                 random_state: int = None):
        
        self.search_space = search_space
        self.random_state = random_state
    
    def generate_configuration(self):
        pass