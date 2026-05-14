class BaseComponent:
    def __init__(self, prefix: str):
        self.prefix: str = prefix
        self.config = None
        self.required_hyperparameters = []

    def construct(self, config):
        pass

    def reset(self):
        pass

    def check(self, config):
        """Function to verify all required hyperparameters are present in the config
        :param config: Configuration object
        :return:
        """
        # Find missing hyperparameters
        missing_hyperparameters = [k for k in self.required_hyperparameters if k not in config]

        # Raise error if any missing hyperparameters are found
        if len(missing_hyperparameters) > 0:
            raise ValueError(f"Hyperparameters: {missing_hyperparameters} not found in config.")

        return True