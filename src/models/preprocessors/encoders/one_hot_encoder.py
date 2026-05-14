from src.models.base_component import BaseComponent
from sklearn.preprocessing import OneHotEncoder


class OneHotEncoderComponent(BaseComponent):
    def __init__(self):
        super().__init__("OneHotEncoder")
        # OneHotEncoder does not require any specific hyperparameters
        self.required_hyperparameters = []

    def construct(self, config):
        """
        Constructs an instance of OneHotEncoder without requiring specific hyperparameters.
        By default: sparse=False for easier handling in pandas/numpy.
        """
        self.check(config)

        return OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )