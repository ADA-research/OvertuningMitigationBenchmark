from src.models.base_component import BaseComponent
from sklearn.preprocessing import OrdinalEncoder


class OrdinalEncoderComponent(BaseComponent):
    def __init__(self):
        super().__init__("OrdinalEncoder")
        # OrdinalEncoder does not require any specific hyperparameters
        self.required_hyperparameters = []

    def construct(self, config):
        """
        Constructs an instance of OrdinalEncoder without requiring specific hyperparameters.
        """
        self.check(config)

        return OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
