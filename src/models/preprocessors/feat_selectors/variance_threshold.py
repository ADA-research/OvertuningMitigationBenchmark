from sklearn.feature_selection import VarianceThreshold

from src.models.base_component import BaseComponent
from src.models.preprocessors.feat_selectors.keep_n_features_wrapper import KeepNFeaturesWrapper

class VarianceThresholdComponent(BaseComponent):
    def __init__(self):
        super().__init__("VarianceThreshold")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_threshold"
        ]

    def construct(self, config):
        """Construct VarianceThreshold feature selector."""
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix)
        }

        self.check(self.config)

        return KeepNFeaturesWrapper(
            VarianceThreshold(
                threshold=self.config[f"{self.prefix}_threshold"]
            ),
            threshold=self.config[f"{self.prefix}_threshold"]
        )
