"""
ElasticNet Regression Space Configuration.

This module defines the `ElasticNetSpace` class, which is responsible
for constructing the hyperparameter search space for the ElasticNet regression
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'ElasticNet'.

Classes:
    - ElasticNetSpace: Constructs the hyperparameter space for ElasticNet regression.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class ElasticNetSpace(BaseComponentSpace):
    """
    Defines the configuration space for the ElasticNet model.

    This class builds the search space for fitting an ElasticNet model
    via ConfigSpace. It includes hyperparameters such as the regularization strength
    (`alpha`) and the L1 ratio (`l1_ratio`), which controls the mix of L1 and L2 penalties.

    Inherits:
        BaseComponentSpace: Handles conditional addition of hyperparameters
        dependent on the model choice.

    Attributes:
        component_hp (ConfigSpace.hyperparameter.Hyperparameter):
            Reference to the top-level component hyperparameter, which determines
            the model type in the pipeline.
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the ElasticNet hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'ElasticNet').
        """
        super().__init__(component_hp, name="ElasticNet", seed=seed)

        # Add hyperparameters specific to ElasticNet
        self.add_hyperparameters((
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_alpha", 0.0001, 1.0, default_value=0.01, log=True
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_l1_ratio", 0.0, 1.0, default_value=0.5
            ),
        ))
