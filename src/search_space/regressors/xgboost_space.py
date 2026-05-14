"""
XGBoost Space Configuration.

This module defines the `XGBoostSpace` class, which is responsible
for constructing the hyperparameter search space for the XGBoost
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'XGBoost'.

Classes:
    - XGBoostSpace: Constructs the hyperparameter space for XGBoost.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace


class XGBoostSpace(BaseComponentSpace):
    """
    Defines the configuration space for the XGBoost model.

    This class builds the search space for fitting an XGBoost model
    via ConfigSpace. It includes hyperparameters such as the number of
    estimators, learning rate, tree depth, subsampling rates, and
    regularization parameters, which are constrained by the model
    being XGBoost in the specified pipeline.

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
        Initializes the XGBoost hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'XGBoost').
        """
        super().__init__(component_hp, name="XGBoost", seed=seed)

        # Add hyperparameters specific to XGBoost
        self.add_hyperparameters((
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_n_estimators", 50, 500, default_value=100
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_learning_rate", 0.01, 0.3, default_value=0.1, log=True
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_max_depth", 3, 10, default_value=6
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_subsample", 0.6, 1.0, default_value=1.0
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_colsample_bytree", 0.6, 1.0, default_value=1.0
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_reg_alpha", 0.0001, 10.0, default_value=0.0001, log=True
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_reg_lambda", 0.1, 10.0, default_value=1.0, log=True
            )
        ))
