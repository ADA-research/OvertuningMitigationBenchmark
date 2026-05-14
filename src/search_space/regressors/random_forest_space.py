"""
Random Forest Regressor Space Configuration.

This module defines the `RandomForestSpace` class, which is responsible
for constructing the hyperparameter search space for the RandomForestRegressor
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'RandomForestRegressor'.

Classes:
    - RandomForestSpace: Constructs the hyperparameter space for RandomForestRegressor.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class RandomForestSpace(BaseComponentSpace):
    """
    Defines the configuration space for the RandomForestRegressor model.

    This class builds the search space for fitting a RandomForestRegressor model
    via ConfigSpace. It includes hyperparameters such as the number of
    estimators, maximum tree depth, and criteria for splitting, which are
    constrained by the model being RandomForestRegressor in the specified pipeline.

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
        Initializes the RandomForestRegressor hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'RandomForestRegressor').
        """
        super().__init__(component_hp, name="RandomForest", seed=seed)

        # TabArena hyperparameter space (page 44 of paper)
        # https://arxiv.org/pdf/2506.16791

        self.add_hyperparameters((
            ConfigSpace.Constant(
                f"{self.name}_n_estimators", 50
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_max_features", lower=0.4, upper=1.0
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_min_samples_split", 2, 4, log=True
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_bootstrap", choices=[True, False], default_value=True
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_min_impurity_decrease", lower=0.00001, upper=0.001, log=True
            )
        ))

        # Add hyperparameter separately for max_samples - bootstrap condition
        self.space.add_hyperparameter(
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_max_samples", lower=0.5, upper=1.0
            ),
        )

        # Add condition for max_samples to be equal to bootstrap
        self.space.add_condition(
            ConfigSpace.AndConjunction(
            ConfigSpace.EqualsCondition(
                self.space.get_hyperparameter(f"{self.name}_max_samples"),
                self.space.get_hyperparameter(f"{self.name}_bootstrap"),
                True),
                ConfigSpace.EqualsCondition(
                    self.space.get_hyperparameter(f"{self.name}_max_samples"),
                    self.space.get_hyperparameter("model"),
                    "RandomForest")
            )
        )

        # # Add hyperparameters specific to RandomForestRegressor
        # self.add_hyperparameters((
        #     ConfigSpace.UniformIntegerHyperparameter(
        #         f"{self.name}_n_estimators", 10, 200, default_value=100
        #     ),
        #     ConfigSpace.UniformIntegerHyperparameter(
        #         f"{self.name}_max_depth", 1, 25, default_value=10
        #     ),
        #     ConfigSpace.UniformIntegerHyperparameter(
        #         f"{self.name}_min_samples_split", 2, 20, default_value=2
        #     ),
        #     ConfigSpace.UniformIntegerHyperparameter(
        #         f"{self.name}_min_samples_leaf", 1, 20, default_value=1
        #     ),
        #     ConfigSpace.CategoricalHyperparameter(
        #         f"{self.name}_criterion", choices=["squared_error", "absolute_error", "friedman_mse", "poisson"], default_value="squared_error"
        #     ),
        #     ConfigSpace.CategoricalHyperparameter(
        #         f"{self.name}_bootstrap", choices=[True, False], default_value=True
        #     ),
        #     ConfigSpace.CategoricalHyperparameter(
        #         f"{self.name}_max_features", choices=["sqrt", "log2", None], default_value="sqrt"
        #     )
        # ))
