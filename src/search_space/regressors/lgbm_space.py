"""
LightGBM Space Configuration.

This module defines the `LGBMSpace` class, which is responsible
for constructing the hyperparameter search space for the LightGBM
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'LGBM'.

Classes:
    - LGBMSpace: Constructs the hyperparameter space for LightGBM.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

import math

class LGBMSpace(BaseComponentSpace):
    """
    Defines the configuration space for the LightGBM model.

    This class builds the search space for fitting a LightGBM model
    via ConfigSpace. It includes hyperparameters such as the number of
    estimators, learning rate, number of leaves, feature and bagging
    fractions, and regularization parameters, which are constrained by
    the model being LGBM in the specified pipeline.

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
        Initializes the LightGBM hyperparameter search space.

        We adapt the search space from https://arxiv.org/pdf/2407.04491, page 43:
        Table C.10: Hyperparameter seach space for LGBM-HPO, adapted from Prokhorenkova et al. [51]:

        Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and
        Andrey Gulin. CatBoost: Unbiased boosting with categorical features. In Neural Information
        Processing Systems, 2018.

        with 1000 estimators instead of 5000:

        n_estimators: 1000
        bagging_freq: 1
        early_stopping_rounds: 300
        num_leaves: LogUniformInt[1, e7]
        learning_rate: LogUniform[e-7, 1]
        subsample: Uniform[0.5, 1]
        feature_fraction: Uniform[0.5, 1]
        min_data_in_leaf: LogUniformInt[1, e6]
        min_sum_hessian_in_leaf: LogUniform[e-16, e5]
        lambda_l1: Random{0, LogUniform[e-16, e2]}
        lambda_l2: Random{0, LogUniform[e-16, e2]}
        """

        super().__init__(component_hp, name="LGBM", seed=seed)

        # Search space from PyTabkit paper
        # Fixed hyperparameters
        self.add_hyperparameters((
            ConfigSpace.Constant(
                f"{self.name}_n_estimators", 1000
            ),
            ConfigSpace.Constant(
                f"{self.name}_bagging_freq", 1
            ),
        ))

        # Continuous hyperparameters
        self.add_hyperparameters((
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_learning_rate",
                lower=math.exp(-7),
                upper=1.0,
                default_value=0.01,
                log=True
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_num_leaves", lower=2, upper=1097,
                default_value=31, log=True
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_feature_fraction", lower=0.5, upper=1.0, default_value=1.0
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_bagging_fraction", lower=0.5, upper=1.0, default_value=1.0
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_min_data_in_leaf", lower=1, upper=403,
                default_value=20, log=True
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_min_sum_hessian_in_leaf",
                lower=math.exp(-16),
                upper=math.exp(5),
                default_value=1e-3,
                log=True
            ),
        ))

        # Conditional regularization parameters (can be 0 or log-uniform)
        lambda_l1_use = ConfigSpace.CategoricalHyperparameter(
            f"{self.name}_lambda_l1_use", choices=[False, True], default_value=False
        )
        lambda_l1_value = ConfigSpace.UniformFloatHyperparameter(
            f"{self.name}_lambda_l1_value",
            lower=math.exp(-16),
            upper=math.exp(2),
            default_value=1e-3,
            log=True
        )

        lambda_l2_use = ConfigSpace.CategoricalHyperparameter(
            f"{self.name}_lambda_l2_use", choices=[False, True], default_value=False
        )
        lambda_l2_value = ConfigSpace.UniformFloatHyperparameter(
            f"{self.name}_lambda_l2_value",
            lower=math.exp(-16),
            upper=math.exp(2),
            default_value=1e-3,
            log=True
        )

        self.add_hyperparameters((lambda_l1_use, lambda_l2_use))

        # Directly add regularization parameters to the underlying space
        self.space.add((lambda_l1_value, lambda_l2_value))

        # Add conditions: lambda values are only active when their corresponding 'use' flag is True
        self.space.add((
            ConfigSpace.AndConjunction(
                ConfigSpace.EqualsCondition(lambda_l1_value, lambda_l1_use, True),
                ConfigSpace.EqualsCondition(lambda_l1_value, self.space.get_hyperparameter("model"), "LGBM")
            ),
            ConfigSpace.AndConjunction(
                ConfigSpace.EqualsCondition(lambda_l2_value, lambda_l2_use, True),
                ConfigSpace.EqualsCondition(lambda_l2_value, self.space.get_hyperparameter("model"), "LGBM")
            ),
        ))
