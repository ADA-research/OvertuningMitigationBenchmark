import ConfigSpace
from ConfigSpace import ConfigurationSpace

from src.search_space.regressors.adaboost_space import AdaBoostSpace
from src.search_space.regressors.bagging_space import BaggingSpace
from src.search_space.regressors.catboost_space import CatBoostSpace
from src.search_space.regressors.decision_tree_space import DecisionTreeSpace
from src.search_space.regressors.elasticnet_space import ElasticNetSpace
from src.search_space.regressors.extra_tree_space import ExtraTreeSpace
from src.search_space.regressors.extra_trees_space import ExtraTreesSpace
from src.search_space.regressors.fastaimlp_space import FastAIMLPSpace
from src.search_space.regressors.gradient_boosting_space import GradientBoostingSpace
from src.search_space.regressors.kneighbors_space import KNeighborsSpace
from src.search_space.regressors.lasso_space import LassoSpace
from src.search_space.regressors.lgbm_space import LGBMSpace
from src.search_space.regressors.mlp_space import MLPSpace
from src.search_space.regressors.random_forest_space import RandomForestSpace
from src.search_space.regressors.realmlp_space import RealMLPSpace
from src.search_space.regressors.ridge_space import RidgeSpace
from src.search_space.regressors.sgd_space import SGDSpace
from src.search_space.regressors.xgboost_space import XGBoostSpace

namespace = {
    'AdaBoost': AdaBoostSpace,
    'Bagging': BaggingSpace,
    'CatBoost': CatBoostSpace,
    'DecisionTree': DecisionTreeSpace,
    'ElasticNet': ElasticNetSpace,
    'ExtraTree': ExtraTreeSpace,
    'ExtraTrees': ExtraTreesSpace,
    'FastAIMLP': FastAIMLPSpace,
    'GradientBoosting': GradientBoostingSpace,
    'KNeighbors': KNeighborsSpace,
    'Lasso': LassoSpace,
    'LGBM': LGBMSpace,
    'MLP': MLPSpace,
    'RealMLP': RealMLPSpace,
    'RandomForest': RandomForestSpace,
    'Ridge': RidgeSpace,
    'SGD': SGDSpace,
    'XGBoost': XGBoostSpace,
}

class RegressionSearchSpace:
    """
    Represents a regression search space for model regressors.

    This class is designed to provide a search space for regression models
    and their respective hyperparameters. It leverages a configuration space
    to facilitate the definition and management of search spaces. The
    `RegressionSearchSpace` class is intended for use in optimization tasks where
    regressor models and their parameters are selected for improving
    performance or accuracy.

    :ivar config: The configuration object containing settings and information
                  related to regressors.
    :type config: Any
    :ivar space: The configuration search space that combines all regressor-
                 specific hyperparameter spaces.
    :type space: ConfigurationSpace
    :ivar regressor_hp: A categorical hyperparameter for selecting regressors
                        from the defined configuration.
    :type regressor_hp: ConfigSpace.CategoricalHyperparameter
    :ivar regressor_spaces: A list of individual regressor-specific search
                            spaces constructed based on the provided
                            configuration.
    :type regressor_spaces: list
    """
    def __init__(self, config=None, seed=0):
        self.config = config
        regressors = config.search_space.regressors

        self.space = ConfigurationSpace(name="RegressionSearchSpace", seed=seed)

        self.regressor_hp = ConfigSpace.CategoricalHyperparameter(name="model", choices=regressors)

        self.space.add(self.regressor_hp)

        self.regressor_spaces = [
            namespace[x] for x in regressors
        ]

        for regressor_space in self.regressor_spaces:
            initialized_regressor_space = regressor_space(self.regressor_hp, seed=seed)

            self.space.add(initialized_regressor_space.get_hyperparameters())
            self.space.add(initialized_regressor_space.space.conditions)
