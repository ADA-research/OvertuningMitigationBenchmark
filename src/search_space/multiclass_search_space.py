import ConfigSpace
from ConfigSpace import ConfigurationSpace

from src.search_space.classifiers.adaboost_space import AdaBoostSpace
from src.search_space.classifiers.bagging_space import BaggingSpace
from src.search_space.classifiers.fastaimlp_space import FastAIMLPSpace
from src.search_space.classifiers.decision_tree_space import DecisionTreeSpace
from src.search_space.classifiers.extra_tree_space import ExtraTreeSpace
from src.search_space.classifiers.extra_trees_space import ExtraTreesSpace
from src.search_space.classifiers.gaussian_nb_space import GaussianNBSpace
from src.search_space.classifiers.gradient_boosting_space import GradientBoostingSpace
from src.search_space.classifiers.kneighbors_space import KNeighborsSpace
from src.search_space.classifiers.linear_discriminant_analysis_space import LinearDiscriminantAnalysisSpace
from src.search_space.classifiers.logistic_regression_space import LogisticRegressionSpace
from src.search_space.classifiers.mlp_space import MLPSpace
from src.search_space.classifiers.passive_agressive_space import PassiveAggressiveSpace
from src.search_space.classifiers.random_forest_space import RandomForestSpace
from src.search_space.classifiers.realmlp_space import RealMLPSpace
from src.search_space.classifiers.ridge_space import RidgeSpace
from src.search_space.classifiers.sgd_space import SGDSpace
from src.search_space.classifiers.xgboost_space import XGBoostSpace
from src.search_space.classifiers.lgbm_space import LGBMSpace
from src.search_space.classifiers.catboost_space import CatBoostSpace

# Note: Lasso and ElasticNet are excluded as they are regression models, not classifiers
namespace = {
    'AdaBoost': AdaBoostSpace,
    'Bagging': BaggingSpace,
    'CatBoost': CatBoostSpace,
    'DecisionTree': DecisionTreeSpace,
    'ExtraTree': ExtraTreeSpace,
    'ExtraTrees': ExtraTreesSpace,
    'FastAIMLP': FastAIMLPSpace,
    'GaussianNB': GaussianNBSpace,
    'GradientBoosting': GradientBoostingSpace,
    'KNeighbors': KNeighborsSpace,
    'LDA': LinearDiscriminantAnalysisSpace,
    'LGBM': LGBMSpace,
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysisSpace,
    'LogisticRegression': LogisticRegressionSpace,
    'MLP': MLPSpace,
    'RealMLP': RealMLPSpace,
    'PassiveAggressive': PassiveAggressiveSpace,
    'RandomForest': RandomForestSpace,
    'Ridge': RidgeSpace,
    'SGD': SGDSpace,
    'XGBoost': XGBoostSpace,
}

class MulticlassSearchSpace:
    """
    Represents a multiclass search space for model classifiers.

    This class is designed to provide a search space for multiclass classifiers
    and their respective hyperparameters. It leverages a configuration space
    to facilitate the definition and management of search spaces. The
    `MulticlassSearchSpace` class is intended for use in optimization tasks where
    classifier models and their parameters are selected for improving
    performance or accuracy on multiclass classification problems.

    All classifiers included support multiclass classification either natively
    or through One-vs-All/One-vs-One strategies. Note that Lasso and ElasticNet
    are excluded as they are regression models, not classification models.

    :ivar config: The configuration object containing settings and information
                  related to classifiers.
    :type config: Any
    :ivar space: The configuration search space that combines all classifier-
                 specific hyperparameter spaces.
    :type space: ConfigurationSpace
    :ivar classifier_hp: A categorical hyperparameter for selecting classifiers
                         from the defined configuration.
    :type classifier_hp: ConfigSpace.CategoricalHyperparameter
    :ivar classifier_spaces: A list of individual classifier-specific search
                             spaces constructed based on the provided
                             configuration.
    :type classifier_spaces: list
    """
    def __init__(self, config=None, seed=0):
        self.config = config
        classifiers = config.search_space.classifiers

        self.space = ConfigurationSpace(name="MulticlassClassificationSearchSpace", seed=seed)

        self.classifier_hp = ConfigSpace.CategoricalHyperparameter(name="model", choices=classifiers)

        self.space.add(self.classifier_hp)

        self.classifier_spaces = [
            namespace[x] for x in classifiers
        ]

        for classifier_space in self.classifier_spaces:

            initialized_classifier_space = classifier_space(self.classifier_hp, seed=seed)

            self.space.add(initialized_classifier_space.get_hyperparameters())
            self.space.add(initialized_classifier_space.space.conditions)