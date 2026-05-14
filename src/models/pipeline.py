import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector

from src.models.base_component import BaseComponent
from src.models.classifiers.fastaimlp import FastAIMLPModel
from src.models.classifiers.realmlp import RealMLPModel
from src.models.classifiers.xgboost import XGBoostModel

# Dimensionality Reducers
from src.models.preprocessors.dim_reducers.fastica_reducer import FastICAComponent
from src.models.preprocessors.dim_reducers.pca_reducer import PCAComponent

# Encoders
from src.models.preprocessors.encoders.one_hot_encoder import OneHotEncoderComponent
from src.models.preprocessors.encoders.ordinal_encoder import OrdinalEncoderComponent

# Feature Selectors
from src.models.preprocessors.feat_selectors.select_k_best import SelectKBestComponent
from src.models.preprocessors.feat_selectors.select_percentile import SelectPercentileComponent
from src.models.preprocessors.feat_selectors.variance_threshold import VarianceThresholdComponent

# Scalers
from src.models.preprocessors.scalers.standard_scaler import StandardScalerComponent
from src.models.preprocessors.scalers.robust_scaler import RobustScalerComponent
from src.models.preprocessors.scalers.quantile_transformer import QuantileTransformerComponent
from src.models.preprocessors.scalers.minmax_scaler import MinMaxScalerComponent
from src.models.preprocessors.scalers.normalizer import NormalizerComponent
from src.models.preprocessors.scalers.power_transformer import PowerTransformerComponent
from src.models.preprocessors.scalers.maxabs_scaler import MaxAbsScalerComponent

# Imputers - Categorical
from src.models.preprocessors.imputers.categorical.simple_imputer import CategoricalSimpleImputerComponent

# Imputers - Numerical
from src.models.preprocessors.imputers.numerical.simple_imputer import NumericalSimpleImputerComponent
from src.models.preprocessors.imputers.numerical.knn_imputer import KNNImputerComponent
from src.models.preprocessors.imputers.numerical.iterative_imputer import IterativeImputerComponent

# Classifiers
from src.models.classifiers.adaboost import AdaBoostModel
from src.models.classifiers.bagging import BaggingModel
from src.models.classifiers.decision_tree import DecisionTreeModel
from src.models.classifiers.elasticnet import ElasticNetModel
from src.models.classifiers.extra_tree import ExtraTreeModel
from src.models.classifiers.extra_trees import ExtraTreesModel
from src.models.classifiers.gaussian_nb import GaussianNBModel
from src.models.classifiers.gradient_boosting import GradientBoostingModel
from src.models.classifiers.kneighbors import KNeighborsModel
from src.models.classifiers.lasso import LassoModel
from src.models.classifiers.linear_discriminant_analysis import LinearDiscriminantAnalysisModel
from src.models.classifiers.logistic_regression import LogisticRegressionModel
from src.models.classifiers.mlp import MLPModel
from src.models.classifiers.passive_agressive import PassiveAggressiveModel
from src.models.classifiers.random_forest import RandomForestModel
from src.models.classifiers.ridge import RidgeModel
from src.models.classifiers.sgd import SGDModel
from src.models.classifiers.lgbm import LGBMModel
from src.models.classifiers._catboost import CatBoostModel
from src.models.classifiers.xgboost import XGBoostModel

# Regressors
from src.models.regressors.adaboost import AdaBoostRegressorModel
from src.models.regressors.bagging import BaggingRegressorModel
from src.models.regressors.decision_tree import DecisionTreeRegressorModel
from src.models.regressors.elasticnet import ElasticNetRegressorModel
from src.models.regressors.extra_tree import ExtraTreeRegressorModel
from src.models.regressors.extra_trees import ExtraTreesRegressorModel
from src.models.regressors.gradient_boosting import GradientBoostingRegressorModel
from src.models.regressors.fastaimlp import FastAIMLPRegressorModel
from src.models.regressors.kneighbors import KNeighborsRegressorModel
from src.models.regressors.lasso import LassoRegressorModel
from src.models.regressors.mlp import MLPRegressorModel
from src.models.regressors.random_forest import RandomForestRegressorModel
from src.models.regressors.realmlp import RealMLPRegressorModel
from src.models.regressors.ridge import RidgeRegressorModel
from src.models.regressors.sgd import SGDRegressorModel
from src.models.regressors.lgbm import LGBMRegressorModel
from src.models.regressors._catboost import CatBoostRegressorModel
from src.models.regressors.xgboost import XGBoostRegressorModel


component_map = {
    "FastICA": FastICAComponent,
    "PCA": PCAComponent,
    "OneHotEncoder": OneHotEncoderComponent,
    "OrdinalEncoder": OrdinalEncoderComponent,
    "SelectKBest": SelectKBestComponent,
    "SelectPercentile": SelectPercentileComponent,
    "VarianceThreshold": VarianceThresholdComponent,
    "StandardScaler": StandardScalerComponent,
    "RobustScaler": RobustScalerComponent,
    "QuantileTransformer": QuantileTransformerComponent,
    "MinMaxScaler": MinMaxScalerComponent,
    "Normalizer": NormalizerComponent,
    "PowerTransformer": PowerTransformerComponent,
    "MaxAbsScaler": MaxAbsScalerComponent,
    "CategoricalSimpleImputer": CategoricalSimpleImputerComponent,
    "NumericalSimpleImputer": NumericalSimpleImputerComponent,
    "KNNImputer": KNNImputerComponent,
    "IterativeImputer": IterativeImputerComponent,
}

classification_component_map = {
    "AdaBoost": AdaBoostModel,
    "Bagging": BaggingModel,
    "DecisionTree": DecisionTreeModel,
    "ElasticNet": ElasticNetModel,
    "ExtraTree": ExtraTreeModel,
    "ExtraTrees": ExtraTreesModel,
    "FastAIMLP": FastAIMLPModel,
    "GaussianNB": GaussianNBModel,
    "GradientBoosting": GradientBoostingModel,
    "KNeighbors": KNeighborsModel,
    "Lasso": LassoModel,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysisModel,
    "LogisticRegression": LogisticRegressionModel,
    "MLP": MLPModel,
    "RealMLP": RealMLPModel,
    "PassiveAggressive": PassiveAggressiveModel,
    "RandomForest": RandomForestModel,
    "Ridge": RidgeModel,
    "SGD": SGDModel,
    "LGBM": LGBMModel,
    "XGBoost": XGBoostModel,
    "CatBoost": CatBoostModel
}

regression_component_map = {
    "AdaBoost": AdaBoostRegressorModel,
    "Bagging": BaggingRegressorModel,
    "DecisionTree": DecisionTreeRegressorModel,
    "ElasticNet": ElasticNetRegressorModel,
    "ExtraTree": ExtraTreeRegressorModel,
    "ExtraTrees": ExtraTreesRegressorModel,
    "FastAIMLP": FastAIMLPRegressorModel,
    "GradientBoosting": GradientBoostingRegressorModel,
    "KNeighbors": KNeighborsRegressorModel,
    "Lasso": LassoRegressorModel,
    "MLP": MLPRegressorModel,
    "RealMLP": RealMLPRegressorModel,
    "RandomForest": RandomForestRegressorModel,
    "Ridge": RidgeRegressorModel,
    "SGD": SGDRegressorModel,
    "LGBM": LGBMRegressorModel,
    "XGBoost": XGBoostRegressorModel,
    "CatBoost": CatBoostRegressorModel
}


class PipelineComponent(BaseComponent):
    def __init__(self, problem_type="binary"):
        super().__init__("Pipeline")

        self.required_hyperparameters = [
            "model",
            "scaler",
            "dim_reducer",
            "feat_selector",
            "imputer",
            "cat_imputer",
            "encoder"
        ]

        self.problem_type = problem_type


    def construct(self, config):
        self.check(config)

        config = dict(config)
        config["_problem_type"] = self.problem_type

        # Select classification/regression model based on problem type
        if self.problem_type in ["binary", "multiclass"]:
            model = classification_component_map[config["model"]]().construct(config)
        else:
            model = regression_component_map[config["model"]]().construct(config)

        scaler = "passthrough"
        if config["scaler"] != 'None':
            scaler = component_map[config["scaler"]]().construct(config)

        dim_reducer = "passthrough"
        if config["dim_reducer"] != 'None':
            dim_reducer = component_map[config["dim_reducer"]]().construct(config)

        feat_selector = "passthrough"
        if config["feat_selector"] != 'None':
            feat_selector = component_map[config["feat_selector"]]().construct(config)

        imputer = component_map[config["imputer"]]().construct(config)
        cat_imputer = component_map[config["cat_imputer"]]().construct(config)
        encoder = component_map[config["encoder"]]().construct(config)

        categorical_transformer = sklearn.pipeline.Pipeline(steps=[
            ("Categorical Imputer", cat_imputer),
            ("Encoder", encoder)
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("Numerical Imputer", imputer, selector(dtype_exclude="category")),
                ("CategoricalPreprocessor", categorical_transformer, selector(dtype_include="category"))
            ]
        )

        return sklearn.pipeline.Pipeline(steps=[
            ("Preprocessor", preprocessor),
            ("Feature Selector", feat_selector),
            ("Scaler", scaler),
            ("Dimensionality Reducer", dim_reducer),
            ("Model", model)
        ])
