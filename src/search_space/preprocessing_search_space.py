import ConfigSpace

from src.search_space.preprocessors.dim_reducers.fastica_space import FastICASpace
from src.search_space.preprocessors.dim_reducers.pca_space import PCASpace
from src.search_space.preprocessors.feat_selectors.select_kbest_space import SelectKBestSpace
from src.search_space.preprocessors.feat_selectors.select_percentile_space import \
    SelectPercentileSpace
from src.search_space.preprocessors.feat_selectors.variance_threshold_space import \
    VarianceThresholdSpace
from src.search_space.preprocessors.imputers.categorical.simple_imputer_space import \
    CategoricalSimpleImputerSpace
from src.search_space.preprocessors.imputers.numerical.iterative_imputer_space import \
    IterativeImputerSpace
from src.search_space.preprocessors.imputers.numerical.knn_imputer_space import KNNImputerSpace
from src.search_space.preprocessors.imputers.numerical.simple_imputer_space import \
    NumericalSimpleImputerSpace
from src.search_space.preprocessors.scalers.quantile_transformer_space import \
    QuantileTransformerSpace
from src.search_space.preprocessors.scalers.robust_scaler_space import RobustScalerSpace
from src.search_space.preprocessors.scalers.standard_scaler_space import StandardScalerSpace
from src.search_space.preprocessors.scalers.normalizer_space import NormalizerSpace

namespace = {
    'FastICA': FastICASpace,
    'PCA': PCASpace,
    'SelectKBest': SelectKBestSpace,
    'SelectPercentile': SelectPercentileSpace,
    'VarianceThreshold': VarianceThresholdSpace,
    'NumericalSimpleImputer': NumericalSimpleImputerSpace,
    'IterativeImputer': IterativeImputerSpace,
    'KNNImputer': KNNImputerSpace,
    'QuantileTransformer': QuantileTransformerSpace,
    'RobustScaler': RobustScalerSpace,
    'StandardScaler': StandardScalerSpace,
    'Normalizer': NormalizerSpace
}


class PreprocessingSearchSpace:
    def __init__(self, config=None):
        self.config = config

        self.space = ConfigSpace.ConfigurationSpace(name="PreprocessingSearchSpace", seed=config.random_state)

        self.scaler_hp = ConfigSpace.CategoricalHyperparameter(name="scaler",
                                                               choices=config.search_space.preprocessors.scalers)

        self.dim_reducer_hp = ConfigSpace.CategoricalHyperparameter(name="dim_reducer",
                                                                    choices=config.search_space.preprocessors.dim_reducers)
        self.feat_selector_hp = ConfigSpace.CategoricalHyperparameter(name="feat_selector",
                                                                      choices=config.search_space.preprocessors.feat_selectors)
        self.imputer_hp = ConfigSpace.CategoricalHyperparameter(name="imputer",
                                                                choices=config.search_space.preprocessors.imputers)
        self.cat_imputer_hp = ConfigSpace.CategoricalHyperparameter(name="cat_imputer",
                                                                    choices=["CategoricalSimpleImputer"])
        self.encoder_hp = ConfigSpace.CategoricalHyperparameter(name="encoder",
                                                                choices=config.search_space.preprocessors.encoders)

        self.space.add([
            self.scaler_hp,
            self.dim_reducer_hp,
            self.feat_selector_hp,
            self.imputer_hp,
            self.cat_imputer_hp,
            self.encoder_hp
        ])

        for scaler in self.config.search_space.preprocessors.scalers:
            if scaler != 'None' and scaler in namespace:
                scaler_space = namespace[scaler](self.scaler_hp, seed=config.random_state)
                self.space.add(scaler_space.get_hyperparameters())
                self.space.add(scaler_space.space.conditions)

        for dim_reducer in self.config.search_space.preprocessors.dim_reducers:
            if dim_reducer != 'None':
                dim_reducer_space = namespace[dim_reducer](self.dim_reducer_hp, seed=config.random_state)
                self.space.add(dim_reducer_space.get_hyperparameters())
                self.space.add(dim_reducer_space.space.conditions)

        for feat_selector in self.config.search_space.preprocessors.feat_selectors:
            if feat_selector != 'None':
                feat_selector_space = namespace[feat_selector](self.feat_selector_hp,
                                                               problem_type=config.problem_type,
                                                               seed=config.random_state)

                self.space.add(feat_selector_space.get_hyperparameters())
                self.space.add(feat_selector_space.space.conditions)

        for imputer in self.config.search_space.preprocessors.imputers:
            imputer_space = namespace[imputer](self.imputer_hp, seed=config.random_state)

            self.space.add(imputer_space.get_hyperparameters())
            self.space.add(imputer_space.space.conditions)

        # Add the search space of the only possible categorical imputer
        cat_imputer_space = CategoricalSimpleImputerSpace(self.cat_imputer_hp, seed=config.random_state)

        self.space.add(cat_imputer_space.get_hyperparameters())
        self.space.add(cat_imputer_space.space.get_conditions())
