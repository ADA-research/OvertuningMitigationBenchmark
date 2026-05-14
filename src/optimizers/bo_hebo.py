import numpy as np
import pandas as pd
import time
import torch
import random
from scipy.stats import qmc
from sklearn.utils import check_random_state

from ConfigSpace import (
    CategoricalHyperparameter,
    Constant,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,

)

from ConfigSpace import ConfigurationSpace, Configuration
from src.history.data_classes import Surrogate
from src.optimizers.base_optimizer import BaseOptimizer

from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace
from hebo.acquisitions.acq import Mean, Sigma
from hebo.models.model_factory import get_model


class HEBOOptimizer(BaseOptimizer):
    def __init__(
            self,
            search_space: ConfigurationSpace,
            initial_iterations: int,
            random_state=None
    ):
        super().__init__(search_space, random_state)

        self.initial_iterations = initial_iterations
        self.search_space = search_space
        self.design_space = self._configspace_to_design_space(search_space=search_space)

        # HEBO is not reproducible by using the scramble_seed argument alone
        # Seeding below was established through experimentation
        # Several tests are dedicated to checking HEBO reproducibility
        self.seed = random_state

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Monkey patch default_rng to always be seeded
        _original_default_rng = np.random.default_rng
        def _seeded_default_rng(*args, **kwargs):
            return _original_default_rng(self.seed)
        np.random.default_rng = _seeded_default_rng

        torch.manual_seed(self.seed)  # GPs
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.hebo = HEBO(
            self.design_space,
            rand_sample=initial_iterations,
            scramble_seed=self.seed
        )

        self.surrogate = None

        self._last_df = None

    def generate_configuration(self):
        # Start configuration generation timer
        start_time = time.perf_counter()

        # With few observations, hebo suggest can fail due to Cholesky error
        # In those cases, we catch the error and quasi-sample a random config instead
        # TODO: We should probably log how often this happens
        try:
            # Get HEBO suggested configuration
            df_config = self.hebo.suggest(n_suggestions=1)

        except Exception as e:
            # If HEBO suggest fails, sample random configuration
            df_config = self.hebo.quasi_sample(1)

            # Store suggested config for tell
        self._last_df = df_config

        print(f"Suggest: {time.perf_counter() - start_time}")
        # Return Configuration object so backend can work fully with ConfigSpace
        return self._hebo_df_to_configuration(df_config), time.perf_counter() - start_time

    def tell(self, score):
        y = np.array([[score]], dtype=float)

        self.hebo.observe(self._last_df, y)

        return None

    @staticmethod
    def _configspace_to_design_space(search_space: ConfigurationSpace) -> DesignSpace:
        """Helper function to convert a ConfigSpace to a HEBO DesignSpace.
        """
        # Define list of hebo hyperparameters
        hebo_params = []

        # List configspace hyperparameters
        for hp_name in list(search_space.get_active_hyperparameters(search_space.get_default_configuration())):
            # Get hyperparameter
            hp = search_space[hp_name]

            # Categorical conversion
            if isinstance(hp, CategoricalHyperparameter):

                # Skip categorical hps with one option
                if hp.size == 1:
                    continue

                hebo_params.append(
                    {
                        'name': hp_name,
                        'type': 'cat',
                        'categories': list(hp.choices)
                    }
                )

            # Conversion of Floats
            elif isinstance(hp, UniformFloatHyperparameter):
                # Log values have a different type in HEBO, log space has type 'pow'
                if hp.log:
                    hebo_params.append(
                        {
                            'name': hp_name,
                            'type': 'pow',
                            'lb': hp.lower,
                            'ub': hp.upper,
                            'base': 10
                        }
                    )

                # Regular float hyperparameter has type 'num'
                else:
                    hebo_params.append(
                        {
                            'name': hp_name,
                            'type': 'num',
                            'lb': hp.lower,
                            'ub': hp.upper
                        }
                    )

            # Convert Integer hyperparameters
            # TODO: Possibly handle log space here as well, although LGBM and RealMLP do not have integer hps in log space
            elif isinstance(hp, UniformIntegerHyperparameter):
                hebo_params.append(
                    {
                        'name': hp_name,
                        'type': 'int',
                        'lb': hp.lower,
                        'ub': hp.upper
                    }
                )

        # Return HEBO design space
        return DesignSpace().parse(hebo_params)

    def _hebo_df_to_configuration(self, df: pd.DataFrame) -> Configuration:
        df = self._normalize_nullable_categorical_nans(df)

        # Parse first row
        config_dict = df.iloc[0].to_dict()

        # Sample default config to get non-tunable hyperparameters from search space
        config = self.search_space.get_default_configuration()

        for key in config_dict:
            config[key] = config_dict[key]

        # Return configuration
        return config

    def _normalize_nullable_categorical_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace NaN with None for categorical dims where None is a valid category.

        HEBO can emit NaN for None-valued categorical choices. Its transform step
        expects the original category key and crashes on NaN.
        """
        if df is None or df.empty:
            return df

        normalized = df.copy()
        for name, param in self.design_space.paras.items():
            if name not in normalized.columns:
                continue

            categories = getattr(param, "categories", None)
            if categories is None:
                continue

            if any(category is None for category in categories):
                # Arrow/String dtypes can coerce None back to NA unless cast to object first.
                normalized[name] = normalized[name].astype(object)
                normalized[name] = normalized[name].where(normalized[name].notna(), None)

        return normalized

    def count_number_of_hyperparameters(self) -> int:
        """Counts the number of actual, tunable hyperparameters in the search space."""
        n_hps = 0

        for hp in self.search_space.values():
            # Add categorical hyperparameter if there is more than one choice
            if isinstance(hp, CategoricalHyperparameter) and hp.size == 1:
                continue
            # Skip constants
            elif isinstance(hp, Constant):
                continue
            else:
                n_hps += 1

        return n_hps

    def get_hebo_surrogate_model(self):
        model = get_model(
            self.hebo.model_name,
            self.design_space.num_numeric,
            self.design_space.num_categorical,
            1,
            **self.hebo.model_config,
        )

        return model

    @staticmethod
    def _calculate_nadeau_bengio_threshold(cv_fold_scores):
        """
        Implementation of the Makarova threshold calculation.
        The Decay in this function is the correction by Nadeau & Bengio (2003).
        """
        if cv_fold_scores is None or len(cv_fold_scores) < 2:
            return 0.0

        cv_values = np.array(cv_fold_scores)
        n_folds = len(cv_values)

        # 1. Calculate Standard Deviation (Population based, as per snippet usage of np.var)
        sd_cv_values = np.sqrt(np.var(cv_values))

        # 2. Calculate Decay (Nadeau & Bengio 2003 correction)
        decay = np.sqrt((1 / n_folds + 1 / (n_folds - 1)))

        # 3. Corrected Threshold
        return sd_cv_values * decay

    def _compute_beta_t(self, delta, beta_scale_factor, t):
        """Calculates scaling factor Beta_t based on Theorem 1 + Scaling Factor."""
        if t == 0:
            return 1.0

        numerator = self.count_number_of_hyperparameters() * (t ** 2) * (np.pi ** 2)
        denominator = 6 * delta
        beta = 2 * np.log(numerator / denominator)
        return beta * beta_scale_factor

    def early_stopping_makarova_triggered(self, cv_fold_scores):
        """
        Computes Makarova Early Stopping. Does not early stop optimization but returns True if optimization was early
        stopped, according to the method.

        # (Source 1) For original implementation:
        # https://github.com/amazon-science/bo-early-stopping/blob/cffd7d367b5a3fc2abd1ba045300bb5aae29459b/src/enhanced_gp.py#L122

        # (Source 2) For HEBO implementation in overtuning paper:
        # https://github.com/slds-lmu/paper_2025_overtuning/blob/c4a84cd48eec24d544f3559b546797624fd934ee/overtunebench/overtunebench/samplers/hebo_makarova.py

        According to both sources:
        delta = 0.1
        beta_scale_factor = 0.2
        """
        delta = 0.1
        beta_scale_factor = 0.2

        t = self.hebo.X.shape[0]

        # Early stopping logic only applies after some iterations (e.g. 20 in paper)
        if t < 20 + self.initial_iterations:
            return False

        # 1. Get observed configurations from history
        # Concatenate all observed DataFrames
        observed_df = self._normalize_nullable_categorical_nans(self.hebo.X)
        X, Xe = self.design_space.transform(observed_df)

        # 2. Sample non-observed configurations used to estimate full search space potential
        X_space_df = self.design_space.sample(2000)  # 2000 in paper
        X_space_df = self._normalize_nullable_categorical_nans(X_space_df)
        X_space, Xe_space = self.design_space.transform(X_space_df)

        try:
            # 3. Predict on observed and space samples
            model = self.get_hebo_surrogate_model()
            start_time = time.perf_counter()
            model.fit(X, Xe, torch.FloatTensor(self.hebo.y))
            print(f"Makarova fit: {time.perf_counter() - start_time}")

            mu = Mean(model)
            sig = Sigma(model)  # internally calculated as -Sigma

            mean_obs, std_obs = mu(X, Xe), -sig(X, Xe)
            mean_space, std_space = mu(X_space, Xe_space), -sig(X_space, Xe_space)

            # mean_obs = mean_obs.flatten()
            # mean_space = mean_space.flatten()

            # 5. Calculate beta for confidence bounds
            beta_t = self._compute_beta_t(delta, beta_scale_factor, t)
            sqrt_beta_t = np.sqrt(beta_t)

            # 6. Calculate bounds on spaces

            # UCB (Pessimistic on Observed) = Mean + Uncertainty
            # LCB (Optimistic on Space)     = Mean - Uncertainty

            # UCB on observations -- What is the WORST estimation of the BEST configuration (so where are we currently AT LEAST (with high confidence) in optimizing)
            ucb_obs = mean_obs + (sqrt_beta_t * std_obs)
            min_ucb_obs = ucb_obs.min()  # Pick the best config according to surrogate

            # LCB on space -- What is the BEST estimation of all configs in the search space (so where could we AT MOST (with high confidence) go in optimizing)
            lcb_space = mean_space - (sqrt_beta_t * std_space)
            min_lcb_space = lcb_space.min()  # Pick the best possible config according to surrogate

            # 8. Calculate final regret (so how much can we possibly (with high confidence) still improve by tuning)
            regret = min_ucb_obs - min_lcb_space

            # 9. Calculate current estimated noise of CV scores
            threshold_std = self._calculate_nadeau_bengio_threshold(cv_fold_scores)

            # Stop if potential gain (regret) is indistinguishable from noise (threshold)
            triggered = bool(regret <= threshold_std)

            # Return if Makarova Early Stopping was triggered
            return triggered
            
        except Exception as e:
            print(f"Error in Makarova early stopping calculation: {e}")
            return False

    def get_surrogate_predictions(self, config):
        # Below a certain number of iterations, HEBO fails to make surrogate predictions
        # To align with SMAC surrogates, set to initial configurations
        if self.hebo.X.shape[0] < self.initial_iterations:
            return None

        # Get and fit HEBO surrogate model on observations
        start_time = time.perf_counter()
        model = self.get_hebo_surrogate_model()
        observed_df = self._normalize_nullable_categorical_nans(self.hebo.X)
        X, Xe = self.design_space.transform(observed_df)

        try:
            # Get HEBO suggested configuration
            model.fit(X, Xe, torch.FloatTensor(self.hebo.y))
            print(f"Surrogate: {time.perf_counter() - start_time}")

        except Exception as e:
            # If HEBO suggest fails, sample random configuration
            print(f"HEBO encountered Error in getting surrogate predictions: {e}")

            return Surrogate(
                mean=None,
                std=None,
                acquisition=None,
            )

        # Predict mean and variance for config
        mu = Mean(model)
        sig = Sigma(model)  # internally calculated as -Sigma

        # We always tell first, and call this method second, so self._last_df should be the current config
        last_df = self._normalize_nullable_categorical_nans(self._last_df)
        X_config, Xe_config = self.design_space.transform(last_df)
        mean_config, std_config = mu(X_config, Xe_config), -sig(X_config, Xe_config)
        print()
        
        return Surrogate(
            mean=float(mean_config),
            std=float(std_config),
            acquisition=0.0
        )
