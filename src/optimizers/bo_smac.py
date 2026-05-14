import numpy as np
import time
import threadpoolctl
from smac import HyperparameterOptimizationFacade, Scenario, BlackBoxFacade
from smac.runhistory.dataclasses import TrialValue
from smac.intensifier import Intensifier
from smac.model.random_forest import RandomForest

from ConfigSpace import CategoricalHyperparameter, Constant
from ConfigSpace import ConfigurationSpace
from src.history.data_classes import Surrogate
from src.optimizers.base_optimizer import BaseOptimizer



class SMACOptimizer(BaseOptimizer):
    def __init__(
            self,
            search_space: ConfigurationSpace,
            initial_iterations: int,
            surrogate_model: str = "random_forest",
            random_forest_n_trees=None,
            output_directory="smac3_output",
            random_state=None
    ):
        super().__init__(search_space, random_state)

        def dummy_target_function(config, seed):
            return 0.0

        scenario = Scenario(
            self.search_space,
            deterministic=True,
            seed=self.random_state,
            n_workers=1,
            max_budget=1,
            output_directory=output_directory
        )

        if surrogate_model == "random_forest":
            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                self.smac = HyperparameterOptimizationFacade(
                    scenario=scenario,
                    target_function=dummy_target_function,
                    initial_design=HyperparameterOptimizationFacade.get_initial_design(
                        scenario,
                        n_configs=initial_iterations
                    ),
                    overwrite=True,
                    logging_level=40,
                    intensifier=Intensifier(
                        scenario=scenario,
                        max_config_calls=1
                    ),
                    model=RandomForest(
                        configspace=scenario.configspace,
                        n_trees=random_forest_n_trees,
                    )
                )

        elif surrogate_model == "gaussian_process":
            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                self.smac = BlackBoxFacade(
                    scenario=scenario,
                    target_function=dummy_target_function,
                    initial_design=BlackBoxFacade.get_initial_design(
                        scenario,
                        n_configs=initial_iterations
                    ),
                    overwrite=True,
                    logging_level=40,
                    intensifier=None,
                )

        else:
            raise NotImplementedError(f"Surrogate model {surrogate_model} not implemented.")

        self.surrogate = surrogate_model
        self.initial_iterations = initial_iterations

    def generate_configuration(self):
        # Start configuration generation timer
        start_time = time.perf_counter()

        info = self.smac.ask()
        assert info.seed is not None
        self._current_info = info

        return info.config, time.perf_counter() - start_time

    def tell(self, score):
        """
        Tell SMAC the score for the last asked configuration.
        Score should be the validation score (higher is better).
        SMAC expects to minimize, so we use 1 - score.
        """
        if not hasattr(self, '_current_info'):
            raise ValueError("Must call generate_configuration() before tell()")

        value = TrialValue(score)
        self.smac.tell(self._current_info, value)

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

    @staticmethod
    def _calculate_nadeau_bengio_threshold(cv_fold_scores):
        """
        Implementation of the Makarova thrshold calculation.
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

        run_history = self.smac.runhistory

        # Use get_all_trials() to retrieve a mapping of TrialInfo to TrialValue
        observed_trials = [run_history.get_trials(c)[0] for c in run_history.config_ids]

        t = len(observed_trials)

        # Early stopping logic only applies after some iterations (e.g. 20 in paper)
        # We add the number of initial configurations to this
        if t < 20 + self.initial_iterations or self.smac._model is None:
            return False

        # 1. Parse observed configurations
        X_configs = [trial.config for trial in observed_trials]

        # 2. Sample non-observed configurations used to estimate full search space potential
        X_space_configs = self.search_space.sample_configuration(size=2000)  # 2000 in paper

        X_obs_array = np.array([c.get_array() for c in X_configs])
        X_space_array = np.array([c.get_array() for c in X_space_configs])

        # 3. Predict mean and var of both candidate sets
        mean_obs, var_obs = self.smac._model.predict(X_obs_array)
        mean_space, var_space = self.smac._model.predict(X_space_array)

        # 4. Calculate std of observations and search space predictions
        std_obs = np.sqrt(var_obs).flatten()
        std_space = np.sqrt(var_space).flatten()

        mean_obs = mean_obs.flatten()
        mean_space = mean_space.flatten()

        # 5. Calculate beta for confidence bounds
        beta_t = self._compute_beta_t(delta, beta_scale_factor, t)
        sqrt_beta_t = np.sqrt(beta_t)

        # 6. Calculate bounds on spaces

        # UCB (Pessimistic on Observed) = Mean + Uncertainty
        # LCB (Optimistic on Space)     = Mean - Uncertainty

        # UCB on observations -- What is the WORST estimation of the BEST configuration (so where are we currently AT LEAST (with high confidence) in optimizing)
        ucb_obs = mean_obs + (sqrt_beta_t * std_obs)
        min_ucb_obs = np.min(ucb_obs)  # Pick the best config according to surrogate

        # LCB on space -- What is the BEST estimation of all configs in the search space (so where could we AT MOST (with high confidence) go in optimizing)
        lcb_space = mean_space - (sqrt_beta_t * std_space)
        min_lcb_space = np.min(lcb_space)  # Pick the best possible config according to surrogate

        # 8. Calculate final regret (so how much can we possibly (with high confidence) still improve by tuning)
        regret = min_ucb_obs - min_lcb_space

        # 9. Calculate current estimated noise of CV scores
        threshold_std = self._calculate_nadeau_bengio_threshold(cv_fold_scores)

        # Stop if potential gain (regret) is indistinguishable from noise (threshold)
        triggered = bool(regret <= threshold_std)

        # Return if Makarova Early Stopping was triggered
        return triggered

    def get_surrogate_predictions(self, config):
        if self.smac._model is None:
            return None

        if self.surrogate == "random_forest" and self.smac._model._rf is None:
            return None

        if self.surrogate == "gaussian_process" and not self.smac._model._is_trained:
            return None

        config_array = config.get_array()
        surrogate_mean, surrogate_var = self.smac._model.predict(
            np.expand_dims(config_array, axis=0)
        )

        acquisition_value = 0.0
        try:
            if hasattr(self.smac, 'acquisition') and hasattr(self.smac.acquisition, 'function'):
                acquisition_value = self.smac.acquisition.function(np.expand_dims(config_array, axis=0))
        except:
            pass

        return Surrogate(
            mean=float(surrogate_mean[0][0]),
            std=float(np.sqrt(surrogate_var[0][0])),
            acquisition=float(acquisition_value) if isinstance(acquisition_value, (float, np.float64)) else 0.0
        )
