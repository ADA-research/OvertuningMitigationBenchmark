from typing import List
import gc
import torch
import ctypes
import time
from sklearn.utils import shuffle
from tqdm import tqdm

from src.models.pipeline import PipelineComponent
from src.history.run_history import RunHistory, MLPlanRunHistory
from src.optimizers.bo_hebo import HEBOOptimizer
from src.resamplers.base_resampler import BaseResampler
from src.resamplers.online_resamplers.cv_resampler import CVResampler
from src.metrics.metric import BaseMetric
from src.optimizers.base_optimizer import BaseOptimizer
from src.optimizers.bo_smac import SMACOptimizer
from src.mitigations.racing.base_racing import BaseRacing
from src.history.data_classes import Run
from src.evaluators.base_evaluator import Evaluator
from src.mitigations.thresholdout import Thresholdout



class Runner:
    def __init__(
            self,
            iterations: int = 250,
            resampler: BaseResampler = None,
            optimizer: BaseOptimizer = None,
            metric: BaseMetric = None,
            problem_type: str = "classification",
            mitigator=None,
            retrain: bool = True,
            random_state: int = None,
            folds_per_repeat: int = 1,  # Total Number of Folds
            repeats: int = 1,
            # Total Number of repeats (Total Number of Folds = Total number of repeats x Folds per repeat)
            racing_strategy: BaseRacing = BaseRacing(),
            offline_data: bool = False,
            shuffled: bool = False,  # Shuffle Fold Order after each race iteration (Keeping repeats together)
            fully_evaluate: bool = True,  # Does the Incumbent have to be fully evaluated at every time step
            require_full_block: bool = True,
            # Require a candidate to be evaluated on the same number of folds as the incumbent to replace it
            display_name="", # Name to display during optimization
            model_threads: int = 1,
    ):

        self.iterations = iterations

        self.resampler = resampler
        self.optimizer = optimizer
        self.metric = metric
        self.retrain = retrain
        self.mitigator = mitigator
        self.random_state = random_state
        self.racing_strategy = racing_strategy
        self.offline = offline_data

        self.folds_per_repeat = folds_per_repeat
        self.repeats = repeats
        self.total_folds = folds_per_repeat * repeats

        self.shuffled = shuffled
        self.fully_evaluate = fully_evaluate
        self.require_full_block = require_full_block
        self.display_name = display_name
        self.model_threads = max(1, int(model_threads))

        if not self.offline:
            self.evaluator = Evaluator(metric)
        else:
            self.evaluator = OfflineEvaluator()

        self.history = RunHistory()

        if mitigator == "mlplan":
            self.history = MLPlanRunHistory()

        self.pipeline = PipelineComponent(problem_type=problem_type)

    def optimize(self) -> None:
        # If we are dealing with set based methods we use set_based_optimize() all methods except for GreedyKFold utilize a synchronous evaluation of a candidate set
        if self.racing_strategy.required_configs > 1:
            if isinstance(self.racing_strategy, GreedyKFold) or isinstance(self.racing_strategy, GreedyKFoldB):
                self.greedy_k_fold()
            else:
                self.set_based_optimize()
            return

        with tqdm(total=self.iterations, desc=f"Optimizing {self.display_name}") as pbar:

            for i in range(self.iterations):
                # Sample configuration
                config, config_suggestion_time = self.optimizer.generate_configuration()

                model_config = dict(config)
                model_config["_model_threads"] = self.model_threads
                model_config["_problem_type"] = self.pipeline.problem_type
                model_config["_metric_name"] = self.metric.name

                estimator = self.pipeline.construct(model_config)

                run = Run(
                    config=config,
                    optimizer_suggest_time=config_suggestion_time,
                    iteration=i,
                    total_folds=self.total_folds
                )

                for (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_sel, y_sel) in self.resampler:
                    if 1 <= len(run.folds) and self.racing_strategy.should_stop(run, self.history, self.require_full_block):
                        break

                    fold = self.evaluator.evaluate(
                        estimator,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        X_test,
                        y_test,
                        X_sel,
                        y_sel
                    )

                    run.add_fold(fold)

                if self.retrain:
                    retrain_fold = self.evaluator.retrain_and_evaluate(
                        estimator,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        X_test,
                        y_test,
                        X_sel,
                        y_sel
                    )

                    run.add_retrain_evaluation(retrain_fold)

                # Return validation score to optimizer
                if isinstance(self.optimizer, (SMACOptimizer, HEBOOptimizer)):
                    score_to_tell_optimizer = run.average_val_score()

                    if isinstance(self.mitigator, Thresholdout):
                        score_to_tell_optimizer = self.mitigator.query(
                            run.average_train_score(), run.average_val_score()
                        )

                    self.optimizer.tell(score_to_tell_optimizer)
                    run.surrogate = self.optimizer.get_surrogate_predictions(config)

                self.history.add_run(run)

                # Calculate if optimization should be stopped according to Makarova early stopping
                # Note we do not actually stop the process
                if isinstance(self.optimizer, (SMACOptimizer, HEBOOptimizer)) and isinstance(self.resampler, CVResampler):

                    incumbent_fold_scores = [f.scores.val for f in self.history.incumbent.folds]

                    early_stopped_triggered = self.optimizer.early_stopping_makarova_triggered(incumbent_fold_scores)
                    self.history.history[-1].early_stopped_makarova = early_stopped_triggered

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Val": self.history.incumbent.average_val_score(),
                        "Test": self.history.incumbent.average_test_score(),
                        "Meta-overfitting": self.history.meta_overfitting(),
                        "Overtuning": self.history.overtuning(),
                        "Best Model": self.history.incumbent.config["model"]
                    }
                )
