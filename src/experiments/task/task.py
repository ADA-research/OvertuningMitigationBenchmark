import time
import numpy as np

from src.experiments.task.task_config import TaskConfig
from src.datasets.tabarena_dataloader import TabarenaDataLoader
from src.datasets.offline_dataloader import OfflineDataLoader
from src.datasets.toy_dataloader import ToyDataLoader
from src.mitigations.racing.bergman_racing import EarlyStoppingCVBergman, RobustBergman
from src.mitigations.racing.base_racing import BaseRacing
from src.optimizers.bo_hebo import HEBOOptimizer
from src.optimizers.mlplan_phase_two import MLPlanPhaseTwoOptimizer
from src.resamplers.online_resamplers.cv_selection_set_resampler import CVSelectionSetResampler
from src.resamplers.online_resamplers.holdout_resampler import HoldoutResampler
from src.resamplers.online_resamplers.holdout_selection_set_resampler import HoldoutSelectionSetResampler
from src.resamplers.online_resamplers.cv_resampler import CVResampler
from src.optimizers.random_search import RandomSearch
from src.optimizers.bo_smac import SMACOptimizer
from src.search_space.search_space import SearchSpace
from src.target.runner import Runner
from src.experiments.task.task_data_splitter import TaskDataSplitter
from src.metrics.metric import Metric
from src.mitigations.thresholdout import Thresholdout


def run_task(task_config: TaskConfig):
    # Collect dataset
    if task_config.offline_data_loading:
        dataloader = OfflineDataLoader()
    else:
        dataloader = TabarenaDataLoader()

    # # Load toy dataset for testing purposes if dataset is a string
    # if task_config.dataset_id == "toy":
    #     dataloader = ToyDataLoader()
    #     X, y, label_map = dataloader.load(
    #         task_config.problem_type,
    #         task_config.outer_evaluation.train_size if isinstance(task_config.outer_evaluation.train_size,
    #                                                               int) else 2000
    #     )

    X, y, label_map = dataloader.load(task_config.dataset_id, problem_type=task_config.problem_type)

    # Store mapping of labels
    task_config.label_map = label_map

    # Make outer data splitter object
    outer_data_splitter = TaskDataSplitter(
        task_config.outer_evaluation,
        task_config.random_state
    )

    X_train, X_test, y_train, y_test = outer_data_splitter.make_outer_split(X, y)

    if task_config.debug:
        X_train = X_train[:500]
        y_train = y_train[:500]

    # If we use a selection set
    if task_config.evaluation.selection_size is not None and task_config.evaluation.selection_size > 0.0:

        if task_config.evaluation.resampling == "holdout":
            resampler = HoldoutSelectionSetResampler
        elif task_config.evaluation.resampling == "cv":
            resampler = CVSelectionSetResampler
        else:
            raise NotImplementedError("Resampling Strategy not Implemented")

    else:
        if task_config.evaluation.resampling == "holdout":
            resampler = HoldoutResampler
        elif task_config.evaluation.resampling == "cv":
            resampler = CVResampler
        else:
            raise NotImplementedError("Resampling Strategy not fully Implemented")

    # Prepare resampler arguments
    resampler_args = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "problem_type": task_config.problem_type,
        "reshuffle": task_config.evaluation.reshuffle,
        "n_repeats": task_config.evaluation.n_repeats,
        "seed": task_config.random_state,
    }

    # Add selection set argument if relevant
    if task_config.evaluation.selection_size is not None and task_config.evaluation.selection_size > 0.0:
        resampler_args["selection_fraction"] = task_config.evaluation.selection_size

    # Add holdout fraction argument if relevant
    if task_config.evaluation.resampling == "holdout":
        resampler_args["holdout_fraction"] = task_config.evaluation.val_size

    if task_config.evaluation.resampling == "cv":
        resampler_args["n_folds"] = task_config.evaluation.n_folds

    if task_config.mitigation_strategy == "mlplan":
        # To run ML-Plan, we need two runners with different resamplers
        # First resampler, we need to exclude 30% of 'select' data
        # which is identical to the HoldoutSelectionSetResampler
        # Note that we do store selection scores for simplicity (but these should not be used)

        # Second resampler is simply a HoldoutResampler

        # Overwrite resamplers: We do holdout selection set to keep the select data
        resampler = HoldoutSelectionSetResampler

        # Overwrite evaluation task config to 5-MCCV
        resampler_args["n_repeats"] = 5
        resampler_args["holdout_fraction"] = 0.3
        resampler_args["selection_fraction"] = 0.3
        resampler_args["reshuffle"] = False

        # How many iterations should we spend on phase 1?
        # If we use k=25 in the paper, we would use 100 iterations (25*2*10MCCV / 5MCCV) on phase 2
        # Note that if there are no 25 random configs within the epsilon distance from the best, we spend less budget
        # For now, we spend 60% of the budget on phase 1, which works for 5/10 MCCV in the phases
        task_config.iterations = int(task_config.iterations * 0.6)

    # Initialize resampler
    resampler = resampler(
        **resampler_args
    )

    search_space = SearchSpace(task_config.to_dict()).get_space()

    if task_config.optimizer == "random_search":
        optimizer = RandomSearch(
            search_space=search_space,
            random_state=task_config.random_state
        )

    elif task_config.optimizer == "smac":
        optimizer = SMACOptimizer(
            search_space=search_space,
            initial_iterations=task_config.bo_initial_random_iterations,
            surrogate_model=task_config.smac_surrogate_model,
            random_forest_n_trees=task_config.smac_surrogate_random_forest_n_trees,
            random_state=task_config.random_state,
            output_directory=f"scratch-local/sschroder/OvertuningBenchmark/smac3_output/{task_config.result_path}"
        )
    elif task_config.optimizer == "hebo":
        optimizer = HEBOOptimizer(
            search_space=search_space,
            initial_iterations=task_config.bo_initial_random_iterations,
            random_state=task_config.random_state
        )

    else:
        raise ValueError("Unknown optimizer")

    # Mitigator initialization
    # Question: Is one mitigator at a time enough?
    mitigator = None

    if task_config.mitigation_strategy == "thresholdout":
        mean_label_to_scale = None
        if task_config.problem_type == "regression":
            mean_label_to_scale = np.mean(y_train) # Take mean train label to scale noise in thresholdout
        
        # Question: What is the number of holdout samples in case of CV? Should we even allow CV?
        mitigator = Thresholdout(
            num_holdout_samples=int(len(X_train) * task_config.evaluation.val_size),
            mean_label_to_scale=mean_label_to_scale
        )
        
    elif task_config.mitigation_strategy == "mlplan":
        mitigator = "mlplan"

    if task_config.racing_strategy == "bergman_forgiving":
        racing_strategy = EarlyStoppingCVBergman(aggressive=False)

    elif task_config.racing_strategy == "bergman_aggressive":
        racing_strategy = EarlyStoppingCVBergman(aggressive=True)

    elif task_config.racing_strategy == "bergman_robust":
        racing_strategy = RobustBergman()

    # Initialize robust bergman based on number after _ in racing strategy
    elif task_config.racing_strategy.startswith("bergman_robust_"):
        racing_strategy = RobustBergman(n_configs=int(task_config.racing_strategy.split("_")[-1]))

    else:
        racing_strategy = BaseRacing()

    runner = Runner(
        iterations=task_config.iterations,
        resampler=resampler,
        optimizer=optimizer,
        metric=Metric(
            task_config.metric,
            problem_type=task_config.problem_type
        ),
        problem_type=task_config.problem_type,
        folds_per_repeat=task_config.evaluation.n_folds,
        repeats=task_config.evaluation.n_repeats,
        retrain=task_config.evaluation.retrain,
        random_state=task_config.random_state,
        mitigator=mitigator,
        racing_strategy=racing_strategy,
        display_name=task_config.result_path,
        model_threads=task_config.model_threads,
    )

    # Execute HPO

    runner.optimize()

    # In case of ML-Plan, we have to do a second runner
    if task_config.mitigation_strategy == "mlplan":
        resampler = HoldoutResampler

        # Overwrite evaluation task config to 10-MCCV
        resampler_args["n_repeats"] = 10
        resampler_args["holdout_fraction"] = 0.3

        del resampler_args["selection_fraction"]

        # Initialize resampler
        resampler = resampler(
            **resampler_args
        )

        optimizer = MLPlanPhaseTwoOptimizer(
            runner.history,
            random_state=task_config.random_state
        )

        runner_phase_two = Runner(
            iterations=optimizer.number_of_selected_configurations(),
            optimizer=optimizer,
            resampler=resampler,
            metric=Metric(
                task_config.metric,
                task_config.problem_type
            ),
            problem_type=task_config.problem_type,
            retrain=False,  # We already retrained in phase one
            display_name=task_config.result_path
        )

        runner_phase_two.optimize()

        # Combine the two phases in the history for phase one
        # This works with the setup where only runner.history is returned
        runner.history.add_mlplan_phase_two_history(
            runner_phase_two.history,
            optimizer.iteration_map_phase_two_to_phase_one
        )

    return runner.history
