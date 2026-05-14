import random
import copy
from src.experiments.experiment import Experiment
from src.experiments.task.task_config import TaskConfig


def benchmark_one_dataset_lite(
    dataset_id,
    model_string="LGBM",
    large_dataset=False,
    problem_type="binary",
    output_dir=".",
    model_threads=1,
):
    """Function to create a benchmark experiment based on TabArena Lite in the overtuning benchmark project
    Creates all HPO tasks in one combination of dataset and model

    In total, for big datasets, we create 78 tasks, and for small datasets, we create 260 tasks (10 repetitions)

    More information in benchmark.md in repository root.
    """

    tasks = []
    
    task_config = TaskConfig()
    task_config.dataset_id = dataset_id

    # Constants not altered in different experiments
    task_config.problem_type = problem_type # binary, multiclass, or regression. Needs to be defined since regression/multiclass not always clear
    task_config.random_state = 78 # Global seed
    task_config.offline = False # Offline if we want to use existing model runs, but support limited for now
    task_config.racing_strategy = "none" # Racing strategy. For this benchmark, we use bergman_aggressive
    task_config.mitigation_strategy = "none" # Mitigation strategy if any. We use thresholdout and mlplan
    task_config.offline_data_loading = False # Load data from disk (avoids openml errors/rate limits, but data needs to be in src/datasets/datasets folder)

    task_config.iterations = 250 # HPO iterations
    task_config.bo_initial_random_iterations = 25  # Random iterations
    task_config.smac_surrogate_model = "gaussian_process"  # We only use GP in SMAC
    task_config.smac_surrogate_random_forest_n_trees = 10 # Unused
    task_config.debug = False # If true, some debug options are activated
    task_config.store_results_google_cloud = False # Store CSV results on GCS
    task_config.store_vectors_google_cloud = False # Store labels/preds on GCS (but large!!)
    task_config.model_threads = max(1, int(model_threads))

    # Search space initialization
    task_config.search_space.classifiers = [model_string]
    task_config.search_space.regressors = [model_string]
    task_config.search_space.scalers = ["StandardScaler"]
    task_config.search_space.encoders = ["OneHotEncoder"]
    task_config.search_space.dim_reducers = ["None"]
    task_config.search_space.feat_selectors = ["None"]
    task_config.search_space.imputers = ["NumericalSimpleImputer"]

    # We always retrain
    task_config.evaluation.retrain = True

    ####################
    # OUTER DATA SPLIT #
    ####################

    # We follow TabArena in the number of repeats
    # TabArena considers a dataset to be large if > 2,500 samples
    outer_n_repeats = 10 if not large_dataset else 3

    if problem_type == "binary":
        metrics = ["roc_auc"]
        
    elif problem_type == "multiclass":
        metrics = ["neg_log_loss"]

    elif problem_type == "regression":
        metrics = ["neg_root_mean_squared_error"] 

    # We do 10 repeats
    for repeat_idx in range(outer_n_repeats):
        # With 3CV
        task_config.outer_evaluation.fold = 0  # TabArena Lite: We now investigate one outer fold per repetition
        task_config.outer_evaluation.repeat = repeat_idx  # Determines the current repeat
        task_config.outer_evaluation.n_repeats = outer_n_repeats  # This tells the data splitter how many splits to make so the two above make sense
        task_config.outer_evaluation.resampling = "cv"  # This tells the data splitter how many splits to make so the two above make sense
        task_config.outer_evaluation.n_folds = 3  # This tells the data splitter how many splits to make so the two above make sense
        task_config.outer_evaluation.train_size = 0  # Only used if resampling is not CV

        # Now we differentiate in optimizers
        for optimizer in ["hebo", "smac"]:
            task_config.optimizer = optimizer

            for metric in metrics: # One metric is investigated
                task_config.metric = metric

                for mlplan in [True, False]:

                    if not mlplan:

                        for reshuffling in [True, False]:
                            task_config.evaluation.reshuffle = reshuffling

                            for racing in [True, False]:
                                task_config.racing_strategy = "bergman_aggressive" if racing else "none"

                                if racing:
                                    # For racing, we do 5x5CV with Aggressive Bergman
                                    task_config.evaluation.n_repeats = 5
                                    task_config.evaluation.resampling = "cv"
                                    task_config.evaluation.n_folds = 5
                                    task_config.mitigation_strategy = "bergman_aggressive"
                                    task_config.evaluation.selection_size = None

                                    # RACING ENDPOINT
                                    tasks.append(copy.deepcopy(task_config))

                                else:
                                    task_config.racing_strategy = "none"

                                    for thresholdout in [True, False]:

                                        if thresholdout:
                                            task_config.mitigation_strategy = "thresholdout"
                                            task_config.evaluation.val_size = 0.2
                                            task_config.evaluation.n_repeats = 1
                                            task_config.evaluation.resampling = "holdout"
                                            task_config.evaluation.selection_size = None

                                            # THRESHOLDOUT ENDPOINT
                                            tasks.append(copy.deepcopy(task_config))

                                        else:
                                            # No specific mitigation
                                            task_config.mitigation_strategy = "none"
                                            task_config.racing_strategy = "none"

                                            # With two inner resampling methods
                                            for inner_resampling in ["cv", "holdout"]:
                                                task_config.evaluation.resampling = inner_resampling
                                                task_config.evaluation.n_repeats = 1

                                                # Will be used if inner resampling is CV
                                                task_config.evaluation.n_folds = 5

                                                # Will be used if inner resampling is holdout
                                                task_config.evaluation.val_size = 0.2

                                                for selection_set in [0.0, 1/6]:
                                                    task_config.evaluation.selection_size = selection_set

                                                    # ALL OTHER ENDPOINTS
                                                    tasks.append(copy.deepcopy(task_config))

                    else:
                        # MLPlan (if mlplan, most other settings are ignored, but resetting for clarity)
                        task_config.mitigation_strategy = "mlplan"
                        task_config.racing_strategy = "none"
                        task_config.evaluation.selection_size = None
                        task_config.evaluation.reshuffle = False

                        # ENDPOINT
                        tasks.append(copy.deepcopy(task_config))

    for task in tasks:
        # Add task result path
        strategy_suffix = {
            "mlplan": "mlplan",
            "thresholdout": "tho",
            "bergman_aggressive": "bergman_aggressive"
        }.get(task.mitigation_strategy, f"{task.evaluation.resampling}{'_sel' if task.evaluation.selection_size and task.evaluation.selection_size > 0 else ''}")

        reshuffling_prefix = "reshuf_" if task.evaluation.reshuffle and task.mitigation_strategy != "mlplan" else ""

        task.result_path = (f"{task.dataset_id}_"
                            f"rep{task.outer_evaluation.repeat}_"
                            f"fold{task.outer_evaluation.fold}_"
                            f"{task.optimizer}_"
                            f"{task.metric}_"
                            f"{reshuffling_prefix}"
                            f"{strategy_suffix}")

    print(f"Tasks collected: {len(tasks)}")

    # CRITICAL: MAKE SURE ALL RESULT PATHS ARE UNIQUE, OTHERWISE WE WILL OVERWRITE RESULTS
    assert len(set([t.result_path for t in tasks])) == len(tasks)

    if not tasks[0].store_vectors_google_cloud:
        print("Labels/preds not stored to cloud")

    if not tasks[0].store_results_google_cloud:
        print("WARNING: CSV results not stored to cloud")

    if not tasks[0].offline_data_loading:
        print("Data will loaded from openml")

    if tasks[0].debug:
        print("DEBUG MODE ON")

    return Experiment(
        f"{model_string}_{dataset_id}",
        tasks,
        output_dir=output_dir,
    )