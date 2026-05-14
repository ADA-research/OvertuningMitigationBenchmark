import copy
import os

from src.experiments.experiment import Experiment

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

N_JOBS = 1

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LIT_LOGGER_INFO"] = "0"

import logging
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import torch

torch.set_num_threads(1)

from src.experiments.task.task_config import DefaultTaskConfig, TaskConfig
from src.experiments.task.task import run_task

task_config = TaskConfig()

# Constants not altered in different experiments
task_config.problem_type = "binary"
task_config.random_state = 78
task_config.offline = False
task_config.racing_strategy = "none"
task_config.mitigation_strategy = "none"
task_config.offline_data_loading = True

task_config.iterations = 5
task_config.bo_initial_random_iterations = 25  # Also used for HEBO
task_config.smac_surrogate_model = "gaussian_process"  # We only use GP in SMAC
task_config.smac_surrogate_random_forest_n_trees = 10
task_config.debug = False
task_config.store_results_google_cloud = False
task_config.store_vectors_google_cloud = False

task_config.search_space.classifiers = ["LGBM"]
task_config.search_space.regressors = ["LGBM"]
task_config.search_space.scalers = ["StandardScaler"]
task_config.search_space.encoders = ["OrdinalEncoder"]
task_config.search_space.dim_reducers = ["None"]
task_config.search_space.feat_selectors = ["None"]
task_config.search_space.imputers = ["NumericalSimpleImputer"]

# We always retrain
task_config.evaluation.retrain = True

# And use accuracy
task_config.metric = "accuracy"
task_config.optimizer = "random_search"
####################
# OUTER DATA SPLIT #
####################

# We follow TabArena in the number of repeats
# TabArena considers a dataset to be large if > 2,500 samples

task_config.outer_evaluation.fold = 0  # Determines the current fold split
task_config.outer_evaluation.repeat = 0  # Determines the current repeat
task_config.outer_evaluation.n_repeats = 16  # This tells the data splitter how many splits to make so the two above make sense
task_config.outer_evaluation.n_folds = 3  # This tells the data splitter how many splits to make so the two above make sense
task_config.outer_evaluation.resampling = "cv"  # This tells the data splitter how many splits to make so the two above make sense
task_config.outer_evaluation.train_size = 0  # Only used if resampling is not CV

# NUMBER OF DATASETS * 2 (RESAMPLING) * 5 (iterations), where resampling is 6 folds per iteration, and holdout 2
# So we benchmark 8 folds * 5 = 40 folds per dataset

task_config.dataset_id = "ilpd"

task_config.racing_strategy = "none"
task_config.mitigation_strategy = "none"
task_config.racing_strategy = "none"

# With two inner resampling methods
task_config.evaluation.resampling = "cv"
task_config.evaluation.n_repeats = 1

# Will be used if inner resampling is CV
task_config.evaluation.n_folds = 5

# Will be used if inner resampling is holdout
task_config.evaluation.val_size = 0.2

task_config.evaluation.selection_size = 0.0

tasks = []

for i in range(1):
    task_config.outer_evaluation.repeat = i
    task_config.result_path = f"HEBOTEST_rep{i}"

    tasks.append(copy.deepcopy(task_config))

experiment = Experiment(
    f"HEBOTEST",
    tasks
)

experiment.run(N_JOBS)
