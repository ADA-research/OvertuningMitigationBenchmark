import yaml
import random

from src.experiments.experiment import Experiment
from src.datasets.tabarena_dataloader import TabarenaDataLoader
from src.experiments.task.task_config import TaskConfig










def hebo():
    random.seed(0)

    with open("src/experiments/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    task_config = TaskConfig().from_config(config)

    task_config.dataset_id = 1590
    task_config.offline_data_loading = True

    task_config.problem_type = "binary"

    task_config.outer_evaluation.resampling = "holdout"
    task_config.outer_evaluation.train_size = 1250

    task_config.iterations = 250
    task_config.bo_initial_random_iterations = 25
    task_config.optimizer = "hebo"
    task_config.smac_surrogate_model = "gaussian_process"

    # Minimalistic search space
    task_config.search_space.scalers = ["StandardScaler"]
    task_config.search_space.encoders = ["OrdinalEncoder"]
    task_config.search_space.dim_reducers = ["None"]
    task_config.search_space.feat_selectors = ["None"]
    task_config.search_space.imputers = ["NumericalSimpleImputer"]
    task_config.search_space.classifiers = ["LGBM"]

    task_config.store_results_google_cloud = False
    task_config.store_vectors_google_cloud = False

    task_config.result_path = f"hebo_test"

    return Experiment(
        "hebo_test",
        [task_config]
    )

def mlplan(random_state: int = 42) -> Experiment:
    random.seed(random_state)

    with open("src/experiments/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    task_config = TaskConfig().from_config(config)

    task_config.dataset_id = 1590
    task_config.offline_data_loading = True

    task_config.problem_type = "binary"

    task_config.outer_evaluation.resampling = "holdout"
    task_config.outer_evaluation.train_size = 500

    task_config.iterations = 250
    task_config.bo_initial_random_iterations = 25
    task_config.optimizer = "smac"
    task_config.smac_surrogate_model = "random_forest"
    task_config.mitigation_strategy = "mlplan"

    # Minimalistic search space
    task_config.search_space.scalers = ["StandardScaler"]
    task_config.search_space.encoders = ["OrdinalEncoder"]
    task_config.search_space.dim_reducers = ["None"]
    task_config.search_space.feat_selectors = ["None"]
    task_config.search_space.imputers = ["NumericalSimpleImputer"]
    task_config.search_space.classifiers = ["RandomForest", "DecisionTree", "LogisticRegression"]

    task_config.store_results_google_cloud = False
    task_config.store_vectors_google_cloud = False

    task_config.result_path = f"mlplan_test"

    return Experiment(
        "mlplan_test",
        [task_config]
    )

def makarova(random_state: int = 42) -> Experiment:
    random.seed(random_state)

    with open("src/experiments/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    tasks = []

    for seed in [random.randint(1000000, 999999999) for _ in range(100)]:
        for num_train_samples in [500, 1000, 5000]:
            for reshuffling in [True, False]:
                task_config = TaskConfig().from_config(config)
                task_config.dataset_id = 1590
                task_config.offline_data_loading = True
                task_config.problem_type = "binary"

                task_config.outer_evaluation.resampling = "holdout"
                task_config.outer_evaluation.train_size = num_train_samples

                task_config.evaluation.resampling = "cv"
                task_config.evaluation.n_folds = 5
                task_config.evaluation.retrain = True
                task_config.evaluation.reshuffle = reshuffling

                task_config.iterations = 250
                task_config.bo_initial_random_iterations = 25
                task_config.optimizer = "smac"

                task_config.smac_surrogate_model = "gaussian_process"
                task_config.smac_surrogate_random_forest_n_trees = 10

                # Minimalistic search space
                task_config.search_space.scalers = ["StandardScaler"]
                task_config.search_space.encoders = ["OrdinalEncoder"]
                task_config.search_space.dim_reducers = ["None"]
                task_config.search_space.feat_selectors = ["None"]
                task_config.search_space.imputers = ["NumericalSimpleImputer"]
                task_config.search_space.classifiers = ["LGBM"]

                task_config.store_results_google_cloud = True
                task_config.store_vectors_google_cloud = False

                task_config.result_path = f"makarova_lgbm_{seed}_{num_train_samples}_reshuf{reshuffling}"
                tasks.append(task_config)

    return Experiment(
        "makarova_2",
        tasks
    )


def bergman_vs_5cv(dataset_id: int, random_state: int, num_repeats: int) -> Experiment:
    random.seed(random_state)
    tasks = []

    with open("src/experiments/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Start with seed to ensure consistent data splitting across experiments
    for seed in [random.randint(1000000, 999999999) for _ in range(num_repeats)]:
        for num_train_samples in [5000, 500, 1000]:
            for bergman in [True, False]:
                task_id = random.randint(1000000, 999999999)

                task_config = TaskConfig.from_config(config)
                task_config.task_id = task_id

                task_config.optimizer = "smac"
                task_config.dataset_id = dataset_id
                task_config.random_state = seed

                task_config.evaluation.resampling = "cv"
                task_config.evaluation.n_folds = 5

                if bergman:
                    task_config.racing_strategy = "bergman_robust"
                    task_config.evaluation.n_repeats = 5
                else:
                    task_config.evaluation.n_repeats = 1

                task_config.evaluation.train_size = num_train_samples
                task_config.evaluation.retrain = True

                task_config.result_path = f"{seed}_{num_train_samples}_{'Bergman' if bergman else 'CV'}"

                tasks.append(task_config)

    return Experiment(
        f"Bergman5x5_vs_5cv",
        tasks,
        continue_experiment=False
    )


def all_binary_test():
    data_loader = TabarenaDataLoader()
    tasks = []

    with open("src/experiments/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    for ds in data_loader.get_all_binary():
        task_id = random.randint(1000000, 999999999)

        task_config = TaskConfig.from_config(config)
        task_config.task_id = task_id
        task_config.random_state = task_id

        task_config.optimizer = "smac"
        task_config.dataset_id = ds

        task_config.evaluation.resampling = "cv"
        task_config.evaluation.n_folds = 2
        task_config.iterations = 20
        task_config.bo_initial_random_iterations = 20
        task_config.evaluation.train_size = 0.66
        task_config.evaluation.retrain = True
        task_config.evaluation.reshuffle = False
        task_config.store_results_google_cloud = False

        task_config.result_path = f"ds_{ds}"

        tasks.append(task_config)

    return Experiment(
        f"AllBinaryTest",
        tasks,
        continue_experiment=False
    )


def reshuffling_bo(dataset_id: int, random_state: int, num_repeats: int) -> Experiment:
    random.seed(random_state)
    tasks = []

    with open("src/experiments/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Start with seed to ensure consistent data splitting across experiments
    for seed in [random.randint(1000000, 999999999) for _ in range(num_repeats)]:
        for num_train_samples in [5000, 500, 1000]:
            for cv in [5, 10]:
                for reshuffling in [True, False]:
                    task_id = random.randint(1000000, 999999999)

                    task_config = TaskConfig.from_config(config)
                    task_config.task_id = task_id

                    task_config.optimizer = "smac"
                    task_config.dataset_id = dataset_id
                    task_config.random_state = seed

                    task_config.evaluation.resampling = "cv"
                    task_config.evaluation.n_folds = cv
                    task_config.evaluation.train_size = num_train_samples
                    task_config.evaluation.retrain = True
                    task_config.evaluation.reshuffle = reshuffling

                    task_config.result_path = f"{seed}_{num_train_samples}_{cv}cv_{'reshuffling' if reshuffling else ''}"

                    tasks.append(task_config)

    return Experiment(
        f"Reshuffling",
        tasks,
        continue_experiment=False
    )


def bergman_reshuffling_bo(dataset_id: int, random_state: int, num_repeats: int) -> Experiment:
    random.seed(random_state)
    tasks = []

    with open("default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Start with seed to ensure consistent data splitting across experiments
    for seed in [random.randint(1000000, 999999999) for _ in range(num_repeats)]:
        for num_train_samples in [500, 1000, 5000]:
            for cv in [5, 10]:
                for reshuffling in [True, False]:
                    for bergman in ["robust", "forgiving", "none"]:

                        task_id = random.randint(1000000, 999999999)

                        task_config = TaskConfig.from_config(config)
                        task_config.task_id = task_id

                        task_config.optimizer = "smac"
                        task_config.dataset_id = dataset_id
                        task_config.random_state = seed

                        task_config.evaluation.resampling = "cv"
                        task_config.evaluation.n_folds = cv
                        task_config.evaluation.train_size = num_train_samples
                        task_config.evaluation.retrain = True
                        task_config.evaluation.reshuffle = reshuffling

                        if bergman == "forgiving":
                            task_config.racing_strategy = "bergman_forgiving"

                        elif bergman == "robust":
                            task_config.racing_strategy = "bergman_robust"

                        else:
                            task_config.racing_strategy = "none"

                        task_config.result_path = f"{seed}_{num_train_samples}_{cv}cv_{bergman if bergman else 'noRacing'}_reshuffling{reshuffling}"

                        tasks.append(task_config)

        return Experiment(
            f"Bergman_BO",
            tasks,
            continue_experiment=False
        )


def bayesian_optimization(dataset_id: int, random_state: int, num_repeats: int) -> Experiment:
    random.seed(random_state)
    tasks = []

    with open("default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Start with seed to ensure consistent data splitting across experiments
    for seed in [random.randint(1000000, 999999999) for _ in range(num_repeats)]:
        for num_train_samples in [500, 1000, 5000]:
            for cv in [5]:
                task_id = random.randint(1000000, 999999999)

                task_config = TaskConfig.from_config(config)
                task_config.task_id = task_id

                task_config.optimizer = "smac"
                task_config.dataset_id = dataset_id
                task_config.random_state = seed

                task_config.evaluation.n_folds = cv

                task_config.evaluation.train_size = num_train_samples
                task_config.evaluation.retrain = True

                task_config.result_path = f"{seed}_{num_train_samples}_{cv}cv"

                tasks.append(task_config)

    return Experiment(
        f"BO_1590_default",
        tasks,
        continue_experiment=False
    )


def thresholdout_initial_experiments(dataset_id: int, random_state: int, num_repeats: int) -> Experiment:
    """
    Creates a list of tasks for the following experiments:
    - Always retrains
    - Compares dataset sizes of 500, 1000, and 5000 samples
    - Compares THO and no THO


    :param dataset_id: The identifier of the dataset for which tasks are configured.
    :type dataset_id: int or str
    :param random_state: The seed for the random number generator to ensure reproducibility.
    :type random_state: int
    :param num_repeats: The number of distinct random seeds to generate tasks for.
    :type num_repeats: int
    :return: A list of task configurations for the specified experiment setup.
    :rtype: list
    """
    random.seed(random_state)
    tasks = []

    with open("default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Start with seed to ensure consistent data splitting across experiments
    for seed in [random.randint(1000000, 999999999) for _ in range(num_repeats)]:
        for thresholdout in [True, False]:
            for num_train_samples in [500, 1000, 5000]:
                for holdout_fraction in [0.2, 0.33]:
                    task_id = random.randint(1000000, 999999999)

                    task_config = TaskConfig.from_config(config)
                    task_config.task_id = task_id

                    if thresholdout:
                        task_config.mitigation_strategy = "thresholdout"

                    task_config.optimizer = "smac"
                    task_config.dataset_id = dataset_id
                    task_config.random_state = seed

                    task_config.evaluation.resampling = "holdout"
                    task_config.evaluation.val_size = val_size

                    task_config.evaluation.train_size = num_train_samples

                    task_config.evaluation.retrain = True

                    task_config.result_path = f"{seed}_{num_train_samples}_{holdout_fraction}val_{'tho' if thresholdout else 'noTho'}"

                    tasks.append(task_config)

    return Experiment(
        f"Thresholdout_adult",
        tasks,
        continue_experiment=False
    )


def selection_set_experiments(dataset_id: int, random_state: int, num_repeats: int) -> Experiment:
    """
    Creates a list of tasks for the following experiments:
    - Compares no selection set, 10% selection set, and 25% selection set
    - Always retrains
    - Compares 3, 5, and 10-fold CV
    - Only Bayesian optimization
    - Compares dataset sizes of 500, 1000, and 5000 samples


    :param dataset_id: The identifier of the dataset for which tasks are configured.
    :type dataset_id: int or str
    :param random_state: The seed for the random number generator to ensure reproducibility.
    :type random_state: int
    :param num_repeats: The number of distinct random seeds to generate tasks for.
    :type num_repeats: int
    :return: A list of task configurations for the specified experiment setup.
    :rtype: list
    """
    random.seed(random_state)
    tasks = []

    with open("default.yaml", "r") as f:
        config = yaml.safe_load(f)

    for seed in [random.randint(1000000, 999999999) for _ in range(num_repeats)]:
        for selection_size in [None, 0.1, 0.25]:
            for num_train_samples in [500, 1000, 5000]:
                for cv in [3, 5, 10]:
                    task_id = random.randint(1000000, 999999999)

                    task_config = TaskConfig.from_config(config)
                    task_config.task_id = task_id
                    task_config.optimizer = "smac"
                    task_config.dataset_id = dataset_id
                    task_config.random_state = seed

                    task_config.evaluation.selection_size = selection_size
                    task_config.evaluation.resampling = "cv"
                    task_config.evaluation.n_folds = cv
                    task_config.evaluation.train_size = num_train_samples

                    task_config.evaluation.retrain = True

                    task_config.result_path = f"{seed}_{num_train_samples}_{cv}cv_{selection_size}selection"

                    tasks.append(task_config)

    return Experiment(
        f"Selection_set_CV",
        tasks,
        continue_experiment=False
    )


def selection_set_vs_default(dataset_id, num_train_samples, random_state):
    """
    Create simple experiment testing BO and RS on all classification datasets, using holdout
    Returns: Experiment object
    """
    random.seed(random_state)
    tasks = []

    for _ in range(100):

        seed = random.randint(1000000, 999999999)

        for selection_size in [None, 0.1, 0.25]:
            with open("default.yaml", "r") as f:
                config = yaml.safe_load(f)

            task_id = random.randint(1000000, 999999999)

            task_config = TaskConfig.from_config(config)
            task_config.task_id = task_id
            task_config.optimizer = "smac"
            task_config.dataset_id = dataset_id
            task_config.random_state = seed

            task_config.evaluation.selection_size = selection_size
            task_config.evaluation.resampling = "cv"
            task_config.evaluation.n_folds = 5
            task_config.evaluation.train_size = num_train_samples
            task_config.evaluation.retrain = True

            tasks.append(task_config)

    return Experiment(
        f"Selection_set_{num_train_samples}_samples",
        tasks,
        continue_experiment=False
    )


def bo_on_one_dataset_multiple_seeds(dataset_id, random_state):
    """
    Create simple experiment testing BO on one dataset, using holdout
    Returns: Experiment object
    """
    random.seed(random_state)
    tasks = []

    for _ in range(100):
        with open("default.yaml", "r") as f:
            config = yaml.safe_load(f)

        seed = random.randint(1000000, 999999999)
        task_id = random.randint(1000000, 999999999)

        config["task_id"] = task_id
        config["optimizer"] = "smac"
        config["dataset_id"] = dataset_id
        config["random_state"] = seed

        task_config = TaskConfig.from_config(config)

        tasks.append(task_config)

    return Experiment(
        "BO_ONE_DATASET",
        tasks,
        continue_experiment=False
    )


def bo(random_state):
    """
    Create simple experiment testing BO on all classification datasets, using holdout
    Returns: Experiment object
    """
    """
        Create simple experiment testing BO and RS on all classification datasets, using holdout
        Returns: Experiment object

        """
    random.seed(random_state)

    data_loader = TabarenaDataLoader()

    print(data_loader.get_all_binary())
    assert False

    tasks = []

    for method in ["smac"]:
        for dataset_id in data_loader.get_all_binary():
            with open("default.yaml", "r") as f:
                config = yaml.safe_load(f)

            seed = random.randint(1000000, 999999999)
            task_id = random.randint(1000000, 999999999)

            config["task_id"] = task_id
            config["optimizer"] = method
            config["dataset_id"] = dataset_id
            config["random_state"] = seed

            task_config = TaskConfig.from_config(config)

            tasks.append(task_config)

    return Experiment(
        "BO",
        tasks,
        continue_experiment=False
    )


def bo_vs_rs(random_state):
    """
    Create simple experiment testing BO and RS on all classification datasets, using holdout
    Returns: Experiment object

    """
    random.seed(random_state)

    data_loader = TabarenaDataLoader()

    tasks = []

    for method in ["random_search", "smac"]:
        for dataset_id in data_loader.get_all_binary():
            with open("default.yaml", "r") as f:
                config = yaml.safe_load(f)

            seed = random.randint(1000000, 999999999)
            task_id = random.randint(1000000, 999999999)

            config["task_id"] = task_id
            config["optimizer"] = method
            config["dataset_id"] = dataset_id
            config["random_state"] = seed

            task_config = TaskConfig.from_config(config)

            tasks.append(task_config)

    return Experiment(
        "BO_vs_RS",
        tasks,
        continue_experiment=False
    )


def get_debug_task(random_state) -> Experiment:
    random.seed(random_state)
    tasks = []

    with open("default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Start with seed to ensure consistent data splitting across experiments
    task_id = random.randint(1000000, 999999999)

    task_config = TaskConfig.from_config(config)
    task_config.task_id = task_id

    task_config.optimizer = "smac"
    task_config.dataset_id = 1590
    task_config.random_state = random_state
    task_config.iterations = 10
    task_config.bo_initial_random_iterations = 5

    task_config.evaluation.resampling = "cv"
    task_config.evaluation.n_folds = 5
    task_config.evaluation.train_size = 500
    task_config.evaluation.retrain = True
    task_config.evaluation.reshuffle = False

    task_config.result_path = f"__debug__"

    tasks.append(task_config)

    return Experiment(
        f"DEBUG",
        tasks,
        continue_experiment=False
    )
