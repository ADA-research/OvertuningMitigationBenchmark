import yaml

from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, Union


@dataclass
class OuterEvaluationConfig:
    """Configuration for outer dataset evaluation"""
    resampling: str = "holdout"
    train_size: float = 0.5
    n_folds: int = 1
    n_repeats: int = 1

    # Every task will execute HPO on 1 outer repeat/fold combination
    # We need the number of folds and repeats to select the correct combination
    # For holdout, we do not use the fold and repeat

    # Fold to run this task on
    fold: int = 0

    # Repeat to run this task on
    repeat: int = 0


@dataclass
class InnerEvaluationConfig:
    """Configuration for model evaluation"""
    resampling: str = "holdout"
    val_size: float = 0.2
    selection_size: float | None = None
    n_folds: int = 1
    reshuffle: bool = False
    retrain: bool = True
    n_repeats: int = 1


@dataclass
class SearchSpaceConfig:
    """Search space configuration"""
    # Estimator components
    classifiers: List[str] = field(default_factory=list)
    regressors: List[str] = field(default_factory=list)

    # Preprocessing components
    scalers: List[str] = field(default_factory=list)
    dim_reducers: List[str] = field(default_factory=list)
    imputers: List[str] = field(default_factory=list)
    encoders: List[str] = field(default_factory=list)
    feat_selectors: List[str] = field(default_factory=list)


@dataclass
class TaskConfig:
    """
    """
    # Basic task information (from your config)
    task_id: int = 0
    random_state: int = 78
    problem_type: str = "binary"
    label_map: Optional[Dict[int, str]] = None
    debug: bool = False

    # Dataset and experiment configuration
    dataset_id: int = 0
    optimizer: str = "random_search"
    metric: str = "accuracy"
    offline: bool = False
    offline_data_loading: bool = False
    store_results_google_cloud: bool = True  # Flag to store results in GCS
    store_vectors_google_cloud: bool = True  # Flag to store predictions and labels in GCS
    model_threads: int = 1

    # Outer Evaluation configuration
    outer_evaluation: OuterEvaluationConfig = field(default_factory=OuterEvaluationConfig)

    # Inner Evaluation configuration
    evaluation: InnerEvaluationConfig = field(default_factory=InnerEvaluationConfig)

    # Search space configuration
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)

    iterations: int = 250

    # Bayesian optimization warm-start configuration
    bo_initial_random_iterations: int = 25
    smac_surrogate_model: str = "random_forest"
    smac_surrogate_random_forest_n_trees: int = 10 # Only relevant if smac_surrogate_model == "random_forest"

    # Mitigation and racing strategies (can be added as needed)
    mitigation_strategy: str = "none"
    racing_strategy: str = "none"

    result_path: str = ""

    # # By default, store results in a directory with task id
    # @property
    # def result_path(self):
    #     return f"{self.dataset_id}_rep{self.outer_evaluation.repeat}_fold{self.outer_evaluation.fold}_{self.optimizer}_{self.metric}_{self.evaluation.resampling}"

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'TaskConfig':
        """Create TaskConfig from config dictionary (e.g., loaded from YAML)"""

        # Parse evaluation config
        outer_eval_config_dict = config_dict['outer_evaluation']
        outer_eval_config = OuterEvaluationConfig()

        # Convert list format to dict format for easier processing
        if isinstance(outer_eval_config_dict, list):
            outer_eval_dict = {}
            for item in outer_eval_config_dict:
                if isinstance(item, dict):
                    outer_eval_dict.update(item)
            outer_eval_config_dict = outer_eval_dict

        outer_eval_config.resampling = outer_eval_config_dict['resampling']
        outer_eval_config.train_size = outer_eval_config_dict['train_size']
        outer_eval_config.n_folds = outer_eval_config_dict['n_folds']
        outer_eval_config.n_repeats = outer_eval_config_dict['n_repeats']
        outer_eval_config.fold = outer_eval_config_dict['fold']
        outer_eval_config.repeat = outer_eval_config_dict['repeat']

        eval_config_dict = config_dict['evaluation']
        eval_config = InnerEvaluationConfig()

        # Convert list format to dict format for easier processing
        if isinstance(eval_config_dict, list):
            eval_dict = {}
            for item in eval_config_dict:
                if isinstance(item, dict):
                    eval_dict.update(item)
            eval_config_dict = eval_dict

        eval_config.resampling = eval_config_dict['resampling']
        eval_config.val_size = eval_config_dict['val_size']
        eval_config.selection_size = eval_config_dict['selection_size']
        eval_config.reshuffle = eval_config_dict['reshuffle']
        eval_config.retrain = eval_config_dict['retrain']
        eval_config.n_folds = eval_config_dict['n_folds']
        eval_config.n_repeats = eval_config_dict['n_repeats']

        # Parse search space config
        search_space = SearchSpaceConfig()
        search_space.classifiers = config_dict['classifiers']
        search_space.regressors = config_dict['regressors']

        preprocessors = config_dict['preprocessors']
        search_space.scalers = preprocessors['scalers']
        search_space.dim_reducers = preprocessors['dim_reducers']
        search_space.imputers = preprocessors['imputers']
        search_space.encoders = preprocessors['encoders']
        search_space.feat_selectors = preprocessors['feat_selectors']

        return cls(
            task_id=config_dict['task_id'],
            random_state=config_dict['random_state'],
            problem_type=config_dict['problem_type'],
            dataset_id=config_dict['dataset_id'],
            debug=config_dict['debug'],
            optimizer=config_dict['optimizer'],
            metric=config_dict['metric'],
            offline=config_dict['offline'],
            offline_data_loading=config_dict['offline_data_loading'],
            store_results_google_cloud=config_dict['store_results_google_cloud'],
            store_vectors_google_cloud=config_dict['store_vectors_google_cloud'],
            model_threads=config_dict.get('model_threads', 1),
            outer_evaluation=outer_eval_config,
            evaluation=eval_config,
            search_space=search_space,
            iterations=config_dict['iterations'],
            bo_initial_random_iterations=config_dict['bo_initial_random_iterations'],
            smac_surrogate_model=config_dict['smac_surrogate_model'],
            smac_surrogate_random_forest_n_trees=config_dict['smac_surrogate_random_forest_n_trees'],
            mitigation_strategy=config_dict['mitigation_strategy'],
            racing_strategy=config_dict['racing_strategy']
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "random_state": self.random_state,
            "problem_type": self.problem_type,
            "dataset_id": self.dataset_id,
            "debug": self.debug,
            "optimizer": self.optimizer,
            "metric": self.metric,
            "offline": self.offline,
            "offline_data_loading": self.offline_data_loading,
            "store_results_google_cloud": self.store_results_google_cloud,
            "store_vectors_google_cloud": self.store_vectors_google_cloud,
            "model_threads": self.model_threads,
            "outer_evaluation": {
                "resampling": self.outer_evaluation.resampling,
                "train_size": self.outer_evaluation.train_size,
                "n_folds": self.outer_evaluation.n_folds,
                "n_repeats": self.outer_evaluation.n_repeats,
                "fold": self.outer_evaluation.fold,
                "repeat": self.outer_evaluation.repeat,
            },
            "evaluation": {
                "resampling": self.evaluation.resampling,
                "val_size": self.evaluation.val_size,
                "selection_size": self.evaluation.selection_size,
                "reshuffle": self.evaluation.reshuffle,
                "retrain": self.evaluation.retrain,
                "n_folds": self.evaluation.n_folds,
                "n_repeats": self.evaluation.n_repeats,
            },
            "search_space": {
                "classifiers": self.search_space.classifiers,
                "regressors": self.search_space.regressors,
                "preprocessors": {
                    "scalers": self.search_space.scalers,
                    "dim_reducers": self.search_space.dim_reducers,
                    "imputers": self.search_space.imputers,
                    "encoders": self.search_space.encoders,
                    "feat_selectors": self.search_space.feat_selectors,
                }
            },
            "max_iterations": self.iterations,
            "bo_initial_random_iterations": self.bo_initial_random_iterations,
            "smac_surrogate_model": self.smac_surrogate_model,
            "smac_surrogate_random_forest_n_trees": self.smac_surrogate_random_forest_n_trees,
            "mitigation_strategy": self.mitigation_strategy,
            "racing_strategy": self.racing_strategy,
        }

    def to_yaml(self, file_path: str) -> str:
        """Convert task to YAML format"""

        task_dict = self.to_dict()
        yaml_str = yaml.dump(task_dict, default_flow_style=False, sort_keys=False)

        if file_path:
            with open(file_path, 'w') as f:
                f.write(yaml_str)

        return yaml_str


class DefaultTaskConfig(TaskConfig):
    def __init__(self):
        outer_evaluation = OuterEvaluationConfig(
            resampling="holdout",
            train_size=500,
            n_folds=1,
            n_repeats=1,
            fold=0,
            repeat=0
        )

        evaluation = InnerEvaluationConfig(
            resampling="holdout",
            val_size=0.2,
            selection_size=None,
            n_folds=1,
            reshuffle=False,
            retrain=True,
            n_repeats=1
        )

        search_space = SearchSpaceConfig(
            classifiers=["LGBM"],
            regressors=["LGBM"],
            scalers=["StandardScaler"],
            dim_reducers=['None'],
            imputers=['NumericalSimpleImputer'],
            encoders=['OrdinalEncoder'],
            feat_selectors=['None']
        )

        super().__init__(
            task_id=0,
            random_state=0,
            problem_type="binary",
            debug=True,
            dataset_id=1590,
            offline_data_loading=True,
            optimizer="smac",
            metric="accuracy",
            offline=False,
            outer_evaluation=outer_evaluation,
            evaluation=evaluation,
            search_space=search_space,
            iterations=5,
            mitigation_strategy="none",
            racing_strategy="none"
        )
