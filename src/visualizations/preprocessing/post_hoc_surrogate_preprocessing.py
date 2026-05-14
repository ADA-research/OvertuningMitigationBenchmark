"""
Post-hoc surrogate preprocessing for fair surrogate-based incumbent selection.

This module recomputes surrogate-based incumbent trajectories for benchmark runs
by, at each iteration t, fitting the surrogate on observations 0..t and
predicting all evaluated configurations 0..t with that same fitted model.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch
import yaml
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
	CategoricalHyperparameter,
	Constant,
	FloatHyperparameter,
	IntegerHyperparameter,
	OrdinalHyperparameter,
)
from tqdm import tqdm
from joblib import Parallel, delayed

from hebo.acquisitions.acq import Mean

from src.optimizers.bo_hebo import HEBOOptimizer
from src.optimizers.bo_smac import SMACOptimizer
from src.search_space.search_space import SearchSpace
from src.visualizations.preprocessing.preprocessing import (
	calculate_trajectories,
	convert_negative_metric_to_loss,
	extract_metadata_from_config,
	get_ensembled_test_scores_per_iteration,
)

from src.visualizations.preprocessing.shared_utils import resolve_artifact_paths


REQUIRED_TRAJECTORY_COLUMNS = [
	"iteration",
	"experiment_id",
	"experiment_source",
	"raw_experiment_id",
	"dataset_id",
	"model",
	"optimizer",
	"metric",
	"problem_type",
	"mitigation",
	"reshuffling",
	"inner_split",
	"selection_set_size",
	"repetition",
	"outer_fold",
	"random_state",
	"outer_n_folds",
	"outer_n_repeats",
	"val_performance",
	"retrain_test_loss",
	"ensembled_test_loss",
	"retrain_meta_overfitting",
	"ensemble_meta_overfitting",
	"retrain_regret",
	"ensemble_regret",
	"retrain_overtuning",
	"ensemble_overtuning",
	"relative_retrain_overtuning",
	"relative_ensemble_overtuning",
]

def is_benchmark_experiment(task_config: dict) -> bool:
    """
    Check if a task config corresponds to a benchmark experiment:
    - No mitigation strategy (mitigation_strategy == 'none')
    - No selection set (evaluation.selection_size == 0 or None)
    - No reshuffling (evaluation.reshuffle == False)
    """
    mitigation = task_config.get('mitigation_strategy', 'none')
    if mitigation != 'none':
        return False
    
    evaluation = task_config.get('evaluation', {})
    selection_size = evaluation.get('selection_size')
    if selection_size and selection_size > 0:
        return False
    
	# Although not a benchmark experiment, we do want to investigate reshuffling
    # reshuffling = evaluation.get('reshuffle', False)
    # if reshuffling:
    #     return False
    
    return True


def _warmup_iteration_count(task_config: dict) -> int:
	return int(task_config.get("bo_initial_random_iterations", 25))


def _observed_validation_incumbent(values: np.ndarray, upto_index: int) -> Tuple[int, float]:
	incumbent = int(np.argmin(values[: upto_index + 1]))
	return incumbent, float(values[incumbent])


def load_task_config(task_dir: Path) -> dict:
	config_path = task_dir / "task_config.yaml"
	if not config_path.exists():
		raise FileNotFoundError(f"task_config.yaml not found in {task_dir}")

	with open(config_path, "r") as handle:
		return yaml.safe_load(handle)


def build_search_space_from_task_config(task_config: dict):
	return SearchSpace(task_config).get_space()


def _parse_iteration_key(key: str) -> int:
	if not key.startswith("iter_"):
		raise ValueError(f"Invalid iteration key in configs.yaml: '{key}'")
	return int(key.split("iter_")[-1])


def load_iteration_configs(configs_path: Path) -> Dict[int, dict]:
	if not configs_path.exists():
		raise FileNotFoundError(f"configs.yaml not found: {configs_path}")

	with open(configs_path, "r") as handle:
		raw = yaml.safe_load(handle) or {}

	parsed: Dict[int, dict] = {}
	for key, value in raw.items():
		parsed[_parse_iteration_key(str(key))] = dict(value)

	return parsed


def _cast_categorical_value(raw_value, hp: CategoricalHyperparameter):
	choices = list(hp.choices)

	if raw_value in choices:
		return raw_value

	raw_str = str(raw_value)

	if raw_str == "None" and any(choice is None for choice in choices):
		return None

	for choice in choices:
		if str(choice) == raw_str:
			return choice

	return raw_value


def cast_value_to_hyperparameter(raw_value, hp):
	if raw_value is None:
		return None

	if isinstance(hp, Constant):
		target_value = hp.value
		if isinstance(target_value, bool):
			if isinstance(raw_value, str):
				return raw_value.lower() == "true"
			return bool(raw_value)
		if isinstance(target_value, int):
			return int(float(raw_value))
		if isinstance(target_value, float):
			return float(raw_value)
		if target_value is None:
			return None
		return str(raw_value)

	if isinstance(hp, CategoricalHyperparameter):
		return _cast_categorical_value(raw_value, hp)

	if isinstance(hp, OrdinalHyperparameter):
		return _cast_categorical_value(raw_value, CategoricalHyperparameter(hp.name, hp.sequence))

	if isinstance(hp, IntegerHyperparameter):
		return int(float(raw_value))

	if isinstance(hp, FloatHyperparameter):
		return float(raw_value)

	return raw_value


def reconstruct_configuration(search_space, raw_config: dict) -> Configuration:
	values = {}
	for hp in search_space.values():
		name = hp.name
		if name not in raw_config:
			continue
		values[name] = cast_value_to_hyperparameter(raw_config[name], hp)

	return Configuration(
		search_space,
		values=values,
		allow_inactive_with_values=True,
	)


def reconstruct_configurations_for_iterations(
	search_space,
	configs_per_iteration: Dict[int, dict],
	iterations: List[int],
) -> List[Configuration]:
	reconstructed = []
	for iteration in iterations:
		if iteration not in configs_per_iteration:
			raise KeyError(f"Missing iter_{iteration} in configs.yaml")
		reconstructed.append(reconstruct_configuration(search_space, configs_per_iteration[iteration]))
	return reconstructed


def _normalize_nullable_categorical_nans(df: pd.DataFrame, design_space) -> pd.DataFrame:
	if df is None or df.empty:
		return df

	normalized = df.copy()
	for name, param in design_space.paras.items():
		if name not in normalized.columns:
			continue

		categories = getattr(param, "categories", None)
		if categories is None:
			continue

		if any(category is None for category in categories):
			normalized[name] = normalized[name].astype(object)
			normalized[name] = normalized[name].where(normalized[name].notna(), None)

	return normalized


def configurations_to_hebo_dataframe(configurations: List[Configuration], design_space) -> pd.DataFrame:
	rows: List[dict] = []
	columns = list(design_space.paras.keys())

	for config in configurations:
		row = {}
		for col in columns:
			row[col] = config[col] if col in config else None
		rows.append(row)

	return pd.DataFrame(rows, columns=columns)


def _fit_predict_means_smac(
	optimizer: SMACOptimizer,
	configurations: List[Configuration],
	val_scores: np.ndarray,
	upto_index: int,
) -> np.ndarray:
	model = optimizer.smac._model

	X_obs = np.array([config.get_array() for config in configurations[: upto_index + 1]])
	y_obs_2d = val_scores[: upto_index + 1].reshape(-1, 1)

	try:
		model.train(X_obs, y_obs_2d)
	except Exception:
		model.train(X_obs, y_obs_2d.ravel())

	mean, _ = model.predict(X_obs)
	return np.asarray(mean, dtype=float).reshape(-1)


def compute_post_hoc_surrogate_incumbents_smac(
	configurations: List[Configuration],
	val_scores: List[float],
	search_space,
	task_config: dict,
	show_progress: bool,
	desc: str,
) -> Tuple[List[int], List[float]]:
	warmup_iterations = _warmup_iteration_count(task_config)

	with tempfile.TemporaryDirectory(prefix="post_hoc_smac_") as temp_output:
		optimizer = SMACOptimizer(
			search_space=search_space,
			initial_iterations=warmup_iterations,
			surrogate_model=str(task_config.get("smac_surrogate_model", "gaussian_process")),
			random_forest_n_trees=task_config.get("smac_surrogate_random_forest_n_trees"),
			output_directory=temp_output,
			random_state=task_config.get("random_state"),
		)

		values = np.asarray(val_scores, dtype=float)
		incumbent_indices: List[int] = []
		incumbent_means: List[float] = []

		iterator = range(len(configurations))
		if show_progress:
			iterator = tqdm(iterator, desc=desc)

		for t in iterator:
			if t < warmup_iterations:
				incumbent, mean = _observed_validation_incumbent(values, t)
				incumbent_indices.append(incumbent)
				incumbent_means.append(mean)
				continue

			means = _fit_predict_means_smac(
				optimizer=optimizer,
				configurations=configurations,
				val_scores=values,
				upto_index=t,
			)
			incumbent = int(np.argmin(means))
			incumbent_indices.append(incumbent)
			incumbent_means.append(float(means[incumbent]))

	return incumbent_indices, incumbent_means


def compute_post_hoc_surrogate_incumbents_hebo(
	configurations: List[Configuration],
	val_scores: List[float],
	search_space,
	task_config: dict,
	show_progress: bool,
	desc: str,
) -> Tuple[List[int], List[float]]:
	warmup_iterations = _warmup_iteration_count(task_config)

	optimizer = HEBOOptimizer(
		search_space=search_space,
		initial_iterations=warmup_iterations,
		random_state=task_config.get("random_state"),
	)

	values = np.asarray(val_scores, dtype=float)
	all_df = configurations_to_hebo_dataframe(configurations, optimizer.design_space)

	incumbent_indices: List[int] = []
	incumbent_means: List[float] = []

	iterator = range(len(configurations))
	if show_progress:
		iterator = tqdm(iterator, desc=desc)

	for t in iterator:
		if t < warmup_iterations:
			incumbent, mean = _observed_validation_incumbent(values, t)
			incumbent_indices.append(incumbent)
			incumbent_means.append(mean)
			continue

		y_obs = values[: t + 1].reshape(-1, 1)
		observed_df = _normalize_nullable_categorical_nans(all_df.iloc[: t + 1].copy(), optimizer.design_space)

		try:
			X_obs, Xe_obs = optimizer.design_space.transform(observed_df)
			model = optimizer.get_hebo_surrogate_model()
			model.fit(X_obs, Xe_obs, torch.FloatTensor(y_obs))

			mu = Mean(model)
			mean_tensor = mu(X_obs, Xe_obs)
			if isinstance(mean_tensor, torch.Tensor):
				means = mean_tensor.detach().cpu().numpy().astype(float).reshape(-1)
			else:
				means = np.asarray(mean_tensor, dtype=float).reshape(-1)

			incumbent = int(np.argmin(means))
			incumbent_indices.append(incumbent)
			incumbent_means.append(float(means[incumbent]))

		except Exception as exc:
			fallback_incumbent, fallback_mean = _observed_validation_incumbent(values, t)
			print(
				"[post_hoc_surrogate] WARNING: HEBO surrogate fit/predict failed at "
				f"iteration={t}; falling back to validation incumbent. error={exc}"
			)
			incumbent_indices.append(fallback_incumbent)
			incumbent_means.append(fallback_mean)

	return incumbent_indices, incumbent_means


def compute_post_hoc_surrogate_trajectories_for_run(
	task_dir: Path,
	experiment_name: str,
	show_progress: bool = True,
) -> Dict:
	task_dir = Path(task_dir)

	try:
		task_config = load_task_config(task_dir)

		if not is_benchmark_experiment(task_config):
			return {
				"status": "SKIPPED",
				"reason": "Not a benchmark experiment",
				"task_dir": str(task_dir),
			}

		optimizer_name = str(task_config.get("optimizer", "")).lower()
		if optimizer_name not in {"smac", "hebo"}:
			return {
				"status": "SKIPPED",
				"reason": f"Unsupported optimizer for surrogate post-hoc: {optimizer_name}",
				"task_dir": str(task_dir),
			}

		history_path = task_dir / "history.csv"
		configs_path = task_dir / "configs.yaml"
		if not history_path.exists():
			return {
				"status": "SKIPPED",
				"reason": "No history.csv found",
				"task_dir": str(task_dir),
			}

		history_df = pd.read_csv(history_path)
		if history_df.empty:
			return {
				"status": "SKIPPED",
				"reason": "history.csv is empty",
				"task_dir": str(task_dir),
			}

		grouped = history_df.sort_values("iteration").groupby("iteration", sort=True)
		iterations = sorted(grouped.groups.keys())

		val_scores = grouped["avg_val_score"].first().astype(float).tolist()
		retrained_test_scores = grouped["avg_test_score"].first().astype(float).tolist()

		with ExitStack() as stack:
			artifact_paths = resolve_artifact_paths(task_dir)
			artifacts = [stack.enter_context(np.load(path)) for path in artifact_paths]
			cv_test_scores_map = get_ensembled_test_scores_per_iteration(
				history_df,
				artifacts,
				problem_type=task_config["problem_type"],
				metric_name=task_config["metric"],
			)

		cv_ensembled_test_scores = [float(cv_test_scores_map[it]) for it in iterations]

		search_space = build_search_space_from_task_config(task_config)
		iter_configs = load_iteration_configs(configs_path)
		configurations = reconstruct_configurations_for_iterations(search_space, iter_configs, iterations)

		run_name = task_config.get("result_path", task_dir.name)
		if optimizer_name == "smac":
			incumbent_indices, incumbent_surrogate_means = compute_post_hoc_surrogate_incumbents_smac(
				configurations=configurations,
				val_scores=val_scores,
				search_space=search_space,
				task_config=task_config,
				show_progress=show_progress,
				desc=f"Post-hoc surrogate SMAC {run_name}/{experiment_name}",
			)
		else:
			incumbent_indices, incumbent_surrogate_means = compute_post_hoc_surrogate_incumbents_hebo(
				configurations=configurations,
				val_scores=val_scores,
				search_space=search_space,
				task_config=task_config,
				show_progress=show_progress,
				desc=f"Post-hoc surrogate HEBO {run_name}/{experiment_name}",
			)

		return {
			"status": "SUCCESS",
			"task_dir": str(task_dir),
			"iterations": iterations,
			"val_scores": val_scores,
			"retrained_test_scores": retrained_test_scores,
			"cv_ensembled_test_scores": cv_ensembled_test_scores,
			"post_hoc_surrogate_incumbent_indices": incumbent_indices,
			"post_hoc_surrogate_incumbent_means": incumbent_surrogate_means,
		}

	except Exception as exc:
		return {
			"status": "FAILED",
			"task_dir": str(task_dir),
			"error": str(exc),
			"traceback": traceback.format_exc(),
		}


def preprocess_post_hoc_surrogate_experiment(
	results_dir: str,
	n_jobs: int = 1,
	verbose: bool = True,
) -> Optional[pd.DataFrame]:
	results_path = Path(results_dir)
	if not results_path.exists():
		print(f"Results directory does not exist: {results_dir}")
		return None

	task_dirs = [directory for directory in results_path.iterdir() if directory.is_dir()]
	if not task_dirs:
		print(f"No task directories found in {results_dir}")
		return None

	if verbose:
		print(f"Found {len(task_dirs)} task directories")
		print("Computing post-hoc surrogate trajectories")

	results = Parallel(n_jobs=n_jobs)(
		delayed(compute_post_hoc_surrogate_trajectories_for_run)(
			task_dir=task_dir,
			experiment_name=results_path.name,
			show_progress=verbose,
		)
		for task_dir in tqdm(task_dirs, disable=not verbose, desc="Scanning runs")
	)

	raw_rows = []
	success_count = 0
	skip_count = 0
	fail_count = 0

	for result in results:
		if result["status"] == "SUCCESS":
			success_count += 1
			task_dir = Path(result["task_dir"])
			task_config = load_task_config(task_dir)

			metadata = extract_metadata_from_config(task_config)
			metadata["experiment_source"] = results_path.name
			metadata["raw_experiment_id"] = task_dir.name
			metadata["experiment_id"] = f"{results_path.name}::{task_dir.name}"

			for index, iteration in enumerate(result["iterations"]):
				row = {
					"iteration": int(iteration),
					"avg_val_score": float(result["val_scores"][index]),
					"retrained_test_score": float(result["retrained_test_scores"][index]),
					"cv_ensembled_test_score": float(result["cv_ensembled_test_scores"][index]),
					"post_hoc_surrogate_incumbent_iteration": int(
						result["post_hoc_surrogate_incumbent_indices"][index]
					),
					"post_hoc_surrogate_incumbent_mean": float(
						result["post_hoc_surrogate_incumbent_means"][index]
					),
				}
				row.update(metadata)
				raw_rows.append(row)

		elif result["status"] == "SKIPPED":
			skip_count += 1
			if verbose:
				print(f"Skipped {result['task_dir']}: {result['reason']}")
		else:
			fail_count += 1
			if verbose:
				print(f"Failed {result['task_dir']}: {result['error']}")

	if verbose:
		print(f"\nResults: {success_count} success, {skip_count} skipped, {fail_count} failed")

	if not raw_rows:
		print("No trajectory rows generated")
		return None

	results_df = pd.DataFrame(raw_rows)
	results_df = convert_negative_metric_to_loss(results_df)

	metadata_cols = [
		column
		for column in [
			"experiment_id",
			"experiment_source",
			"raw_experiment_id",
			"dataset_id",
			"model",
			"optimizer",
			"metric",
			"problem_type",
			"mitigation",
			"reshuffling",
			"inner_split",
			"selection_set_size",
			"repetition",
			"outer_fold",
			"random_state",
			"outer_n_folds",
			"outer_n_repeats",
		]
		if column in results_df.columns
	]

	trajectory_rows = []
	group_iterator = results_df.groupby("experiment_id")
	if verbose:
		group_iterator = tqdm(group_iterator, desc="Calculating post-hoc surrogate trajectories")

	for _, run_df in group_iterator:
		run_df = run_df.sort_values("iteration")

		val_scores = run_df["avg_val_score"].tolist()
		retrained_test_scores = run_df["retrained_test_score"].tolist()
		cv_ensembled_test_scores = run_df["cv_ensembled_test_score"].tolist()
		incumbent_indices = run_df["post_hoc_surrogate_incumbent_iteration"].astype(int).tolist()

		trajectory = calculate_trajectories(
			val_scores=val_scores,
			retrained_test_scores=retrained_test_scores,
			ensembled_test_scores=cv_ensembled_test_scores,
			incumbent_indices=incumbent_indices,
		)

		metadata = {column: run_df[column].iloc[0] for column in metadata_cols}
		iterations = run_df["iteration"].tolist()

		for index, iteration in enumerate(iterations):
			row = {"iteration": int(iteration)}
			row.update(metadata)
			for key, values in trajectory.items():
				if key == "scores_to_attach":
					continue
				row[key] = values[index]
			trajectory_rows.append(row)

	trajectories_df = pd.DataFrame(trajectory_rows)
	missing_required_columns = [
		column for column in REQUIRED_TRAJECTORY_COLUMNS if column not in trajectories_df.columns
	]
	if missing_required_columns:
		raise ValueError(f"Missing required trajectory columns: {missing_required_columns}")

	trajectories_df = trajectories_df[REQUIRED_TRAJECTORY_COLUMNS]

	results_output_path = results_path / "results_post_hoc_surrogate.csv"
	trajectories_output_path = results_path / "trajectories_post_hoc_surrogate.csv"
	results_df.to_csv(results_output_path, index=False)
	trajectories_df.to_csv(trajectories_output_path, index=False)

	if verbose:
		print(f"Saved post-hoc surrogate raw scores to {results_output_path}")
		print(f"Saved post-hoc surrogate trajectories to {trajectories_output_path}")
		print(f"Shape: {trajectories_df.shape}")

	return trajectories_df

