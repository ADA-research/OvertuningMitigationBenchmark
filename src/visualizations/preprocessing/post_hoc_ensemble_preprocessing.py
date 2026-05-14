"""
Post-hoc ensemble preprocessing: Caruana greedy ensemble selection for trajectories.

This module computes trajectories using Caruana greedy ensemble selection,
applied only to benchmark experiments (no mitigation, no selection set, no reshuffling).
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import traceback

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed

from src.metrics.metric import Metric
from src.visualizations.preprocessing.preprocessing import (
    extract_metadata_from_config,
    convert_negative_metric_to_loss,
    calculate_trajectories,
)
from src.visualizations.preprocessing.shared_utils import (
    coerce_predictions_for_metric,
    resolve_artifact_paths,
)


PROBLEM_TYPE_TO_METRIC = {
    "binary": "roc_auc",
    "multiclass": "neg_log_loss",
    "regression": "neg_root_mean_squared_error",
}


def _sanitize_non_finite_array(
        values: np.ndarray,
        *,
        stats: Optional[dict],
        iteration: Optional[int],
) -> np.ndarray:
    """Replace non-finite values with 0.0 and track replacement statistics."""
    arr = np.asarray(values, dtype=np.float64)

    if stats is not None:
        stats['total_values_seen'] = stats.get('total_values_seen', 0) + int(arr.size)

    mask = ~np.isfinite(arr)
    replaced = int(mask.sum())
    if replaced == 0:
        return arr

    arr = arr.copy()
    arr[mask] = 0.0

    if stats is not None:
        stats['replaced_non_finite_values'] = stats.get('replaced_non_finite_values', 0) + replaced
        if iteration is not None:
            stats.setdefault('iterations_with_replacements', set()).add(int(iteration))

    return arr


def get_metric_name_from_problem_type(problem_type: str) -> str:
    try:
        return PROBLEM_TYPE_TO_METRIC[problem_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported problem_type '{problem_type}'. Expected one of: {list(PROBLEM_TYPE_TO_METRIC.keys())}"
        ) from exc


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
    
    reshuffling = evaluation.get('reshuffle', False)
    if reshuffling:
        return False
    
    return True


def load_task_config(task_dir: Path) -> dict:
    """Load task_config.yaml from task directory."""
    config_path = task_dir / "task_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"task_config.yaml not found in {task_dir}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_val_preds_per_iteration(
        df: pd.DataFrame,
        artifacts,
        problem_type: str,
    stats: Optional[dict] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[int]]:
    """
    Get validation predictions and labels per iteration using OOF concatenation across folds.
    
    Returns:
        (val_preds_per_iteration, val_labels_per_iteration)
        where each maps iteration -> concatenated OOF arrays.
    """
    if not isinstance(artifacts, list):
        artifacts = [artifacts]

    val_preds_per_iteration = {}
    val_labels_per_iteration = {}
    skipped_iterations: list[int] = []
    
    for iteration, group_df in df.groupby('iteration'):
        n_folds = len(group_df)
        fold_preds: list[np.ndarray] = []
        fold_labels: list[np.ndarray] = []
        missing_folds: list[tuple[int, str]] = []
        
        # Keep deterministic ordering for OOF concatenation.
        for _, fold_row in group_df.sort_values('fold').iterrows():
            fold_id = int(fold_row['fold'])
            pred_key = f"iter_{iteration}/fold_{fold_id}/val_preds"
            labels_key = f"iter_{iteration}/fold_{fold_id}/val_labels"
            found_pred = False
            found_labels = False
            
            for artifact in artifacts:
                if pred_key in artifact:
                    found_pred = True
                    pred = _sanitize_non_finite_array(
                        artifact[pred_key],
                        stats=stats,
                        iteration=int(iteration),
                    )
                    fold_preds.append(pred)
                if labels_key in artifact:
                    found_labels = True
                    labels = _sanitize_non_finite_array(
                        artifact[labels_key],
                        stats=stats,
                        iteration=int(iteration),
                    )
                    fold_labels.append(labels)

                if found_pred and found_labels:
                    break
            
            if not (found_pred and found_labels):
                missing_folds.append((fold_id, f"{pred_key} | {labels_key}"))
        
        if len(fold_preds) < max(1, n_folds - 1):
            skipped_iterations.append(int(iteration))
            if stats is not None:
                stats.setdefault('skipped_iterations', set()).add(int(iteration))
            print(
                f"[post_hoc] WARNING: Skipping iteration {iteration} due to missing "
                f"validation predictions ({len(fold_preds)}/{n_folds} folds)."
            )
            continue
        if len(fold_labels) != len(fold_preds):
            skipped_iterations.append(int(iteration))
            if stats is not None:
                stats.setdefault('skipped_iterations', set()).add(int(iteration))
            print(
                f"[post_hoc] WARNING: Skipping iteration {iteration} due to mismatched "
                f"validation preds/labels (preds={len(fold_preds)}, labels={len(fold_labels)})."
            )
            continue
        
        # OOF validation set is the concatenation of fold predictions and labels.
        ensemble_val_pred = np.concatenate(fold_preds, axis=0)
        val_labels = np.concatenate(fold_labels, axis=0)
        
        # Coerce predictions to appropriate format
        ensemble_val_pred = coerce_predictions_for_metric(
            preds=ensemble_val_pred,
            problem_type=problem_type,
            labels=val_labels,
        )
        
        val_preds_per_iteration[iteration] = ensemble_val_pred
        val_labels_per_iteration[iteration] = val_labels
    
    return val_preds_per_iteration, val_labels_per_iteration, skipped_iterations


def get_test_preds_per_iteration(
        df: pd.DataFrame,
        artifacts,
        problem_type: str,
    stats: Optional[dict] = None,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[int]]:
    """
    Get test predictions per iteration by averaging across folds.
    
    Returns:
        (test_preds_per_iteration, test_labels)
    where test_preds_per_iteration maps iteration -> ensemble test predictions (n_samples, n_classes) or (n_samples,)
    and test_labels is the test set labels
    """
    if not isinstance(artifacts, list):
        artifacts = [artifacts]
    
    # Load test labels once
    test_labels = None
    for artifact in artifacts:
        if 'test_labels' in artifact:
            test_labels = _sanitize_non_finite_array(
                artifact['test_labels'],
                stats=stats,
                iteration=None,
            )
            break
    
    if test_labels is None:
        raise KeyError("Missing 'test_labels' in artifacts archive")
    
    test_preds_per_iteration = {}
    skipped_iterations: list[int] = []
    
    for iteration, group_df in df.groupby('iteration'):
        n_folds = len(group_df)
        fold_preds = []
        missing_folds = []
        
        for _, fold_row in group_df.iterrows():
            fold_id = int(fold_row['fold'])
            key = f"iter_{iteration}/fold_{fold_id}/test_preds"
            found_pred = False
            
            for artifact in artifacts:
                if key in artifact:
                    found_pred = True
                    pred = _sanitize_non_finite_array(
                        artifact[key],
                        stats=stats,
                        iteration=int(iteration),
                    )
                    fold_preds.append(pred)
                    break
            
            if not found_pred:
                missing_folds.append((fold_id, key))
        
        if len(fold_preds) < max(1, n_folds - 1):
            skipped_iterations.append(int(iteration))
            if stats is not None:
                stats.setdefault('skipped_iterations', set()).add(int(iteration))
            print(
                f"[post_hoc] WARNING: Skipping iteration {iteration} due to missing "
                f"test predictions ({len(fold_preds)}/{n_folds} folds)."
            )
            continue
        
        # Ensure shape compatibility
        ref_shape = fold_preds[0].shape
        for pred in fold_preds[1:]:
            if pred.shape != ref_shape:
                skipped_iterations.append(int(iteration))
                if stats is not None:
                    stats.setdefault('skipped_iterations', set()).add(int(iteration))
                print(
                    f"[post_hoc] WARNING: Skipping iteration {iteration} due to "
                    f"mismatched fold prediction shapes: expected {ref_shape}, got {pred.shape}."
                )
                fold_preds = []
                break

        if not fold_preds:
            continue
        
        # Average predictions across folds
        ensemble_test_pred = np.mean(fold_preds, axis=0)
        
        # Coerce predictions to appropriate format
        ensemble_test_pred = coerce_predictions_for_metric(
            preds=ensemble_test_pred,
            problem_type=problem_type,
            labels=test_labels,
        )
        
        test_preds_per_iteration[iteration] = ensemble_test_pred
    
    return test_preds_per_iteration, test_labels, skipped_iterations


def get_retrained_test_preds_per_iteration(
        df: pd.DataFrame,
        artifacts,
        problem_type: str,
        test_labels: np.ndarray,
    stats: Optional[dict] = None,
) -> Tuple[Dict[int, np.ndarray], List[int]]:
    """
    Get retrained-model test predictions per iteration from artifacts.npz.
    """
    if not isinstance(artifacts, list):
        artifacts = [artifacts]

    retrain_test_preds_per_iteration = {}
    skipped_iterations: list[int] = []

    for iteration, _ in df.groupby('iteration'):
        key = f"iter_{iteration}/retrain_test_preds"
        found_pred = False

        for artifact in artifacts:
            if key in artifact:
                found_pred = True
                pred = _sanitize_non_finite_array(
                    artifact[key],
                    stats=stats,
                    iteration=int(iteration),
                )

                pred = coerce_predictions_for_metric(
                    preds=pred,
                    problem_type=problem_type,
                    labels=test_labels,
                )
                retrain_test_preds_per_iteration[iteration] = pred
                break

        if not found_pred:
            skipped_iterations.append(int(iteration))
            if stats is not None:
                stats.setdefault('skipped_iterations', set()).add(int(iteration))
            print(f"[post_hoc] WARNING: Skipping iteration {iteration} due to missing '{key}'.")

    return retrain_test_preds_per_iteration, skipped_iterations


def weighted_ensemble_predictions(
        predictions: List[np.ndarray],
        weights: np.ndarray,
        problem_type: str,
        labels: np.ndarray,
) -> np.ndarray:
    """
    Compute weighted arithmetic mean ensemble predictions and coerce to metric format.
    """
    final_pred = np.zeros_like(predictions[0], dtype=np.float64)
    for idx in range(len(predictions)):
        final_pred += weights[idx] * predictions[idx]

    return coerce_predictions_for_metric(
        preds=final_pred,
        problem_type=problem_type,
        labels=labels,
    )


def caruana_greedy_weights(
        candidate_val_preds: List[np.ndarray],
        val_labels: np.ndarray,
        metric: Metric,
        ensemble_size: int,
        problem_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Caruana greedy ensemble selection.
    
    Args:
        candidate_val_preds: List of validation predictions for each candidate
        val_labels: Validation labels
        metric: Metric object for scoring
        ensemble_size: Number of greedy steps
        problem_type: Problem type (binary, multiclass, regression)
    
    Returns:
        (weights, ensemble_val_pred)
    """
    n_candidates = len(candidate_val_preds)
    
    # Track selected candidate indices (with replacement)
    selected_indices = []
    
    # Current validation ensemble prediction; test is not needed for greedy selection.
    current_val_ensemble = None
    
    for step in range(ensemble_size):
        best_idx = None
        best_val_score = None
        best_hypothetical_val_ensemble = None
        
        for candidate_idx in range(n_candidates):
            candidate_val_pred = candidate_val_preds[candidate_idx]
            
            if step == 0:
                # First step: use candidate prediction directly
                hypothetical_val_ensemble = coerce_predictions_for_metric(
                    preds=candidate_val_pred.copy(),
                    problem_type=problem_type,
                    labels=val_labels,
                )
            else:
                # Average current ensemble with candidate
                if isinstance(candidate_val_pred, np.ndarray) and candidate_val_pred.ndim > 1:
                    # Probabilities
                    n_selected = len(selected_indices)
                    hypothetical_val_ensemble = (
                        n_selected * current_val_ensemble + candidate_val_pred
                    ) / (n_selected + 1)
                else:
                    # Regression
                    n_selected = len(selected_indices)
                    hypothetical_val_ensemble = (
                        n_selected * current_val_ensemble + candidate_val_pred
                    ) / (n_selected + 1)
                
                hypothetical_val_ensemble = coerce_predictions_for_metric(
                    preds=hypothetical_val_ensemble.copy(),
                    problem_type=problem_type,
                    labels=val_labels,
                )
            
            # Score on validation
            val_score = metric.score(val_labels, hypothetical_val_ensemble)
            
            # Update best (minimize metric)
            if best_val_score is None or val_score < best_val_score:
                best_val_score = val_score
                best_idx = candidate_idx
                best_hypothetical_val_ensemble = hypothetical_val_ensemble
        
        # Add best candidate to selection
        selected_indices.append(best_idx)
        current_val_ensemble = best_hypothetical_val_ensemble
    
    # Compute final ensemble weights
    weights = np.bincount(selected_indices, minlength=n_candidates)
    weights = weights / weights.sum()

    final_val_ensemble = weighted_ensemble_predictions(
        predictions=candidate_val_preds,
        weights=weights,
        problem_type=problem_type,
        labels=val_labels,
    )

    return weights, final_val_ensemble


def compute_post_hoc_ensemble_trajectories_for_run(
        task_dir: Path,
        problem_type: str,
        ensemble_size: int,
    experiment_name: str,
) -> Dict:
    """
    Compute post-hoc ensemble trajectories for a single HPO run.
    
    Returns a dictionary with per-iteration post-hoc scores for both test protocols.
    """
    task_dir = Path(task_dir)
    
    try:
        # Load task config to verify it's a benchmark experiment
        task_config = load_task_config(task_dir)
        if not is_benchmark_experiment(task_config):
            return {
                'status': 'SKIPPED',
                'reason': 'Not a benchmark experiment',
                'task_dir': str(task_dir)
            }
        
        # Load history and artifacts
        history_path = task_dir / "history.csv"
        if not history_path.exists():
            return {
                'status': 'SKIPPED',
                'reason': 'No history.csv found',
                'task_dir': str(task_dir)
            }
        
        history_df = pd.read_csv(history_path)
        artifact_paths = resolve_artifact_paths(task_dir)
        artifacts = [np.load(path, allow_pickle=True) for path in artifact_paths]

        run_stats = {
            'total_iterations': int(history_df['iteration'].nunique()),
            'total_values_seen': 0,
            'replaced_non_finite_values': 0,
            'iterations_with_replacements': set(),
            'skipped_iterations': set(),
        }
        
        # Get val and test predictions per iteration
        val_preds_per_iter, val_labels_per_iter, _ = get_val_preds_per_iteration(
            history_df,
            artifacts,
            problem_type,
            stats=run_stats,
        )
        test_preds_per_iter, test_labels, _ = get_test_preds_per_iteration(
            history_df,
            artifacts,
            problem_type,
            stats=run_stats,
        )
        retrain_test_preds_per_iter, _ = get_retrained_test_preds_per_iteration(
            history_df,
            artifacts,
            problem_type,
            test_labels,
            stats=run_stats,
        )
        
        # Create metric from problem type mapping used by benchmark experiments.
        actual_metric_name = get_metric_name_from_problem_type(problem_type)
        metric = Metric(metric_name=actual_metric_name, problem_type=problem_type)

        run_name = task_config.get("result_path", task_dir.name)
        
        # Compute trajectories using Caruana ensemble
        iterations = sorted(
            set(val_preds_per_iter.keys())
            & set(val_labels_per_iter.keys())
            & set(test_preds_per_iter.keys())
            & set(retrain_test_preds_per_iter.keys())
        )

        if not iterations:
            return {
                'status': 'SKIPPED',
                'reason': 'No valid iterations after filtering missing/corrupt predictions',
                'task_dir': str(task_dir),
                'run_stats': {
                    'total_iterations': run_stats['total_iterations'],
                    'skipped_iterations': len(run_stats['skipped_iterations']),
                    'iterations_with_replacements': len(run_stats['iterations_with_replacements']),
                    'total_values_seen': run_stats['total_values_seen'],
                    'replaced_non_finite_values': run_stats['replaced_non_finite_values'],
                },
            }

        val_scores = []
        cv_ensembled_test_scores = []
        retrained_test_scores = []
        
        for iteration in tqdm(iterations, desc=f"Ensembling {run_name}/{experiment_name}"):
            # Get all candidates up to and including this iteration
            candidate_indices = [i for i in iterations if i <= iteration]
            candidate_val_preds = [val_preds_per_iter[i] for i in candidate_indices]
            candidate_cv_test_preds = [test_preds_per_iter[i] for i in candidate_indices]
            candidate_retrain_test_preds = [retrain_test_preds_per_iter[i] for i in candidate_indices]
            val_labels = val_labels_per_iter[iteration]

            # Caruana greedy selection on validation only.
            weights, ensemble_val_pred = caruana_greedy_weights(
                candidate_val_preds=candidate_val_preds,
                val_labels=val_labels,
                metric=metric,
                ensemble_size=ensemble_size,
                problem_type=problem_type,
            )

            # Apply learned weights to both test protocols.
            ensemble_cv_test_pred = weighted_ensemble_predictions(
                predictions=candidate_cv_test_preds,
                weights=weights,
                problem_type=problem_type,
                labels=test_labels,
            )
            ensemble_retrain_test_pred = weighted_ensemble_predictions(
                predictions=candidate_retrain_test_preds,
                weights=weights,
                problem_type=problem_type,
                labels=test_labels,
            )

            # Score on validation and both test variants.
            val_score = metric.score(val_labels, ensemble_val_pred)
            cv_test_score = metric.score(test_labels, ensemble_cv_test_pred)
            retrain_test_score = metric.score(test_labels, ensemble_retrain_test_pred)

            val_scores.append(val_score)
            cv_ensembled_test_scores.append(cv_test_score)
            retrained_test_scores.append(retrain_test_score)

            print(f"Iter {iteration} finished in {run_name}/{experiment_name}")
        
        return {
            'status': 'SUCCESS',
            'task_dir': str(task_dir),
            'iterations': iterations,
            'val_scores': val_scores,
            'cv_ensembled_test_scores': cv_ensembled_test_scores,
            'retrained_test_scores': retrained_test_scores,
            'n_iterations': len(iterations),
            'run_stats': {
                'total_iterations': run_stats['total_iterations'],
                'skipped_iterations': len(run_stats['skipped_iterations']),
                'iterations_with_replacements': len(run_stats['iterations_with_replacements']),
                'total_values_seen': run_stats['total_values_seen'],
                'replaced_non_finite_values': run_stats['replaced_non_finite_values'],
            },
        }
    
    except Exception as e:
        return {
            'status': 'FAILED',
            'task_dir': str(task_dir),
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def preprocess_post_hoc_ensemble_experiment(
        results_dir: str,
    ensemble_size: int = 40,
        n_jobs: int = 1,
        verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Preprocess all benchmark experiments in a results directory using Caruana post-hoc ensembling.
    
    Args:
        results_dir: Path to experiment results directory
        ensemble_size: Number of greedy ensemble selection steps
        n_jobs: Number of parallel jobs
        verbose: Whether to print progress
    
    Returns:
        DataFrame with full post-hoc ensemble trajectories matching preprocessing.py schema.
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory does not exist: {results_dir}")
        return None
    
    # Find all task directories (one level deep)
    task_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    if not task_dirs:
        print(f"No task directories found in {results_dir}")
        return None
    
    if verbose:
        print(f"Found {len(task_dirs)} task directories")
    
    # Determine problem type from first valid task config
    problem_type = None
    for task_dir in task_dirs:
        try:
            config = load_task_config(task_dir)
            problem_type = config.get('problem_type')
            if problem_type:
                break
        except:
            continue
    
    if not problem_type:
        print("Could not determine problem type from task configs")
        return None
    
    if verbose:
        print(f"Problem type: {problem_type}")
        print(f"Using metric mapping: {problem_type} -> {get_metric_name_from_problem_type(problem_type)}")
    
    # Compute trajectories in parallel
    if verbose:
        print(f"Computing post-hoc ensemble trajectories with ensemble_size={ensemble_size}")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_post_hoc_ensemble_trajectories_for_run)(
            task_dir, problem_type, ensemble_size, results_path.name
        )
        for task_dir in tqdm(task_dirs, disable=not verbose)
    )
    
    # Process results
    raw_rows = []
    success_count = 0
    skip_count = 0
    fail_count = 0
    total_runs_considered = 0
    runs_with_any_fallback = 0
    total_iterations_seen = 0
    total_skipped_iterations = 0
    total_iterations_with_replacements = 0
    total_values_seen = 0
    total_replaced_non_finite_values = 0
    
    for result in results:
        run_stats = result.get('run_stats') if isinstance(result, dict) else None
        if run_stats is not None:
            total_runs_considered += 1
            total_iterations_seen += int(run_stats.get('total_iterations', 0))
            total_skipped_iterations += int(run_stats.get('skipped_iterations', 0))
            total_iterations_with_replacements += int(run_stats.get('iterations_with_replacements', 0))
            total_values_seen += int(run_stats.get('total_values_seen', 0))
            total_replaced_non_finite_values += int(run_stats.get('replaced_non_finite_values', 0))
            if (
                int(run_stats.get('skipped_iterations', 0)) > 0
                or int(run_stats.get('replaced_non_finite_values', 0)) > 0
            ):
                runs_with_any_fallback += 1

        if result['status'] == 'SUCCESS':
            success_count += 1
            task_dir = Path(result['task_dir'])
            
            # Load metadata from task config
            config = load_task_config(task_dir)
            metadata = extract_metadata_from_config(config)
            metadata['experiment_source'] = results_path.name
            metadata['raw_experiment_id'] = task_dir.name
            metadata['experiment_id'] = f"{results_path.name}::{task_dir.name}"

            # Create raw rows (one per iteration), then run the same normalization+trajectory logic
            # used by preprocessing.py for consistent derived columns.
            for i, iteration in enumerate(result['iterations']):
                row = {
                    'iteration': iteration,
                    'avg_val_score': result['val_scores'][i],
                    'retrained_test_score': result['retrained_test_scores'][i],
                    'cv_ensembled_test_score': result['cv_ensembled_test_scores'][i],
                }
                row.update(metadata)
                raw_rows.append(row)
        
        elif result['status'] == 'SKIPPED':
            skip_count += 1
            if verbose:
                print(f"Skipped {result['task_dir']}: {result['reason']}")
        
        else:  # FAILED
            fail_count += 1
            if verbose:
                print(f"Failed {result['task_dir']}: {result['error']}")
    
    if verbose:
        print(f"\nResults: {success_count} success, {skip_count} skipped, {fail_count} failed")
        affected_iterations = total_skipped_iterations + total_iterations_with_replacements
        affected_iteration_pct = (
            100.0 * affected_iterations / total_iterations_seen
            if total_iterations_seen > 0 else 0.0
        )
        affected_run_pct = (
            100.0 * runs_with_any_fallback / total_runs_considered
            if total_runs_considered > 0 else 0.0
        )
        replaced_value_pct = (
            100.0 * total_replaced_non_finite_values / total_values_seen
            if total_values_seen > 0 else 0.0
        )

        print("[post_hoc] Fallback summary (missing/non-finite handling):")
        print(
            f"  - Runs with fallback: {runs_with_any_fallback}/{total_runs_considered} "
            f"({affected_run_pct:.4f}%)"
        )
        print(
            f"  - Iterations skipped due to missing/invalid predictions: "
            f"{total_skipped_iterations}/{total_iterations_seen} "
            f"({(100.0 * total_skipped_iterations / total_iterations_seen) if total_iterations_seen > 0 else 0.0:.4f}%)"
        )
        print(
            f"  - Iterations with non-finite replacements: "
            f"{total_iterations_with_replacements}/{total_iterations_seen} "
            f"({(100.0 * total_iterations_with_replacements / total_iterations_seen) if total_iterations_seen > 0 else 0.0:.4f}%)"
        )
        print(
            f"  - Affected iterations total (skip + replacement): {affected_iterations}/{total_iterations_seen} "
            f"({affected_iteration_pct:.4f}%)"
        )
        print(
            f"  - Non-finite prediction/label values replaced with 0.0: "
            f"{total_replaced_non_finite_values}/{total_values_seen} "
            f"({replaced_value_pct:.6f}%)"
        )
    
    if not raw_rows:
        print("No trajectory rows generated")
        return None

    # Build per-iteration raw scores dataframe and apply the same post-processing as preprocessing.py
    results_df = pd.DataFrame(raw_rows)
    results_df = convert_negative_metric_to_loss(results_df)

    metadata_cols = [c for c in [
        'experiment_id', 'experiment_source', 'raw_experiment_id',
        'dataset_id', 'model', 'optimizer', 'metric', 'problem_type',
        'mitigation', 'reshuffling', 'inner_split', 'selection_set_size',
        'repetition', 'outer_fold', 'random_state', 'outer_n_folds', 'outer_n_repeats',
    ] if c in results_df.columns]

    trajectory_rows = []
    for exp_id, run_df in tqdm(results_df.groupby('experiment_id'), desc="Calculating post-hoc trajectories", disable=not verbose):
        run_df = run_df.sort_values('iteration')

        val_scores = run_df['avg_val_score'].tolist()
        retrained_test_scores = run_df['retrained_test_score'].tolist()
        ensembled_test_scores = run_df['cv_ensembled_test_score'].tolist()

        traj = calculate_trajectories(
            val_scores=val_scores,
            retrained_test_scores=retrained_test_scores,
            ensembled_test_scores=ensembled_test_scores,
            selection_scores=None,
        )

        meta = {col: run_df[col].iloc[0] for col in metadata_cols}
        iterations = run_df['iteration'].tolist()

        for i, iteration in enumerate(iterations):
            row = {'iteration': iteration}
            row.update(meta)
            for key, values in traj.items():
                if key == 'scores_to_attach':
                    continue
                row[key] = values[i]
            trajectory_rows.append(row)

    trajectories_df = pd.DataFrame(trajectory_rows)

    required_columns = [
        'iteration', 'experiment_id', 'experiment_source', 'raw_experiment_id',
        'dataset_id', 'model', 'optimizer', 'metric', 'problem_type',
        'mitigation', 'reshuffling', 'inner_split', 'selection_set_size',
        'repetition', 'outer_fold', 'random_state', 'outer_n_folds', 'outer_n_repeats',
        'val_performance', 'retrain_test_loss',
        'ensembled_test_loss',
        'retrain_meta_overfitting', 'ensemble_meta_overfitting',
        'retrain_regret', 'ensemble_regret',
        'retrain_overtuning', 'ensemble_overtuning',
        'relative_retrain_overtuning', 'relative_ensemble_overtuning',
    ]

    missing_required = [c for c in required_columns if c not in trajectories_df.columns]
    if missing_required:
        raise ValueError(f"Missing required trajectory columns: {missing_required}")

    trajectories_df = trajectories_df[required_columns]

    # Save to CSV
    results_output_path = results_path / "results_post_hoc_ensemble.csv"
    results_df.to_csv(results_output_path, index=False)

    output_path = results_path / "trajectories_post_hoc_ensemble.csv"
    trajectories_df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"Saved post-hoc raw scores to {results_output_path}")
        print(f"Saved post-hoc ensemble trajectories to {output_path}")
        print(f"Shape: {trajectories_df.shape}")
    
    return trajectories_df
