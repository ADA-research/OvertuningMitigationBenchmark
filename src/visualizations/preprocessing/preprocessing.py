import os
import gzip
import io
import time
import yaml
import warnings
from pathlib import Path
from contextlib import ExitStack, contextmanager
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Callable, List, TextIO
from src.metrics.metric import Metric
from src.visualizations.preprocessing.shared_utils import (
    RESULTS_FILE,
    TRAJECTORIES_FILE,
    coerce_predictions_for_metric,
    resolve_artifact_paths,
)
from src.visualizations.preprocessing.combine_normalization import (
    apply_normalization,
    build_anchor_ranges,
)


# Keys are the `method` values in the combined trajectories file (plotting's trajectory_source).
TRAJECTORY_FILES_BY_METHOD = {
    "default": "trajectories.csv",
    "one_se": "trajectories_one_se_selection.csv",
    "makarova": "trajectories_makarova.csv",
    "mlplan": "trajectories_mlplan.csv",
    "post_hoc_ensemble": "trajectories_post_hoc_ensemble.csv",
    "post_hoc_surrogate": "trajectories_post_hoc_surrogate.csv",
}


# sklearn can emit this for imperfect probability vectors in historical artifacts.
# We intentionally suppress it to keep preprocessing logs focused on actionable issues.
warnings.filterwarnings(
    "ignore",
    message=r".*y_pred.*sum to one.*",
    category=UserWarning,
)


def _count_unique_values_in_csv_column(
    csv_path: Path,
    column: str,
    chunk_size: int = 200_000,
) -> int:
    """Count unique non-null values in a CSV column using chunked reads."""
    if not csv_path.exists():
        return 0

    unique_values = set()
    try:
        iterator = pd.read_csv(
            csv_path,
            usecols=[column],
            chunksize=chunk_size,
            low_memory=False,
        )
    except ValueError:
        return 0

    for chunk in iterator:
        values = chunk[column].dropna().astype(str).unique().tolist()
        unique_values.update(values)

    return len(unique_values)


def _dataset_unique_run_counts_by_model(
    csv_path: Path,
    chunk_size: int = 200_000,
) -> tuple[dict[str, dict[str, int]], str | None]:
    """Return model -> (dataset -> unique run count) for a trajectory CSV."""
    if not csv_path.exists():
        return {}, None

    header = pd.read_csv(csv_path, nrows=0)
    if "dataset_id" not in header.columns:
        return {}, None
    if "model" not in header.columns:
        return {}, None

    run_column = None
    if "raw_experiment_id" in header.columns:
        run_column = "raw_experiment_id"
    elif "experiment_id" in header.columns:
        run_column = "experiment_id"
    else:
        return {}, None

    runs_by_model_dataset: dict[str, dict[str, set[str]]] = {}
    with csv_path.open("rb") as source, pd.read_csv(
        source,
        compression="gzip" if csv_path.suffix == ".gz" else None,
        usecols=["model", "dataset_id", run_column],
        chunksize=chunk_size,
        low_memory=False,
    ) as reader:
        for chunk in reader:
            chunk = chunk.dropna(subset=["model", "dataset_id", run_column])
            for (model, dataset_id), runs in chunk.groupby(["model", "dataset_id"])[run_column]:
                model_key = str(model)
                dataset_key = str(dataset_id)
                if model_key not in runs_by_model_dataset:
                    runs_by_model_dataset[model_key] = {}
                if dataset_key not in runs_by_model_dataset[model_key]:
                    runs_by_model_dataset[model_key][dataset_key] = set()
                runs_by_model_dataset[model_key][dataset_key].update(runs.astype(str).unique().tolist())
            _discard_file_cache(source)

    counts_by_model: dict[str, dict[str, int]] = {}
    for model_key in sorted(runs_by_model_dataset.keys()):
        dataset_map = runs_by_model_dataset[model_key]
        sorted_dataset_ids = sorted(dataset_map.keys(), key=lambda x: (not x.isdigit(), x))
        counts_by_model[model_key] = {
            dataset_id: len(dataset_map[dataset_id])
            for dataset_id in sorted_dataset_ids
        }

    return counts_by_model, run_column


def _log_per_experiment_trajectory_run_counts(results_dir: str, verbose: bool) -> None:
    """Log unique run counts for each per-experiment trajectory file."""
    if not verbose:
        return

    results_path = Path(results_dir)
    trajectory_files = {
        "trajectories": results_path / "trajectories.csv",
        "surrogate": results_path / "trajectories_surrogate_selection.csv",
        "one_se": results_path / "trajectories_one_se_selection.csv",
        "makarova": results_path / "trajectories_makarova.csv",
    }

    print("[preprocessing] Unique runs per trajectory file for this experiment directory:")
    for label, csv_path in trajectory_files.items():
        count = _count_unique_values_in_csv_column(csv_path, column="raw_experiment_id")
        run_column = "raw_experiment_id"
        if count == 0:
            count = _count_unique_values_in_csv_column(csv_path, column="experiment_id")
            run_column = "experiment_id"
        print(
            f"  - {label}: {count} unique runs "
            f"({run_column}) in {csv_path.name}"
        )


def _log_combined_trajectory_dataset_run_counts(output_dir: Path, verbose: bool) -> None:
    """Log model -> dataset -> unique run counts for each combined trajectory file."""
    if not verbose:
        return

    trajectory_files = [output_dir / TRAJECTORIES_FILE]

    print("[preprocessing] Model-specific dataset -> unique run counts per combined trajectory file:")
    for csv_path in trajectory_files:
        if not csv_path.exists():
            continue
        counts_by_model, run_column = _dataset_unique_run_counts_by_model(csv_path)
        if run_column is None:
            print(f"  - {csv_path.name}: skipped (missing dataset/run columns)")
            continue
        if not counts_by_model:
            print(f"  - {csv_path.name} ({run_column}): no rows")
            continue
        print(f"  - {csv_path.name} ({run_column}):")
        for model_key, dataset_counts in counts_by_model.items():
            print(f"    * {model_key}: {dataset_counts}")


def _align_score_columns(df: pd.DataFrame, table_kind: str) -> pd.DataFrame:
    """Ensure score columns use a consistent naming scheme across tables.

    Adds alias columns bidirectionally to keep compatibility with existing code.
    """
    out = df.copy()

    if table_kind == "results":
        alias_pairs = [
            ("avg_val_score", "val_loss"),
            ("retrained_test_score", "retrain_test_loss"),
            ("cv_ensembled_test_score", "ensembled_test_loss"),
            ("normalized_avg_val_score", "normalized_val_loss"),
            ("normalized_retrained_test_score", "normalized_retrain_test_loss"),
            ("normalized_cv_ensembled_test_score", "normalized_ensembled_test_loss"),
        ]
    elif table_kind in {
        "default",
        "surrogate",
        "one_se",
        "makarova",
        "mlplan",
        "post_hoc_ensemble",
        "post_hoc_surrogate",
    }:
        alias_pairs = [
            ("val_performance", "val_loss"),
        ]
    else:
        alias_pairs = []

    for left, right in alias_pairs:
        if left in out.columns and right not in out.columns:
            out[right] = out[left]
        if right in out.columns and left not in out.columns:
            out[left] = out[right]

    return out


def _discard_file_cache(handle) -> None:
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def append_csv_file(
    source_path: Path,
    handle: TextIO,
    wrote_header: bool,
    expected_columns: List[str] | None = None,
    table_kind: str = "",
    chunk_size: int = 10_000,
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[bool, List[str] | None]:
    """Append a CSV file to an open output handle with schema alignment by column name.

    This prevents silent column corruption when per-experiment files have the
    same columns in different order. *transform* is applied to every chunk
    before the schema alignment.
    """
    if not source_path.exists():
        return wrote_header, expected_columns

    with source_path.open("rb") as source, pd.read_csv(
        source,
        compression="gzip" if source_path.suffix == ".gz" else None,
        chunksize=chunk_size,
        low_memory=False,
    ) as reader:
        for chunk in reader:
            wrote_header, expected_columns = _append_csv_chunk(
                chunk, source_path, handle, wrote_header, expected_columns,
                table_kind, transform,
            )
            if progress is not None:
                progress(len(chunk))
            _discard_file_cache(source)

    return wrote_header, expected_columns


def _append_csv_chunk(
    chunk, source_path, handle, wrote_header, expected_columns, table_kind, transform,
):
    chunk = _align_score_columns(chunk, table_kind=table_kind)
    if transform is not None:
        chunk = transform(chunk)

    if expected_columns is None:
        expected_columns = list(chunk.columns)
    else:
        missing = [col for col in expected_columns if col not in chunk.columns]
        extra = [col for col in chunk.columns if col not in expected_columns]
        for col in missing:
            chunk[col] = np.nan
        if extra:
            raise ValueError(
                f"Schema mismatch while combining {source_path}. "
                f"Unexpected columns: {extra}. Expected columns: {expected_columns}"
            )
        chunk = chunk.reindex(columns=expected_columns)

    chunk.to_csv(handle, header=not wrote_header, index=False)
    handle.flush()
    raw_output = getattr(handle, "cache_file", None)
    if raw_output is not None:
        current_size = raw_output.tell()
        if current_size - handle.cache_flushed_size >= 64 * 1024 * 1024:
            os.fdatasync(raw_output.fileno())
            _discard_file_cache(raw_output)
            handle.cache_flushed_size = current_size

    return True, expected_columns


@contextmanager
def _open_csv_gz(path: Path):
    """Open a gzip text handle; *path* is only replaced once writing succeeded."""
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("wb") as raw_output:
            with gzip.GzipFile(fileobj=raw_output, mode="wb", compresslevel=6) as compressed:
                with io.TextIOWrapper(compressed, newline="", encoding="utf-8") as handle:
                    handle.cache_file = raw_output
                    handle.cache_flushed_size = 0
                    yield handle
            raw_output.flush()
            os.fdatasync(raw_output.fileno())
            _discard_file_cache(raw_output)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _flush_rows_to_csv(rows: list[dict], output_path: Path, wrote_header: bool) -> bool:
    """Flush buffered row dicts to CSV and clear buffer."""
    if not rows:
        return wrote_header

    pd.DataFrame(rows).to_csv(
        output_path,
        mode="a" if wrote_header else "w",
        header=not wrote_header,
        index=False,
    )
    rows.clear()
    return True


def compute_incumbent_indices_from_selection_scores(
    selection_scores: List[float],
    fold_counts: List[int] | None = None,
) -> List[int]:
    # Only fully-evaluated configs may become incumbent.
    # Use first iteration's fold count as reference for "fully evaluated".
    expected_folds = fold_counts[0] if fold_counts else None
    incumbent_iteration = 0
    incumbent_indices = []

    for i in range(len(selection_scores)):
        is_fully_evaluated = (expected_folds is None or fold_counts[i] >= expected_folds)
        if is_fully_evaluated and selection_scores[i] < selection_scores[incumbent_iteration]:
            incumbent_iteration = i
        incumbent_indices.append(incumbent_iteration)

    return incumbent_indices


def compute_surrogate_selection_scores(surrogate_scores: List[float]) -> List[float]:
    # During warmup surrogate may be NaN; keep first config as incumbent by making warmup scores non-competitive.
    return [float("inf") if pd.isna(score) else score for score in surrogate_scores]


def compute_one_se_incumbent_indices(
        val_scores: List[float],
        val_standard_errors: List[float],
) -> List[int]:
    incumbent_indices = []

    for t in range(len(val_scores)):
        prefix_val_scores = val_scores[:t + 1]
        best_val = min(prefix_val_scores)
        best_idx = prefix_val_scores.index(best_val)
        threshold = val_scores[best_idx] + val_standard_errors[best_idx]

        candidate_indices = [
            idx for idx in range(t + 1)
            if val_scores[idx] <= threshold
        ]
        if not candidate_indices:
            raise ValueError(
                f"No 1-SE incumbent candidate found at step {t}. "
                f"best_val={best_val}, best_idx={best_idx}, threshold={threshold}"
            )
        incumbent_idx = candidate_indices[0]
        incumbent_indices.append(incumbent_idx)

    return incumbent_indices


def compute_makarova_incumbent_indices(
        val_scores: List[float],
        early_stopped_makarova: List[bool],
) -> List[int]:
    """
    Compute incumbent indices using Makarova strategy.
    
    Selects the config with the lowest val score at each iteration,
    until early_stopped_makarova becomes True. Once stopped, keeps
    the same incumbent for the remaining iterations.
    
    Args:
        val_scores: Validation scores per iteration
        early_stopped_makarova: Boolean flag per iteration indicating when to stop
        
    Returns:
        List of incumbent iteration indices, one per iteration
    """
    incumbent_indices = []
    incumbent_iteration = 0
    stopped = False
    
    for i in range(len(val_scores)):
        # Once stopped, keep the same incumbent
        if stopped:
            incumbent_indices.append(incumbent_iteration)
            continue
        
        # Check if we should stop at this iteration
        if pd.notna(early_stopped_makarova[i]) and early_stopped_makarova[i]:
            stopped = True
            incumbent_indices.append(incumbent_iteration)
            continue
        
        # Select the lowest val score seen so far
        if val_scores[i] < val_scores[incumbent_iteration]:
            incumbent_iteration = i
        incumbent_indices.append(incumbent_iteration)
    
    return incumbent_indices


def extract_metadata_from_config(config: dict) -> dict:
    """
    Extract relevant metadata from task_config.yaml

    Returns a dictionary with the following keys:
    - repetition: outer repetition number
    - outer_fold: outer fold number
    - optimizer: optimizer name
    - metric: evaluation metric
    - mitigation: mitigation strategy (mlplan, racing, thresholdout, or empty)
    - reshuffling: True/False/None (None for mitigations)
    - inner_split: cv/holdout/None (None for mitigations)
    - selection_set_size: float/None (None for mitigations)
    """
    metadata = {}

    # Basic experiment settings
    metadata['dataset_id'] = config.get('dataset_id')
    metadata['repetition'] = config.get('outer_evaluation').get('repeat')
    metadata['outer_fold'] = config.get('outer_evaluation').get('fold')
    metadata['optimizer'] = config.get('optimizer')
    metadata['metric'] = config.get('metric')
    metadata['problem_type'] = config.get('problem_type')

    # For now, we assume one model per problem type
    # In the future, add 'CASH' as option?
    if metadata['problem_type'] in ["binary", "multiclass"]:
        metadata['model'] = config.get('search_space').get('classifiers')[0]
    else:
        metadata['model'] = config.get('search_space').get('regressors')[0]

    # Mitigation strategy
    mitigation_strategy = config.get('mitigation_strategy')
    racing_strategy = config.get('racing_strategy')

    # Determine mitigation type
    if mitigation_strategy == 'mlplan':
        metadata['mitigation'] = 'mlplan'
        metadata['reshuffling'] = None
        metadata['inner_split'] = None
        metadata['selection_set_size'] = None

    elif mitigation_strategy == 'thresholdout':
        metadata['mitigation'] = 'thresholdout'
        metadata['reshuffling'] = config.get("evaluation").get("reshuffle")
        metadata['inner_split'] = None
        metadata['selection_set_size'] = None

    elif racing_strategy != 'none':
        metadata['mitigation'] = racing_strategy
        metadata['reshuffling'] = config.get("evaluation").get("reshuffle")
        metadata['inner_split'] = None
        metadata['selection_set_size'] = None

    else:
        # Standard method (no mitigation)
        metadata['mitigation'] = None

        # Reshuffling
        metadata['reshuffling'] = config.get('evaluation').get('reshuffle')

        # Inner split
        metadata['inner_split'] = config.get('evaluation').get('resampling')

        # Selection set size
        metadata['selection_set_size'] = config.get('evaluation').get('selection_size')

    # Additional useful metadata (optional, can be commented out if not needed)
    metadata['problem_type'] = config.get('problem_type')
    metadata['random_state'] = config.get('random_state')
    metadata['outer_n_folds'] = config.get('outer_evaluation').get('n_folds')
    metadata['outer_n_repeats'] = config.get('outer_evaluation').get('n_repeats')

    return metadata


def convert_negative_metric_to_loss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert negated maximization metrics to loss scale by adding +1.

        Important:
        - Only ROC-AUC scores should be shifted here.
        - The other metrics used in this preprocessing pipeline are already losses.

    Example conversion:
        -1.0 (best) -> 0.0
         0.0 (worst) -> 1.0
    """
    df = df.copy()

    score_columns = [
        "train",
        "val",
        "test",
        "selection",
        "avg_val_score",
        "avg_test_score",
        "retrained_test_score",
        "cv_ensembled_test_score",
        "avg_train_score",
        "avg_selection_score",
        "surrogate_mean",
    ]

    convert_mask = df["metric"].astype(str).eq("roc_auc")

    for col in score_columns:
        if col not in df.columns:
            continue

        df.loc[convert_mask, col] = df.loc[convert_mask, col] + 1.0

    return df


def get_ensembled_test_scores_per_iteration(df: pd.DataFrame, artifacts, problem_type: str, metric_name: str):
    """
    Get ensembled test scores per iteration by averaging predictions across folds.
    
    For each iteration:
    - If single fold: return test score from dataframe
    - If multiple folds: ensemble predictions across folds and calculate metric
    
    Args:
        df: Per-fold dataframe for a single HPO run (one row per fold per iteration)
        artifacts: npz dict-like object with predictions and labels
        problem_type: problem type string (binary, multiclass, regression)
        metric_name: sklearn metric name string
        
    Returns:
        Dictionary mapping iteration -> ensembled test score
    """
    if problem_type == "multiclass":
        metric_name = "neg_log_loss"

    metric = Metric(metric_name=metric_name, problem_type=problem_type)

    if not isinstance(artifacts, list):
        artifacts = [artifacts]

    # Load test labels once
    test_labels = None
    for artifact in artifacts:
        if 'test_labels' in artifact:
            test_labels = artifact['test_labels']
            break
    if test_labels is None:
        raise KeyError("Missing 'test_labels' in artifacts archive")
    if not np.isfinite(test_labels).all():
        raise ValueError("Non-finite values found in test_labels")
    
    scores_per_iteration = {}
    bad_iteration_count = 0
    
    for iteration, group_df in df.groupby('iteration'):
        n_folds = len(group_df)
        
        if n_folds == 1:
            # Single fold: use test score from dataframe
            scores_per_iteration[iteration] = group_df['test'].iloc[0]
        else:
            # Multiple folds: ensemble predictions from artifacts
            fold_preds = []
            missing_folds = []
            non_finite_folds = []
            
            for _, fold_row in group_df.iterrows():
                fold_id = int(fold_row['fold'])
                key = f"iter_{iteration}/fold_{fold_id}/test_preds"
                found_pred = False
                for artifact in artifacts:
                    if key in artifact:
                        found_pred = True
                        pred = np.asarray(artifact[key])
                        if not np.isfinite(pred).all():
                            non_finite_folds.append((fold_id, key))
                            break
                        fold_preds.append(pred)
                        break
                if not found_pred:
                    missing_folds.append((fold_id, key))

            min_required_folds = min(3, n_folds)
            if len(fold_preds) < min_required_folds:
                # Assign a pessimistic finite fallback score so preprocessing can continue.
                # The framework minimizes metrics, so a larger value is always worse.
                finite_test_scores = pd.to_numeric(group_df['test'], errors='coerce').to_numpy()
                finite_test_scores = finite_test_scores[np.isfinite(finite_test_scores)]
                if finite_test_scores.size > 0:
                    baseline_worst = float(np.max(finite_test_scores))
                    fallback_score = baseline_worst + max(abs(baseline_worst), 1.0)
                else:
                    fallback_score = 1e12

                bad_iteration_count += 1
                missing_desc = ", ".join([f"fold={fid}" for fid, _ in missing_folds])
                non_finite_desc = ", ".join([f"fold={fid}" for fid, _ in non_finite_folds])
                print(
                    "[preprocessing] WARNING: insufficient valid fold predictions; "
                    f"assigned fallback score at iteration={iteration}. "
                    f"valid_folds={len(fold_preds)}/{n_folds}, required>={min_required_folds}, "
                    f"fallback_score={fallback_score}, missing_folds=[{missing_desc}], "
                    f"non_finite_folds=[{non_finite_desc}]"
                )
                scores_per_iteration[iteration] = fallback_score
                continue
            
            # Ensure all valid fold predictions are shape-compatible for averaging.
            ref_shape = fold_preds[0].shape
            for pred in fold_preds[1:]:
                if pred.shape != ref_shape:
                    raise ValueError(
                        "Mismatched fold prediction shapes for ensembling: "
                        f"iteration={iteration}, expected_shape={ref_shape}, got_shape={pred.shape}"
                    )

            # Ensemble by averaging predictions across valid folds.
            ensembled_pred = np.mean(fold_preds, axis=0)

            if not np.isfinite(ensembled_pred).all():
                raise ValueError(f"Non-finite ensembled predictions for iteration={iteration}")

            ensembled_pred = coerce_predictions_for_metric(
                preds=ensembled_pred,
                problem_type=problem_type,
                labels=test_labels,
            )

            if not np.isfinite(ensembled_pred).all():
                raise ValueError(f"Non-finite coerced ensembled predictions for iteration={iteration}")

            # Calculate metric on ensembled predictions
            ensembled_score = metric.score(test_labels, ensembled_pred)
            
            scores_per_iteration[iteration] = ensembled_score
    
    if bad_iteration_count > 0:
        print(
            "[preprocessing] WARNING SUMMARY: assigned fallback ensemble scores "
            f"for {bad_iteration_count} iteration(s) in this run."
        )

    return scores_per_iteration


def load_experiment(
        results_dir: str,
        verbose: bool = True
) -> pd.DataFrame:
    """
    Load all experiment results from a directory and concatenate them with metadata.

    Args:
        results_dir: Path to directory containing experiment result folders
        output_path: Path where the concatenated CSV should be saved
        verbose: Whether to print progress information

    Returns:
        Concatenated DataFrame with all results and metadata
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_dataframes = []

    # Find all history.csv files at the experiment level only (not in subdirectories)
    # This avoids searching through iteration/fold subdirectories
    history_files = []

    if verbose:
        print(f"Scanning for experiment directories in {results_dir}...")

    # Only look one level deep - each experiment folder should contain history.csv
    for item in results_path.iterdir():
        if item.is_dir():
            history_csv = item / 'history.csv'
            if history_csv.exists():
                history_files.append(history_csv)

    if not history_files:
        raise ValueError(f"No history.csv files found in {results_dir}")

    if verbose:
        print(f"Found {len(history_files)} experiment result folders")
        iterator = tqdm(history_files, desc="Processing experiments")
    else:
        iterator = history_files

    for history_file in iterator:
        experiment_dir = history_file.parent
        experiment_name = experiment_dir.name
        task_config_path = experiment_dir / 'task_config.yaml'
        predictions_paths = resolve_artifact_paths(experiment_dir)

        # Load task config
        with open(task_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract metadata
        metadata = extract_metadata_from_config(config)

        # This is not required in future versions
        if metadata["outer_fold"] is None:
            fold = int(experiment_name.split('_')[3][-1])
            repeat = int(experiment_name.split('_')[2][-1])
            metadata["outer_fold"] = fold
            metadata["repetition"] = repeat

        # Load history CSV
        df = pd.read_csv(history_file)

        with ExitStack() as stack:
            artifacts = [stack.enter_context(np.load(path)) for path in predictions_paths]
            cv_ensembled_test_scores = get_ensembled_test_scores_per_iteration(
                df,
                artifacts,
                problem_type=metadata['problem_type'],
                metric_name=metadata['metric'],
            )

        df["cv_ensembled_test_score"] = df['iteration'].map(cv_ensembled_test_scores)

        df.rename(columns={"avg_test_score": "retrained_test_score"}, inplace=True)

        # Add metadata columns
        for key, value in metadata.items():
            df[key] = value

        # Add experiment identifiers.
        # `experiment_id` must be globally unique across all result folders.
        # The plain experiment_name (e.g. "363621_rep0_fold0_smac_...") can
        # collide between model runs (e.g. LGBM vs RealMLP), so we prefix it.
        df['experiment_source'] = results_path.name
        df['raw_experiment_id'] = experiment_name
        df['experiment_id'] = f"{results_path.name}::{experiment_name}"

        all_dataframes.append(df)

    if not all_dataframes:
        raise ValueError("No valid experiment results were loaded")

    # Concatenate all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    combined_df['early_stopped_makarova'] = [bool(x) if not pd.isna(x) else None for x in
                                             combined_df['early_stopped_makarova']]

    if verbose:
        print(f"\n✓ Successfully loaded {len(all_dataframes)} experiments")
        print(f"\nCombined dataset shape: {combined_df.shape}")
        print(f"Total records: {len(combined_df):,}")
        print(f"\nColumns: {list(combined_df.columns)}")
        print(f"\nDataset breakdown:")
        print(f"  - Datasets: {combined_df['dataset_id'].nunique()}")
        print(f"  - Optimizers: {combined_df['optimizer'].unique().tolist()}")
        print(f"  - Metrics: {combined_df['metric'].unique().tolist()}")
        print(f"  - Mitigations: {combined_df['mitigation'].unique().tolist()}")
        print(f"  - Inner splits: {combined_df['inner_split'].unique().tolist()}")

    return combined_df



def calculate_trajectories(
        val_scores: List[float],
        retrained_test_scores: List[float],
        ensembled_test_scores: List[float],
        selection_scores: List[float] = None,
        incumbent_indices: List[int] = None,
        scores_to_attach: dict = None,
        fold_counts: List[int] | None = None,
) -> dict:
    # Incumbent selection state
    incumbent_iteration = 0

    # Best test-of-incumbent trackers (for overtuning: best test score among all past incumbents)
    best_retrain_test_incumbent_iteration = 0
    best_ensemble_test_incumbent_iteration = 0

    # Best test overall trackers (for regret)
    best_retrain_test_iteration = 0
    best_ensemble_test_iteration = 0

    if scores_to_attach is None:
        scores_to_attach = {}
    scores_to_attach_dict = {k: [] for k, v in scores_to_attach.items() if v is not None}

    result = {
        'val_performance': [],
        'retrain_test_loss': [],
        'ensembled_test_loss': [],
        'retrain_meta_overfitting': [],
        'ensemble_meta_overfitting': [],
        'retrain_regret': [],
        'ensemble_regret': [],
        'retrain_overtuning': [],
        'ensemble_overtuning': [],
        'relative_retrain_overtuning': [],
        'relative_ensemble_overtuning': [],
    }

    if incumbent_indices is None:
        scores_to_select_incumbent_on = selection_scores if selection_scores is not None else val_scores
        incumbent_indices = compute_incumbent_indices_from_selection_scores(
            scores_to_select_incumbent_on, fold_counts=fold_counts,
        )

    for i in range(len(incumbent_indices)):
        previous_incumbent_iteration = incumbent_iteration
        incumbent_iteration = incumbent_indices[i]

        # --- Incumbent selection ---
        if incumbent_iteration != previous_incumbent_iteration:
            # Update best test-of-incumbent when incumbent changes
            if retrained_test_scores[incumbent_iteration] < retrained_test_scores[best_retrain_test_incumbent_iteration]:
                best_retrain_test_incumbent_iteration = incumbent_iteration
            if ensembled_test_scores[incumbent_iteration] < ensembled_test_scores[best_ensemble_test_incumbent_iteration]:
                best_ensemble_test_incumbent_iteration = incumbent_iteration

        # Update best test overall (for regret)
        if retrained_test_scores[i] < retrained_test_scores[best_retrain_test_iteration]:
            best_retrain_test_iteration = i
        if ensembled_test_scores[i] < ensembled_test_scores[best_ensemble_test_iteration]:
            best_ensemble_test_iteration = i

        # --- Incumbent performance ---
        result['val_performance'].append(val_scores[incumbent_iteration])
        result['retrain_test_loss'].append(retrained_test_scores[incumbent_iteration])
        result['ensembled_test_loss'].append(ensembled_test_scores[incumbent_iteration])

        # --- Meta-overfitting ---
        result['retrain_meta_overfitting'].append(retrained_test_scores[incumbent_iteration] - val_scores[incumbent_iteration])
        result['ensemble_meta_overfitting'].append(ensembled_test_scores[incumbent_iteration] - val_scores[incumbent_iteration])

        # --- Regret ---
        result['retrain_regret'].append(retrained_test_scores[incumbent_iteration] - retrained_test_scores[best_retrain_test_iteration])
        result['ensemble_regret'].append(ensembled_test_scores[incumbent_iteration] - ensembled_test_scores[best_ensemble_test_iteration])

        # --- Overtuning ---
        retrain_overtuning = max(0.0, retrained_test_scores[incumbent_iteration] - retrained_test_scores[best_retrain_test_incumbent_iteration])
        ensemble_overtuning = max(0.0, ensembled_test_scores[incumbent_iteration] - ensembled_test_scores[best_ensemble_test_incumbent_iteration])
        result['retrain_overtuning'].append(retrain_overtuning)
        result['ensemble_overtuning'].append(ensemble_overtuning)

        # --- Relative overtuning ---
        retrain_max_gain = retrained_test_scores[0] - retrained_test_scores[best_retrain_test_incumbent_iteration]
        if retrain_max_gain > 0:
            result['relative_retrain_overtuning'].append(retrain_overtuning / retrain_max_gain)
        elif retrain_overtuning > 0:
            result['relative_retrain_overtuning'].append(float('inf'))
        else:
            result['relative_retrain_overtuning'].append(0.0)

        ensemble_max_gain = ensembled_test_scores[0] - ensembled_test_scores[best_ensemble_test_incumbent_iteration]
        if ensemble_max_gain > 0:
            result['relative_ensemble_overtuning'].append(ensemble_overtuning / ensemble_max_gain)
        elif ensemble_overtuning > 0:
            result['relative_ensemble_overtuning'].append(float('inf'))
        else:
            result['relative_ensemble_overtuning'].append(0.0)

        # --- Attached scores ---
        for k, v in scores_to_attach.items():
            if v is not None:
                scores_to_attach_dict[k].append(v[incumbent_iteration])

    result['scores_to_attach'] = scores_to_attach_dict
    
    return result


def preprocess_one_experiment(
        results_dir: str,
    verbose: bool = True,
    return_dataframes: bool = True,
    log_trajectory_run_counts: bool = False,
) -> tuple:
    """
    Load, preprocess, and compute trajectories for one experiment directory.

    Returns:
        (results_df, trajectories_df)
    """
    # Load and concatenate all HPO runs in the experiment directory
    results_df = load_experiment(results_dir, verbose)

    # Convert negated maximization metrics to loss scale
    results_df = convert_negative_metric_to_loss(results_df)

    # Metadata columns to propagate into trajectory rows
    metadata_cols = [c for c in [
        'experiment_id', 'experiment_source', 'raw_experiment_id',
        'dataset_id', 'model', 'optimizer', 'metric', 'problem_type',
        'mitigation', 'reshuffling', 'inner_split', 'selection_set_size',
        'repetition', 'outer_fold', 'random_state', 'outer_n_folds', 'outer_n_repeats',
    ] if c in results_df.columns]

    trajectory_rows = []
    trajectory_rows_surrogate = []
    trajectory_rows_one_se = []
    trajectory_rows_makarova = []

    trajectories_path = Path(results_dir) / "trajectories.csv"
    trajectories_surrogate_path = Path(results_dir) / "trajectories_surrogate_selection.csv"
    trajectories_one_se_path = Path(results_dir) / "trajectories_one_se_selection.csv"
    trajectories_makarova_path = Path(results_dir) / "trajectories_makarova.csv"

    wrote_traj_header = False
    wrote_surrogate_header = False
    wrote_one_se_header = False
    wrote_makarova_header = False

    traj_flush_rows = 50_000

    for exp_id, run_df in tqdm(results_df.groupby('experiment_id'), desc="Calculating trajectories"):
        run_df = run_df.sort_values('iteration')
        grouped = run_df.groupby('iteration')

        val_scores = grouped['avg_val_score'].first().values.tolist()
        retrained_test_scores = grouped['retrained_test_score'].first().values.tolist()
        ensembled_test_scores = grouped['cv_ensembled_test_score'].first().values.tolist()
        surrogate_scores = grouped['surrogate_mean'].first().values.tolist()
        val_standard_errors = grouped['val'].apply(lambda x: x.std(ddof=0) / np.sqrt(len(x))).values.tolist()
        fold_counts = grouped['fold'].nunique().values.tolist()

        selection_set_size = run_df['selection_set_size'].iloc[0]
        if pd.notna(selection_set_size) and selection_set_size > 0 and 'avg_selection_score' in run_df.columns:
            selection_scores = grouped['avg_selection_score'].first().values.tolist()
        else:
            selection_scores = None

        traj = calculate_trajectories(
            val_scores=val_scores,
            retrained_test_scores=retrained_test_scores,
            ensembled_test_scores=ensembled_test_scores,
            selection_scores=selection_scores,
            fold_counts=fold_counts,
        )

        surrogate_selection_scores = compute_surrogate_selection_scores(surrogate_scores)
        surrogate_incumbent_indices = compute_incumbent_indices_from_selection_scores(surrogate_selection_scores)
        traj_surrogate = calculate_trajectories(
            val_scores=val_scores,
            retrained_test_scores=retrained_test_scores,
            ensembled_test_scores=ensembled_test_scores,
            incumbent_indices=surrogate_incumbent_indices,
        )

        one_se_incumbent_indices = compute_one_se_incumbent_indices(
            val_scores=val_scores,
            val_standard_errors=val_standard_errors,
        )
        traj_one_se = calculate_trajectories(
            val_scores=val_scores,
            retrained_test_scores=retrained_test_scores,
            ensembled_test_scores=ensembled_test_scores,
            incumbent_indices=one_se_incumbent_indices,
        )

        early_stopped_makarova_list = run_df.groupby('iteration')["early_stopped_makarova"].first().values.tolist()
        makarova_incumbent_indices = compute_makarova_incumbent_indices(
            val_scores=val_scores,
            early_stopped_makarova=early_stopped_makarova_list,
        )
        traj_makarova = calculate_trajectories(
            val_scores=val_scores,
            retrained_test_scores=retrained_test_scores,
            ensembled_test_scores=ensembled_test_scores,
            incumbent_indices=makarova_incumbent_indices,
        )

        meta = {col: run_df[col].iloc[0] for col in metadata_cols}
        iterations = sorted(run_df['iteration'].unique().tolist())

        for i, iteration in enumerate(iterations):
            base_row = {'iteration': iteration}
            base_row.update(meta)

            row = dict(base_row)
            for key, values in traj.items():
                if key == 'scores_to_attach':
                    continue
                row[key] = values[i]

            row_surrogate = dict(base_row)
            for key, values in traj_surrogate.items():
                if key == 'scores_to_attach':
                    continue
                row_surrogate[key] = values[i]

            row_one_se = dict(base_row)
            for key, values in traj_one_se.items():
                if key == 'scores_to_attach':
                    continue
                row_one_se[key] = values[i]

            row_makarova = dict(base_row)
            for key, values in traj_makarova.items():
                if key == 'scores_to_attach':
                    continue
                row_makarova[key] = values[i]

            trajectory_rows.append(row)
            trajectory_rows_surrogate.append(row_surrogate)
            trajectory_rows_one_se.append(row_one_se)
            trajectory_rows_makarova.append(row_makarova)

            if len(trajectory_rows) >= traj_flush_rows:
                wrote_traj_header = _flush_rows_to_csv(
                    trajectory_rows,
                    trajectories_path,
                    wrote_traj_header,
                )
                wrote_surrogate_header = _flush_rows_to_csv(
                    trajectory_rows_surrogate,
                    trajectories_surrogate_path,
                    wrote_surrogate_header,
                )
                wrote_one_se_header = _flush_rows_to_csv(
                    trajectory_rows_one_se,
                    trajectories_one_se_path,
                    wrote_one_se_header,
                )
                wrote_makarova_header = _flush_rows_to_csv(
                    trajectory_rows_makarova,
                    trajectories_makarova_path,
                    wrote_makarova_header,
                )

    results_df.to_csv(f"{results_dir}/preprocessed_results.csv", index=False)
    del results_df

    wrote_traj_header = _flush_rows_to_csv(trajectory_rows, trajectories_path, wrote_traj_header)
    wrote_surrogate_header = _flush_rows_to_csv(
        trajectory_rows_surrogate,
        trajectories_surrogate_path,
        wrote_surrogate_header,
    )
    wrote_one_se_header = _flush_rows_to_csv(
        trajectory_rows_one_se,
        trajectories_one_se_path,
        wrote_one_se_header,
    )
    wrote_makarova_header = _flush_rows_to_csv(
        trajectory_rows_makarova,
        trajectories_makarova_path,
        wrote_makarova_header,
    )

    if not (wrote_traj_header and wrote_surrogate_header and wrote_one_se_header and wrote_makarova_header):
        raise ValueError(f"No trajectory rows written for experiment directory: {results_dir}")

    if log_trajectory_run_counts:
        _log_per_experiment_trajectory_run_counts(results_dir=results_dir, verbose=verbose)

    if return_dataframes:
        results_df = pd.read_csv(f"{results_dir}/preprocessed_results.csv", low_memory=False)
        trajectories_df = pd.read_csv(f"{results_dir}/trajectories.csv", low_memory=False)
        return results_df, trajectories_df

    return None, None


def preprocess_multiple_experiments(
        experiment_dirs: List[str],
        output_path: str = "src/visualizations/data",
    verbose: bool = True,
    recalculate: bool = False,
) -> None:
    """Combine per-experiment outputs into RESULTS_FILE and TRAJECTORIES_FILE in *output_path*.

    Per-experiment CSVs stay in their experiment directories; these two
    gzip-compressed files are the only combined outputs.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILE
    exp_paths = [Path(exp_dir) for exp_dir in experiment_dirs]

    required_exp_outputs = {
        "results": "preprocessed_results.csv",
        "trajectories": "trajectories.csv",
        "one_se": "trajectories_one_se_selection.csv",
        "makarova": "trajectories_makarova.csv",
    }

    for exp_dir in experiment_dirs:
        if verbose:
            print(f"\nProcessing experiment directory: {exp_dir}")

        exp_output_paths = {
            key: Path(exp_dir) / filename
            for key, filename in required_exp_outputs.items()
        }
        has_all_outputs = all(path.exists() for path in exp_output_paths.values())

        if recalculate or not has_all_outputs:
            if verbose and not recalculate and not has_all_outputs:
                missing_files = [
                    str(path)
                    for path in exp_output_paths.values()
                    if not path.exists()
                ]
                print(
                    "Missing preprocessed files; recalculating from raw artifacts: "
                    f"{missing_files}"
                )
            preprocess_one_experiment(
                exp_dir,
                verbose,
                return_dataframes=False,
                log_trajectory_run_counts=recalculate,
            )
        else:
            if verbose:
                print("Using existing preprocessed files and skipping recalculation")

        mlplan_path = Path(exp_dir) / "trajectories_mlplan.csv"
        if recalculate or not mlplan_path.exists():
            # Import lazily to avoid circular imports at module load time.
            from src.visualizations.preprocessing.mlplan_preprocessing import preprocess_mlplan_experiment

            preprocess_mlplan_experiment(
                results_dir=exp_dir,
                n_jobs=1,
                verbose=verbose,
            )

    if verbose:
        print(f"\nWriting {results_path}")
    wrote_header, columns = False, None
    with _open_csv_gz(results_path) as handle:
        write_started = time.monotonic()
        for experiment_index, exp_path in enumerate(exp_paths, start=1):
            experiment_started = time.monotonic()
            wrote_header, columns = append_csv_file(
                exp_path / "preprocessed_results.csv", handle, wrote_header, columns, table_kind="results",
            )
            if verbose:
                now = time.monotonic()
                print(
                    f"[results] Completed {experiment_index}/{len(exp_paths)} "
                    f"({100 * experiment_index / len(exp_paths):.1f}%); "
                    f"experiment={now - experiment_started:.1f}s; "
                    f"elapsed={now - write_started:.1f}s",
                    flush=True,
                )
        if not wrote_header:
            raise ValueError("No experiments were processed")

    # Raw post-hoc ensemble scores extend the ranges so derived post-hoc values stay within [0, 1].
    ranges_df = build_anchor_ranges(
        results_path,
        supplementary_paths=[
            exp_path / "results_post_hoc_ensemble.csv"
            for exp_path in exp_paths
            if (exp_path / "results_post_hoc_ensemble.csv").exists()
        ],
        verbose=verbose,
    )

    trajectories_path = output_dir / TRAJECTORIES_FILE
    if verbose:
        print(f"Writing {trajectories_path}")
    wrote_header, columns = False, None
    trajectory_started = time.monotonic()
    total_trajectory_jobs = len(TRAJECTORY_FILES_BY_METHOD) * len(exp_paths)
    completed_trajectory_jobs = 0
    with _open_csv_gz(trajectories_path) as handle:
        for method, file_name in TRAJECTORY_FILES_BY_METHOD.items():
            def normalize(chunk: pd.DataFrame, method: str = method) -> pd.DataFrame:
                chunk = apply_normalization(chunk, ranges_df)
                chunk.insert(0, "method", method)
                return chunk

            for exp_path in exp_paths:
                rows_written = 0

                def report_chunk_progress(row_count: int) -> None:
                    nonlocal rows_written
                    previous_rows = rows_written
                    rows_written += row_count
                    if verbose and rows_written // 100_000 > previous_rows // 100_000:
                        print(
                            f"[trajectories] method={method}, experiment={exp_path.name}; "
                            f"{rows_written:,} rows written; "
                            f"elapsed={time.monotonic() - trajectory_started:.1f}s",
                            flush=True,
                        )

                wrote_header, columns = append_csv_file(
                    exp_path / file_name, handle, wrote_header, columns, table_kind=method, transform=normalize,
                    progress=report_chunk_progress,
                )
                completed_trajectory_jobs += 1
                if verbose:
                    now = time.monotonic()
                    print(
                        f"[trajectories] Completed {completed_trajectory_jobs}/"
                        f"{total_trajectory_jobs} "
                        f"({100 * completed_trajectory_jobs / total_trajectory_jobs:.1f}%); "
                        f"method={method}, experiment={exp_path.name}, "
                        f"rows={rows_written:,}; "
                        f"elapsed={now - trajectory_started:.1f}s",
                        flush=True,
                    )

    _log_combined_trajectory_dataset_run_counts(output_dir=output_dir, verbose=verbose)