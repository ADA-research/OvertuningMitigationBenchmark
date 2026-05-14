"""
Post-combine normalization for trajectory files.

Normalizes all trajectory columns using a global empirical range built from ALL
observed raw model scores (not just incumbents).  Normalization is applied once,
after all per-experiment outputs have been concatenated into the combined files.

The anchor source is preprocessed_results.csv, which contains one row per
configuration per experiment (all evaluated models, all iterations).  Using this
full set gives a true empirical [min, max] range so that every model – including
very poor ones – falls inside the [0, 1] band.

Normalized columns written to every combined trajectory file:
    normalized_val_loss
    normalized_retrain_test_loss
    normalized_ensembled_test_loss
    normalized_retrain_overtuning       (= retrain_overtuning / ret_range)
    normalized_ensemble_overtuning      (= ensemble_overtuning / ens_range)

Important normalization contract:
    Overtuning is NOT normalized on its own min/max.
    It is always normalized on the same denominator as the corresponding test
    loss scale (ret_range for retrain, ens_range for ensemble).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import os
import tempfile

import numpy as np
import pandas as pd


GROUP_COLS = ["dataset_id", "repetition", "outer_fold", "metric", "model"]

COMBINED_TRAJECTORY_FILES = [
    "trajectories.csv",
    "trajectories_one_se_selection.csv",
    "trajectories_makarova.csv",
    "trajectories_post_hoc_surrogate.csv",
    "trajectories_post_hoc_ensemble.csv",
    "trajectories_mlplan.csv",
]


def build_normalization_ranges(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-group normalization ranges from all observed model scores.

    Args:
        results_df: DataFrame with columns avg_val_score, retrained_test_score,
                    cv_ensembled_test_score (one row per config per experiment).

    Returns:
        DataFrame with one row per group (dataset_id, repetition, outer_fold, metric, model)
        and columns: val_min, val_max, ret_min, ret_max, ens_min, ens_max.
    """
    missing = [c for c in [*GROUP_COLS, "avg_val_score", "retrained_test_score", "cv_ensembled_test_score"] if c not in results_df.columns]
    if missing:
        raise ValueError(f"build_normalization_ranges: missing columns in results_df: {missing}")

    agg_spec = {
        "val_min": ("avg_val_score", "min"),
        "val_max": ("avg_val_score", "max"),
        "ret_min": ("retrained_test_score", "min"),
        "ret_max": ("retrained_test_score", "max"),
        "ens_min": ("cv_ensembled_test_score", "min"),
        "ens_max": ("cv_ensembled_test_score", "max"),
    }
    return results_df.groupby(GROUP_COLS, dropna=False).agg(**agg_spec).reset_index()


def _combine_partial_ranges(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Merge two per-group range tables into one by taking global min/max."""
    if left.empty:
        return right
    if right.empty:
        return left

    merged = pd.concat([left, right], ignore_index=True)
    return (
        merged.groupby(GROUP_COLS, dropna=False, as_index=False)
        .agg(
            val_min=("val_min", "min"),
            val_max=("val_max", "max"),
            ret_min=("ret_min", "min"),
            ret_max=("ret_max", "max"),
            ens_min=("ens_min", "min"),
            ens_max=("ens_max", "max"),
        )
    )


def build_normalization_ranges_from_csv(
    results_path: Path,
    chunk_size: int = 200_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build normalization ranges from CSV in chunks to avoid OOM."""
    usecols = [*GROUP_COLS, "avg_val_score", "retrained_test_score", "cv_ensembled_test_score"]

    ranges_df: pd.DataFrame | None = None
    total_rows = 0
    chunk_count = 0

    with pd.read_csv(results_path, low_memory=False, usecols=usecols, chunksize=chunk_size) as reader:
        for chunk in reader:
            chunk_count += 1
            total_rows += len(chunk)

            partial_ranges = build_normalization_ranges(chunk)
            if ranges_df is None:
                ranges_df = partial_ranges
            else:
                ranges_df = _combine_partial_ranges(ranges_df, partial_ranges)

            if verbose and chunk_count % 10 == 0:
                print(
                    f"[normalization] Processed {total_rows:,} rows from "
                    f"{chunk_count} chunk(s); current groups={len(ranges_df):,}"
                )

    if ranges_df is None:
        raise ValueError(f"[normalization] No rows found in {results_path}")

    if verbose:
        print(
            f"[normalization] Built anchor ranges from {total_rows:,} rows "
            f"across {len(ranges_df):,} groups (all observed models)"
        )

    return ranges_df


def apply_normalization(df: pd.DataFrame, ranges_df: pd.DataFrame) -> pd.DataFrame:
    """Apply group-level normalization to a trajectory DataFrame.

    Adds (or overwrites) the following columns:
        normalized_val_loss
        normalized_retrain_test_loss
        normalized_ensembled_test_loss
        normalized_retrain_overtuning
        normalized_ensemble_overtuning

    The overtuning columns are normalized by dividing by the SAME test loss
    ranges used for test-loss normalization (no separate overtuning range).
    This is equivalent to recomputing trajectories on normalized test losses
    and then recomputing overtuning, because the scaling is linear.

    Args:
        df: Combined trajectory DataFrame.  Must contain val_performance,
            retrain_test_loss, ensembled_test_loss, retrain_overtuning,
            ensemble_overtuning and the GROUP_COLS.
        ranges_df: Output of build_normalization_ranges().

    Returns:
        Copy of df with normalized columns added/updated.
    """
    required = [*GROUP_COLS, "val_performance", "retrain_test_loss", "ensembled_test_loss",
                "retrain_overtuning", "ensemble_overtuning"]  # GROUP_COLS includes model
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"apply_normalization: missing columns in trajectory df: {missing}")

    out = df.copy()
    merged = out.merge(ranges_df, on=GROUP_COLS, how="left", validate="many_to_one")

    val_range = merged["val_max"] - merged["val_min"]
    ret_range = merged["ret_max"] - merged["ret_min"]
    ens_range = merged["ens_max"] - merged["ens_min"]

    def _normalize_with_range(
        value: pd.Series,
        lower: pd.Series,
        denom: pd.Series,
    ) -> pd.Series:
        return ((value - lower) / denom).where(denom > 1e-14, np.nan)

    def _normalize_overtuning_with_shared_scale(
        overtuning: pd.Series,
        denom: pd.Series,
    ) -> pd.Series:
        # Shared-scale contract: overtuning uses the exact same denominator as
        # its corresponding test-loss normalization.
        return (overtuning / denom).where(denom > 1e-14, np.nan)

    merged["normalized_val_loss"] = _normalize_with_range(
        value=merged["val_performance"],
        lower=merged["val_min"],
        denom=val_range,
    )

    merged["normalized_retrain_test_loss"] = _normalize_with_range(
        value=merged["retrain_test_loss"],
        lower=merged["ret_min"],
        denom=ret_range,
    )

    merged["normalized_ensembled_test_loss"] = _normalize_with_range(
        value=merged["ensembled_test_loss"],
        lower=merged["ens_min"],
        denom=ens_range,
    )

    # Shared-scale overtuning normalization: same denominator as test loss.
    merged["normalized_retrain_overtuning"] = _normalize_overtuning_with_shared_scale(
        overtuning=merged["retrain_overtuning"],
        denom=ret_range,
    )

    merged["normalized_ensemble_overtuning"] = _normalize_overtuning_with_shared_scale(
        overtuning=merged["ensemble_overtuning"],
        denom=ens_range,
    )

    range_cols = ["val_min", "val_max", "ret_min", "ret_max", "ens_min", "ens_max"]
    return merged.drop(columns=range_cols)


def normalize_combined_trajectories(
    data_dir: Path,
    results_file: str = "preprocessed_results.csv",
    file_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """Normalize all combined trajectory files using all observed raw model scores.

    Loads *results_file* (preprocessed_results.csv) to obtain normalization ranges
    from all evaluated configurations across all experiments, then writes normalized
    columns into each combined trajectory file in-place.

    Args:
        data_dir: Directory containing the combined output CSV files.
        results_file: Name of the combined preprocessed results file inside data_dir.
        file_names: Trajectory file names to normalize.  Defaults to
                    COMBINED_TRAJECTORY_FILES.
        verbose: Whether to print progress information.
    """
    if file_names is None:
        file_names = COMBINED_TRAJECTORY_FILES

    results_path = data_dir / results_file
    if not results_path.exists():
        raise FileNotFoundError(
            f"[normalization] Preprocessed results file not found: {results_path}. "
            "Run preprocessing before normalization."
        )

    ranges_df = build_normalization_ranges_from_csv(
        results_path=results_path,
        chunk_size=200_000,
        verbose=verbose,
    )

    # Extend anchor ranges with any supplementary results files (e.g. post-hoc
    # ensemble raw scores) so that derived scores from those sources also fall
    # within [0, 1] after normalization.  The combined ranges take the global
    # min and max across all anchor sources per group.
    supplementary_files = [
        "preprocessed_results_post_hoc_ensemble.csv",
        "preprocessed_results_post_hoc_surrogate.csv",
    ]
    for sup_name in supplementary_files:
        sup_path = data_dir / sup_name
        if not sup_path.exists():
            continue
        sup_ranges = build_normalization_ranges_from_csv(
            results_path=sup_path,
            chunk_size=200_000,
            verbose=False,
        )
        before = len(ranges_df)
        ranges_df = _combine_partial_ranges(ranges_df, sup_ranges)
        if verbose:
            print(
                f"[normalization] Extended anchor with {sup_name}: "
                f"{before:,} → {len(ranges_df):,} groups"
            )

    _CHUNK_SIZE = 100_000

    for file_name in file_names:
        file_path = data_dir / file_name
        if not file_path.exists():
            if verbose:
                print(f"[normalization] Skipping missing file: {file_name}")
            continue

        # Write normalized output to a temp file in the same directory, then
        # atomically replace the original so a partial write never corrupts it.
        tmp_fd, tmp_path = tempfile.mkstemp(dir=data_dir, suffix=".tmp")
        os.close(tmp_fd)
        try:
            total_rows = 0
            first_chunk = True
            with pd.read_csv(file_path, chunksize=_CHUNK_SIZE, low_memory=False) as reader:
                for chunk in reader:
                    chunk = apply_normalization(chunk, ranges_df)
                    chunk.to_csv(
                        tmp_path,
                        mode="w" if first_chunk else "a",
                        header=first_chunk,
                        index=False,
                    )
                    total_rows += len(chunk)
                    first_chunk = False
            os.replace(tmp_path, file_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        if verbose:
            print(f"[normalization] Normalized: {file_name} ({total_rows:,} rows)")
