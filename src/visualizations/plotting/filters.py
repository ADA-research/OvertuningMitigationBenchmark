from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_RUN_KEY_CANDIDATES = [
    "trajectory_source",
    "experiment_id",
    "dataset_id",
    "optimizer",
    "inner_split",
    "mitigation",
    "selection_set_size",
    "reshuffling",
    "repetition",
    "outer_fold",
    "model",
]


@dataclass
class TunableFilterResult:
    filtered_df: pd.DataFrame
    included_runs: int
    excluded_runs: int
    median_threshold: float


def run_key_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in _RUN_KEY_CANDIDATES if col in df.columns]


def normalize_dataset_id_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    value_str = str(value).strip()
    if value_str.endswith(".0") and value_str[:-2].isdigit():
        return value_str[:-2]
    return value_str


def apply_subset_filters(
    df: pd.DataFrame,
    dataset_ids: list[str] | None = None,
    optimizers: list[str] | None = None,
    problem_types: list[str] | None = None,
    mitigations: list[str] | None = None,
    inner_splits: list[str] | None = None,
) -> pd.DataFrame:
    """Apply common row filters before passing data to plotting functions."""
    out = df.copy()

    if dataset_ids and "dataset_id" in out.columns:
        allow = {normalize_dataset_id_value(v) for v in dataset_ids}
        ds = out["dataset_id"].map(normalize_dataset_id_value)
        out = out.loc[ds.isin(allow)]

    if optimizers and "optimizer" in out.columns:
        allow = {str(v).strip().lower() for v in optimizers}
        series = out["optimizer"].fillna("").astype(str).str.strip().str.lower()
        out = out.loc[series.isin(allow)]

    if problem_types and "problem_type" in out.columns:
        allow = {str(v).strip().lower() for v in problem_types}
        series = out["problem_type"].fillna("").astype(str).str.strip().str.lower()
        out = out.loc[series.isin(allow)]

    if mitigations and "mitigation" in out.columns:
        allow = {str(v).strip().lower() for v in mitigations}
        series = out["mitigation"].fillna("").astype(str).str.strip().str.lower()
        out = out.loc[series.isin(allow)]

    if inner_splits and "inner_split" in out.columns:
        allow = {str(v).strip().lower() for v in inner_splits}
        series = out["inner_split"].fillna("").astype(str).str.strip().str.lower()
        out = out.loc[series.isin(allow)]

    return out


def baseline_unmitigated_mask(df: pd.DataFrame) -> pd.Series:
    """Rows with no mitigation, no reshuffling, and no selection set."""
    mask = pd.Series(True, index=df.index)

    if "mitigation" in df.columns:
        mitigation = df["mitigation"].fillna("").astype(str).str.strip().str.lower()
        mask &= mitigation.isin(["", "none"])

    if "reshuffling" in df.columns:
        reshuffling = (
            df["reshuffling"]
            .map(lambda v: "" if pd.isna(v) else str(v).strip().lower())
            .eq("false")
        )
        mask &= reshuffling

    if "selection_set_size" in df.columns:
        selection = pd.to_numeric(df["selection_set_size"], errors="coerce")
        mask &= selection.isna() | (selection == 0.0)

    return mask


def keep_baseline_unmitigated_runs(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[baseline_unmitigated_mask(df)].copy()


def final_rows_per_run(df: pd.DataFrame, iteration_col: str = "iteration") -> pd.DataFrame:
    if df.empty or iteration_col not in df.columns:
        return pd.DataFrame()

    run_cols = run_key_columns(df)
    work = df.copy()
    work["__iter_num"] = pd.to_numeric(work[iteration_col], errors="coerce")
    valid = work.loc[work["__iter_num"].notna()]
    if valid.empty:
        return pd.DataFrame(columns=df.columns)

    if run_cols:
        # Avoid unobserved categorical combinations creating empty groups.
        idx = valid.groupby(run_cols, dropna=False, observed=True)["__iter_num"].idxmax()
    else:
        idx = pd.Index([valid["__iter_num"].idxmax()])

    out = work.loc[idx].drop(columns=["__iter_num"]).reset_index(drop=True)
    return out


def _run_threshold(run_df: pd.DataFrame, score_col: str, base_threshold: float) -> float:
    problem_type = "classification"
    if "problem_type" in run_df.columns:
        problem_types = run_df["problem_type"].dropna().astype(str).str.lower().unique()
        if len(problem_types) > 0:
            problem_type = problem_types[0]

    if problem_type in {"binary", "multiclass", "classification"}:
        return base_threshold

    run_sorted = run_df.sort_values("iteration") if "iteration" in run_df.columns else run_df
    if problem_type == "regression" and score_col in run_sorted.columns and not run_sorted.empty:
        first_score = run_sorted.iloc[0][score_col]
        if pd.notna(first_score):
            return base_threshold * abs(float(first_score))

    return base_threshold


def filter_tunable_runs(
    df: pd.DataFrame,
    score_col: str = "ensembled_test_loss",
    base_threshold: float = 0.001,
) -> TunableFilterResult:
    """Keep runs where tuning improves the score beyond threshold.

    Lower scores are better. A run is "tunable" if:
    first_iteration_score - best_score > threshold_for_run
    """
    if "iteration" not in df.columns:
        raise ValueError("Expected column 'iteration'")
    if score_col not in df.columns:
        raise ValueError(f"Expected column '{score_col}'")

    run_cols = run_key_columns(df)
    work_cols = [*run_cols, "iteration", score_col]
    if "problem_type" in df.columns:
        work_cols.append("problem_type")

    work = df[work_cols].copy()
    work["__iter_num"] = pd.to_numeric(work["iteration"], errors="coerce")
    work["__score_num"] = pd.to_numeric(work[score_col], errors="coerce")

    if run_cols:
        # Use observed groups only to prevent empty categorical groups.
        work["__gid"] = work.groupby(run_cols, dropna=False, observed=True).ngroup()
    else:
        work["__gid"] = 0

    valid = work.loc[work["__iter_num"].notna() & work["__score_num"].notna()]
    if valid.empty:
        return TunableFilterResult(
            filtered_df=pd.DataFrame(columns=df.columns),
            included_runs=0,
            excluded_runs=int(work["__gid"].nunique()),
            median_threshold=float(base_threshold),
        )

    first_idx = valid.sort_values(["__gid", "__iter_num"]).groupby("__gid", dropna=False).head(1).set_index("__gid")
    first_score = first_idx["__score_num"]
    best_score = valid.groupby("__gid", dropna=False)["__score_num"].min()

    threshold = pd.Series(float(base_threshold), index=first_score.index, dtype="float64")
    if "problem_type" in first_idx.columns:
        ptype = first_idx["problem_type"].astype(str).str.lower()
        reg_mask = ptype.eq("regression")
        threshold.loc[reg_mask] = float(base_threshold) * first_score.loc[reg_mask].abs()

    improvement = first_score - best_score
    keep_gids = improvement.index[(improvement > threshold).fillna(False)]

    keep_mask = work["__gid"].isin(keep_gids)
    filtered = df.loc[keep_mask.to_numpy()].copy()

    total_runs = int(work["__gid"].nunique())
    included = int(len(keep_gids))
    excluded = max(total_runs - included, 0)
    median_threshold = float(np.median(threshold.to_numpy(dtype=float))) if len(threshold) > 0 else float(base_threshold)
    return TunableFilterResult(
        filtered_df=filtered,
        included_runs=included,
        excluded_runs=excluded,
        median_threshold=median_threshold,
    )
