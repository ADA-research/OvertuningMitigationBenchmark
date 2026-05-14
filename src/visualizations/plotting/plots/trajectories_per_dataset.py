from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..methods import _baseline_mitigation_mask, _inner_split_is, _reshuffling_is, _selection_zero_or_na


VAL_COLOR = "#1f77b4"
TEST_COLOR = "#ff7f0e"


@dataclass
class BenchmarkTrajectoryStats:
    runs: int
    points: int


def _benchmark_budget_mask(df: pd.DataFrame, budget: str) -> pd.Series:
    if "inner_split" not in df.columns:
        return pd.Series(False, index=df.index)
    return (
        _baseline_mitigation_mask(df)
        & _selection_zero_or_na(df)
        & _reshuffling_is(df, target=False)
        & _inner_split_is(df, budget.strip().lower())
    )


def _budget_title_suffix(budget: str) -> str:
    budget_norm = budget.strip().lower()
    if budget_norm == "cv":
        return "5CV"
    if budget_norm == "holdout":
        return "Holdout"
    return budget.capitalize()


def plot_benchmark_dataset_trajectory(
    trajectories_subset: pd.DataFrame,
    dataset_label: str,
    budget: str,
    iteration_col: str = "iteration",
    val_col: str = "val_performance",
    test_col: str = "ensembled_test_loss",
) -> tuple[plt.Figure, BenchmarkTrajectoryStats]:
    required_cols = {iteration_col, val_col, test_col}
    missing = [col for col in required_cols if col not in trajectories_subset.columns]
    if missing:
        fig, ax = plt.subplots(figsize=(8.1, 5.5))
        ax.text(0.5, 0.5, f"Missing required columns: {missing}", ha="center", va="center", transform=ax.transAxes)
        return fig, BenchmarkTrajectoryStats(runs=0, points=0)

    benchmark_df = trajectories_subset.loc[_benchmark_budget_mask(trajectories_subset, budget=budget)].copy()
    if benchmark_df.empty:
        fig, ax = plt.subplots(figsize=(8.1, 5.5))
        ax.text(0.5, 0.5, f"No benchmark {budget.upper()} rows for dataset", ha="center", va="center", transform=ax.transAxes)
        return fig, BenchmarkTrajectoryStats(runs=0, points=0)

    benchmark_df[iteration_col] = pd.to_numeric(benchmark_df[iteration_col], errors="coerce")
    benchmark_df[val_col] = pd.to_numeric(benchmark_df[val_col], errors="coerce")
    benchmark_df[test_col] = pd.to_numeric(benchmark_df[test_col], errors="coerce")
    benchmark_df = benchmark_df.dropna(subset=[iteration_col, val_col, test_col])
    if benchmark_df.empty:
        fig, ax = plt.subplots(figsize=(8.1, 5.5))
        ax.text(0.5, 0.5, f"No finite benchmark {budget.upper()} rows for dataset", ha="center", va="center", transform=ax.transAxes)
        return fig, BenchmarkTrajectoryStats(runs=0, points=0)

    agg = (
        benchmark_df.groupby(iteration_col)
        .agg(
            val_mean=(val_col, "mean"),
            val_std=(val_col, "std"),
            val_count=(val_col, "count"),
            test_mean=(test_col, "mean"),
            test_std=(test_col, "std"),
            test_count=(test_col, "count"),
        )
        .reset_index()
        .sort_values(iteration_col)
    )
    agg["val_se"] = agg["val_std"].fillna(0.0) / np.sqrt(agg["val_count"].clip(lower=1))
    agg["test_se"] = agg["test_std"].fillna(0.0) / np.sqrt(agg["test_count"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(8.1, 5.5))
    x_vals = agg[iteration_col].to_numpy(dtype=float)

    val_mean = agg["val_mean"].to_numpy(dtype=float)
    val_se = agg["val_se"].to_numpy(dtype=float)
    ax.plot(x_vals, val_mean, color=VAL_COLOR, linewidth=2.25, label="Validation")
    ax.fill_between(x_vals, val_mean - val_se, val_mean + val_se, color=VAL_COLOR, alpha=0.2)

    test_mean = agg["test_mean"].to_numpy(dtype=float)
    test_se = agg["test_se"].to_numpy(dtype=float)
    ax.plot(x_vals, test_mean, color=TEST_COLOR, linewidth=2.25, label="Test")
    ax.fill_between(x_vals, test_mean - test_se, test_mean + test_se, color=TEST_COLOR, alpha=0.2)

    run_keys = [
        col
        for col in ["optimizer", "repetition", "outer_fold", "experiment_id"]
        if col in benchmark_df.columns
    ]
    run_count = benchmark_df[run_keys].drop_duplicates().shape[0] if run_keys else 0

    ax.set_title(
        f"{dataset_label} - LightGBM - {_budget_title_suffix(budget)}",
        fontweight="normal",
        fontsize=16,
    )
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Error", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=13, framealpha=0.88)
    fig.tight_layout()

    return fig, BenchmarkTrajectoryStats(runs=run_count, points=len(agg))