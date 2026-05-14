from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..methods import MethodSpec, build_all_methods_cdf_with_benchmark


@dataclass
class AllMethodsTrajectoryStats:
    plotted_series: int


def plot_all_methods_normalized_test_error_trajectory(
    trajectories: pd.DataFrame,
    budget: str,
    iteration_col: str = "iteration",
    test_col: str = "normalized_ensembled_test_loss",
    method_builder: Callable[[pd.DataFrame, str], list[tuple[MethodSpec, pd.Series]]] = build_all_methods_cdf_with_benchmark,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
    aggregate_per_dataset_first: bool = False,
) -> tuple[plt.Figure, AllMethodsTrajectoryStats]:
    required_cols = {iteration_col, test_col}
    missing = [col for col in required_cols if col not in trajectories.columns]
    if missing:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
            ax.clear()
        ax.text(0.5, 0.5, f"Missing required columns: {missing}", ha="center", va="center", transform=ax.transAxes)
        return fig, AllMethodsTrajectoryStats(plotted_series=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
        ax.clear()

    plotted = 0
    max_iter = 0.0

    for spec, mask in method_builder(trajectories, budget):
        method_df = trajectories.loc[mask, [iteration_col, test_col]].copy()
        if aggregate_per_dataset_first and "dataset_id" in trajectories.columns:
            method_df = trajectories.loc[mask, ["dataset_id", iteration_col, test_col]].copy()
        if method_df.empty:
            continue

        method_df[iteration_col] = pd.to_numeric(method_df[iteration_col], errors="coerce")
        method_df[test_col] = pd.to_numeric(method_df[test_col], errors="coerce")
        method_df = method_df.dropna(subset=[iteration_col, test_col])
        if method_df.empty:
            continue

        if aggregate_per_dataset_first and "dataset_id" in method_df.columns:
            dataset_iter = (
                method_df.groupby(["dataset_id", iteration_col], dropna=False)[test_col]
                .mean()
                .reset_index()
            )
            agg = (
                dataset_iter.groupby(iteration_col)
                .agg(mean=(test_col, "mean"), std=(test_col, "std"), count=(test_col, "count"))
                .reset_index()
                .sort_values(iteration_col)
            )
        else:
            agg = (
                method_df.groupby(iteration_col)
                .agg(mean=(test_col, "mean"), std=(test_col, "std"), count=(test_col, "count"))
                .reset_index()
                .sort_values(iteration_col)
            )
        if agg.empty:
            continue

        x_vals = agg[iteration_col].to_numpy(dtype=float)
        y_mean = agg["mean"].to_numpy(dtype=float)
        # Shade +/- 1 std to provide a clearly visible uncertainty band.
        y_err = agg["std"].fillna(0.0).to_numpy(dtype=float)

        ax.plot(x_vals, y_mean, color=spec.color, linewidth=2.0, label=spec.label, zorder=3)

        max_iter = max(max_iter, float(np.nanmax(x_vals)))
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No plottable trajectories", ha="center", va="center", transform=ax.transAxes)
        return fig, AllMethodsTrajectoryStats(plotted_series=0)

    if max_iter > 0:
        ax.set_xlim(0.0, max_iter)
    ax.set_xlabel("Iterations")
    if test_col.startswith("normalized_"):
        ax.set_ylabel("Normalized test loss")
    else:
        ax.set_ylabel("Test loss")
    ax.grid(True, alpha=0.3, linestyle="--")

    if show_legend:
        ax.legend(loc="lower right", fontsize=12, framealpha=0.95, ncol=1)

    return fig, AllMethodsTrajectoryStats(plotted_series=plotted)