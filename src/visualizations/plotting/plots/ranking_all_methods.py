from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..methods import MethodSpec, build_all_methods_cdf_with_benchmark


@dataclass
class AllMethodsRankingStats:
    plotted_series: int


def _ranking_context_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in ["dataset_id", "optimizer", "repetition", "outer_fold", "model"]
        if col in df.columns
    ]


def _prepare_method_metric_rows(
    trajectories: pd.DataFrame,
    budget: str,
    metric_col: str,
    method_builder: Callable[[pd.DataFrame, str], list[tuple[MethodSpec, pd.Series]]],
    iteration_col: str,
    metric_label: str,
    absolute_metric: bool,
) -> pd.DataFrame:
    context_cols = _ranking_context_columns(trajectories)
    required_cols = [iteration_col, metric_col, *context_cols]
    rows: list[pd.DataFrame] = []

    for spec, mask in method_builder(trajectories, budget):
        method_df = trajectories.loc[mask, required_cols].copy()
        if method_df.empty:
            continue

        method_df[iteration_col] = pd.to_numeric(method_df[iteration_col], errors="coerce")
        method_df[metric_col] = pd.to_numeric(method_df[metric_col], errors="coerce")
        method_df = method_df.dropna(subset=[iteration_col, metric_col])
        if method_df.empty:
            continue

        if absolute_metric:
            method_df[metric_col] = method_df[metric_col].abs()

        # One value per context and iteration for stable cross-method ranking.
        group_cols = [iteration_col, *context_cols]
        method_df = (
            method_df.groupby(group_cols, dropna=False)[metric_col]
            .mean()
            .reset_index()
        )
        method_df["method_label"] = spec.label
        method_df["method_color"] = spec.color
        method_df["metric_name"] = metric_label
        rows.append(method_df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def plot_all_methods_average_rank_trajectory(
    trajectories: pd.DataFrame,
    budget: str,
    metric_col: str,
    metric_label: str,
    absolute_metric: bool = False,
    iteration_col: str = "iteration",
    method_builder: Callable[[pd.DataFrame, str], list[tuple[MethodSpec, pd.Series]]] = build_all_methods_cdf_with_benchmark,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
    aggregate_per_dataset_first: bool = False,
) -> tuple[plt.Figure, AllMethodsRankingStats]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
        ax.clear()

    ranked_rows = _prepare_method_metric_rows(
        trajectories=trajectories,
        budget=budget,
        metric_col=metric_col,
        method_builder=method_builder,
        iteration_col=iteration_col,
        metric_label=metric_label,
        absolute_metric=absolute_metric,
    )

    if ranked_rows.empty:
        ax.text(0.5, 0.5, f"No plottable rows for {metric_label}", ha="center", va="center", transform=ax.transAxes)
        return fig, AllMethodsRankingStats(plotted_series=0)

    if aggregate_per_dataset_first and "dataset_id" in ranked_rows.columns:
        dataset_rows = (
            ranked_rows.groupby([iteration_col, "dataset_id", "method_label", "method_color"], dropna=False)[metric_col]
            .mean()
            .reset_index()
        )
        dataset_rows["rank"] = (
            dataset_rows.groupby([iteration_col, "dataset_id"], dropna=False)[metric_col]
            .rank(method="average", ascending=True)
        )
        avg_rank = (
            dataset_rows.groupby([iteration_col, "method_label", "method_color"], dropna=False)["rank"]
            .mean()
            .reset_index()
            .sort_values(["method_label", iteration_col])
        )
    else:
        context_cols = _ranking_context_columns(ranked_rows)
        rank_group_cols = [iteration_col, *context_cols]
        ranked_rows["rank"] = (
            ranked_rows.groupby(rank_group_cols, dropna=False)[metric_col]
            .rank(method="average", ascending=True)
        )

        avg_rank = (
            ranked_rows.groupby([iteration_col, "method_label", "method_color"], dropna=False)["rank"]
            .mean()
            .reset_index()
            .sort_values(["method_label", iteration_col])
        )

    plotted = 0
    max_iter = 0.0
    for (method_label, method_color), group_df in avg_rank.groupby(["method_label", "method_color"], dropna=False):
        x_vals = pd.to_numeric(group_df[iteration_col], errors="coerce").to_numpy(dtype=float)
        y_vals = group_df["rank"].to_numpy(dtype=float)
        finite = np.isfinite(x_vals) & np.isfinite(y_vals)
        if not finite.any():
            continue
        x_vals = x_vals[finite]
        y_vals = y_vals[finite]
        if len(x_vals) == 0:
            continue

        ax.plot(x_vals, y_vals, color=str(method_color), linewidth=2.0, label=str(method_label))
        max_iter = max(max_iter, float(np.nanmax(x_vals)))
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, f"No plottable ranks for {metric_label}", ha="center", va="center", transform=ax.transAxes)
        return fig, AllMethodsRankingStats(plotted_series=0)

    if max_iter > 0:
        ax.set_xlim(0.0, max_iter)

    max_rank = float(np.nanmax(avg_rank["rank"].to_numpy(dtype=float))) if not avg_rank.empty else 1.0
    ax.set_ylim(1.0, max(1.0, max_rank) + 0.05)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(f"Average rank by {metric_label} (lower is better)")
    ax.grid(True, alpha=0.3, linestyle="--")

    if show_legend:
        ax.legend(loc="lower right", fontsize=12, framealpha=0.95, ncol=1)

    return fig, AllMethodsRankingStats(plotted_series=plotted)