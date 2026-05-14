from dataclasses import dataclass
from typing import Callable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from ..filters import final_rows_per_run
from ..methods import MethodSpec, benchmark_mask, build_delta_methods


def _darken(hex_color: str, factor: float = 0.55) -> str:
    """Return a darkened version of hex_color (factor < 1 = darker)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex((r * factor, g * factor, b * factor))


@dataclass
class DeltaStats:
    points_total: int


def _pair_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in ["dataset_id", "optimizer", "repetition", "outer_fold", "model"]
        if col in df.columns
    ]


def _normalize_dataset_id(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)


def _build_delta_points(
    final_df: pd.DataFrame,
    label: str,
    mitigation_mask: pd.Series,
    budget: str,
    overtuning_col: str,
    test_loss_col: str,
    benchmark_rows_mask: pd.Series | None = None,
    aggregate_per_dataset_first: bool = False,
) -> pd.DataFrame:
    if benchmark_rows_mask is None:
        benchmark_rows_mask = benchmark_mask(final_df, budget=budget, reshuffling=False)
    bench = final_df.loc[benchmark_rows_mask].copy()
    mitig = final_df.loc[mitigation_mask].copy()

    if bench.empty or mitig.empty:
        return pd.DataFrame()

    pair_cols = _pair_columns(final_df)
    if not pair_cols:
        return pd.DataFrame()

    if "dataset_id" in pair_cols:
        bench["dataset_id"] = _normalize_dataset_id(bench["dataset_id"])
        mitig["dataset_id"] = _normalize_dataset_id(mitig["dataset_id"])

    extra_cols = [c for c in ["problem_type"] if c in mitig.columns]
    bench = bench[pair_cols + [overtuning_col, test_loss_col]].rename(
        columns={
            overtuning_col: "benchmark_overtuning",
            test_loss_col: "benchmark_test_loss",
        }
    )
    mitig = mitig[pair_cols + extra_cols + [overtuning_col, test_loss_col]].rename(
        columns={
            overtuning_col: "mitigation_overtuning",
            test_loss_col: "mitigation_test_loss",
        }
    )

    paired = mitig.merge(bench, on=pair_cols, how="inner")
    if paired.empty:
        return paired

    paired["delta_overtuning"] = paired["mitigation_overtuning"] - paired["benchmark_overtuning"]
    paired["delta_test_loss"] = paired["mitigation_test_loss"] - paired["benchmark_test_loss"]
    paired["mitigation_label"] = label

    # To avoid overweighting datasets with many runs, collapse to one average
    # delta per dataset before plotting/summary.
    if aggregate_per_dataset_first and "dataset_id" in paired.columns:
        group_cols = ["dataset_id"]
        agg = (
            paired.groupby(group_cols, dropna=False)
            .agg(
                delta_overtuning=("delta_overtuning", "mean"),
                delta_test_loss=("delta_test_loss", "mean"),
                problem_type=("problem_type", "first") if "problem_type" in paired.columns else ("mitigation_label", "first"),
            )
            .reset_index()
        )
        agg["mitigation_label"] = label
        return agg
    return paired


def plot_relative_overtuning_delta_scatter(
    trajectories: pd.DataFrame,
    budget: str = "cv",
    pad: float = 0.08,
    overtuning_col: str = "normalized_ensemble_overtuning",
    test_loss_col: str = "normalized_ensembled_test_loss",
    clamp_cv_axes: bool = True,
    title_suffix: str = "",
    method_builder: Callable[[pd.DataFrame, str], list[tuple[MethodSpec, pd.Series]]] = build_delta_methods,
    benchmark_mask_builder: Callable[[pd.DataFrame, str], pd.Series] | None = None,
    show_raw_points: bool = False,
    summary_by_problem_type: bool = False,
    fixed_xlim: tuple[float, float] | None = None,
    fixed_ylim: tuple[float, float] | None = None,
    split_method_problem_legend: bool = False,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
    panel_title: str | None = None,
    precomputed_final_rows: pd.DataFrame | None = None,
    aggregate_per_dataset_first: bool = False,
) -> tuple[plt.Figure, DeltaStats]:
    """Delta scatter of overtuning vs test loss deltas.

    All runs are included (no tunable-run filtering).
    Marker color encodes mitigation method.
    Axis limits are auto-zoomed to the data with `pad` margin.
    """
    required_cols = {overtuning_col, test_loss_col}
    missing = [c for c in required_cols if c not in trajectories.columns]
    if missing:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6.7))
        else:
            fig = ax.figure
            ax.clear()
        ax.text(0.5, 0.5, f"Missing required columns: {missing}", ha="center", va="center", transform=ax.transAxes)
        return fig, DeltaStats(0)

    final_df = precomputed_final_rows if precomputed_final_rows is not None else final_rows_per_run(trajectories)
    if final_df.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6.7))
        else:
            fig = ax.figure
            ax.clear()
        ax.text(0.5, 0.5, "No final run rows", ha="center", va="center", transform=ax.transAxes)
        return fig, DeltaStats(0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6.7))
    else:
        fig = ax.figure
        ax.clear()

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    # Per-summary (mean, std) pairs for the 1/3-bar scaling rule
    summary_mean_x: list[float] = []
    summary_std_x: list[float] = []
    summary_mean_y: list[float] = []
    summary_std_y: list[float] = []
    total_points = 0
    any_series = False
    legend_handles: list[Line2D] = []
    benchmark_rows = (
        benchmark_mask(final_df, budget=budget, reshuffling=False)
        if benchmark_mask_builder is None
        else benchmark_mask_builder(final_df, budget)
    )

    for spec, mask in method_builder(final_df, budget):
        paired = _build_delta_points(
            final_df,
            spec.label,
            mask,
            budget,
            overtuning_col,
            test_loss_col,
            benchmark_rows,
            aggregate_per_dataset_first=aggregate_per_dataset_first,
        )
        if paired.empty:
            continue

        x_vals = paired["delta_overtuning"].to_numpy(dtype=float)
        y_vals = paired["delta_test_loss"].to_numpy(dtype=float)
        finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if not finite_mask.any():
            continue
        x_vals = x_vals[finite_mask]
        y_vals = y_vals[finite_mask]
        edge_color = _darken(spec.color)

        if show_raw_points:
            ax.scatter(
                x_vals,
                y_vals,
                alpha=0.12,
                s=28,
                marker="o",
                color=spec.color,
                edgecolors=edge_color,
                linewidths=0.6,
                label="_nolegend_",
                zorder=2,
            )

        mean_x = float(np.mean(x_vals))
        mean_y = float(np.mean(y_vals))
        std_x = float(np.std(x_vals, ddof=1)) if len(x_vals) > 1 else 0.0
        std_y = float(np.std(y_vals, ddof=1)) if len(y_vals) > 1 else 0.0
        ax.errorbar(
            mean_x,
            mean_y,
            xerr=std_x,
            yerr=std_y,
            fmt="o",
            color=spec.color,
            ecolor=spec.color,
            markerfacecolor=spec.color,
            markeredgecolor=edge_color,
            markeredgewidth=1.2,
            markersize=9,
            capsize=5,
            elinewidth=2.0,
            linewidth=1.8,
            alpha=1.0,
            zorder=4,
            label="_nolegend_",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=spec.color,
                markeredgecolor=edge_color,
                markeredgewidth=1.2,
                markersize=8,
                label=spec.label,
            )
        )
        summary_mean_x.append(mean_x)
        summary_std_x.append(std_x)
        summary_mean_y.append(mean_y)
        summary_std_y.append(std_y)

        all_x.append(x_vals)
        all_y.append(y_vals)
        total_points += len(x_vals)
        any_series = True

    if not any_series:
        ax.text(0.5, 0.5, "No plottable mitigation/benchmark pairs", ha="center", va="center", transform=ax.transAxes)
        return fig, DeltaStats(0)

    # Axis limits: explicit fixed limits first; fallback to budget defaults.
    if fixed_xlim is not None and fixed_ylim is not None:
        ax.set_xlim(*fixed_xlim)
        ax.set_ylim(*fixed_ylim)
    elif budget.lower() == "cv" and clamp_cv_axes:
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
    elif budget.lower() == "holdout" and clamp_cv_axes:
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
    else:
        # Sniper-style scaling: include full error-bar extents plus margin,
        # while staying centered on zero for direct directional comparison.
        if summary_mean_x:
            half_x = max(abs(mx) + sx for mx, sx in zip(summary_mean_x, summary_std_x))
            half_y = max(abs(my) + sy for my, sy in zip(summary_mean_y, summary_std_y))
        else:
            x_all = np.concatenate(all_x)
            y_all = np.concatenate(all_y)
            half_x = max(abs(float(np.nanmin(x_all))), abs(float(np.nanmax(x_all))))
            half_y = max(abs(float(np.nanmin(y_all))), abs(float(np.nanmax(y_all))))
        half_x = max(half_x * (1 + pad), 0.01)
        half_y = max(half_y * (1 + pad), 0.01)
        ax.set_xlim(-half_x, half_x)
        ax.set_ylim(-half_y, half_y)

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    if x_lo < 0.0 and y_lo < 0.0:
        quadrant = Rectangle(
            (x_lo, y_lo),
            0.0 - x_lo,
            0.0 - y_lo,
            facecolor="#dcefd9",
            edgecolor="none",
            alpha=0.45,
            zorder=0,
        )
        ax.add_patch(quadrant)
        ax.text(
            x_lo + 0.07 * (0.0 - x_lo),
            y_lo + 0.16 * (0.0 - y_lo),
            "Less overtuning,\nbetter performance",
            ha="left",
            va="bottom",
            fontsize=11,
            color="#355b35",
            fontweight="semibold",
            zorder=1,
        )

    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    overtuning_label = "normalized overtuning" if overtuning_col.startswith("normalized_") else "overtuning"
    test_loss_label = "normalized test loss" if test_loss_col.startswith("normalized_") else "test loss"
    ax.set_xlabel(f"Δ {overtuning_label}", fontsize=16)
    ax.set_ylabel(f"Δ {test_loss_label}", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.3)

    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=12, framealpha=0.88, ncol=1)

    if ax is None:
        fig.tight_layout()
    return fig, DeltaStats(total_points)
