from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..filters import filter_tunable_runs, final_rows_per_run, run_key_columns
from ..methods import build_mitigation_cdf_methods


@dataclass
class MitigationCDFStats:
    included_runs: int
    excluded_runs: int
    threshold: float
    plotted_series: int


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values)
    return values, np.arange(1, len(values) + 1) / len(values)


def _count_unique_runs(df: pd.DataFrame, run_cols: list[str]) -> int:
    if not run_cols:
        return len(df)
    return int(df.drop_duplicates(subset=run_cols).shape[0])


def _normalize_split_values(df: pd.DataFrame, col: str = "inner_split") -> pd.Series:
    """Return lower-cased split labels robustly across object/categorical dtypes."""
    if col not in df.columns:
        return pd.Series(index=df.index, dtype="string")
    series = df[col]
    # Convert to object before filling so categorical columns cannot raise setitem errors.
    return series.astype("object").where(series.notna(), "").astype(str).str.strip().str.lower()


def _log_filter_stats(
    budget: str,
    original_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    included_runs: int,
    excluded_runs: int,
) -> None:
    total_runs = included_runs + excluded_runs
    pct_excl = 100.0 * excluded_runs / total_runs if total_runs > 0 else 0.0
    print(
        f"[cdf_mitigations] {budget} | Total: {included_runs} included, "
        f"{excluded_runs} excluded ({pct_excl:.1f}% excluded) out of {total_runs} runs"
    )

    if "inner_split" not in original_df.columns:
        return

    run_cols = run_key_columns(original_df)
    original_split = _normalize_split_values(original_df)
    filtered_split = _normalize_split_values(filtered_df) if not filtered_df.empty else None
    for split_name in ["holdout", "cv"]:
        orig_mask = original_split == split_name
        total_split = _count_unique_runs(original_df.loc[orig_mask], run_cols)
        if total_split == 0:
            continue

        if filtered_df.empty or "inner_split" not in filtered_df.columns:
            included_split = 0
        else:
            filt_mask = filtered_split == split_name
            included_split = _count_unique_runs(filtered_df.loc[filt_mask], run_cols)

        excluded_split = total_split - included_split
        pct_excl_split = 100.0 * excluded_split / total_split
        print(
            f"[cdf_mitigations] {budget} |   {split_name}: {included_split} included, "
            f"{excluded_split} excluded ({pct_excl_split:.1f}% excluded) out of {total_split} runs"
        )


def plot_relative_overtuning_cdf_mitigations(
    trajectories_subset: pd.DataFrame,
    budget: str,
    improvement_threshold: float = 0.001,
    score_col: str = "ensembled_test_loss",
    overtuning_col: str = "relative_ensemble_overtuning",
    max_overtuning: float = 2.0,
    method_builder=build_mitigation_cdf_methods,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
    panel_title: str | None = None,
) -> tuple[plt.Figure, MitigationCDFStats]:
    """Recreate budget-specific mitigation CDF plots from the legacy implementation.

    This function expects a pre-selected subset dataframe and then applies
    the legacy tunable-run filter.
    """
    tunable = filter_tunable_runs(
        trajectories_subset,
        score_col=score_col,
        base_threshold=improvement_threshold,
    )

    _log_filter_stats(budget, trajectories_subset, tunable.filtered_df, tunable.included_runs, tunable.excluded_runs)

    if tunable.filtered_df.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5.1))
        else:
            fig = ax.figure
            ax.clear()
        ax.text(0.5, 0.5, "No tunable runs after filtering", ha="center", va="center", transform=ax.transAxes)
        return fig, MitigationCDFStats(tunable.included_runs, tunable.excluded_runs, tunable.median_threshold, 0)

    final_rows = final_rows_per_run(tunable.filtered_df)
    if final_rows.empty or overtuning_col not in final_rows.columns:
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5.1))
        else:
            fig = ax.figure
            ax.clear()
        ax.text(0.5, 0.5, f"Missing column: {overtuning_col}", ha="center", va="center", transform=ax.transAxes)
        return fig, MitigationCDFStats(tunable.included_runs, tunable.excluded_runs, tunable.median_threshold, 0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5.1))
    else:
        fig = ax.figure
        ax.clear()
    plotted = 0

    for spec, mask in method_builder(final_rows, budget):
        values = pd.to_numeric(final_rows.loc[mask, overtuning_col], errors="coerce").dropna().to_numpy(dtype=float)
        n_matched = int(mask.sum())
        n_valid = len(values)
        if n_valid == 0:
            print(f"[cdf_mitigations] {budget} | {spec.label}: {n_matched} matched, 0 with valid overtuning — skipping")
            continue
        print(f"[cdf_mitigations] {budget} | {spec.label}: {n_matched} matched, {n_valid} with valid overtuning")
        x_vals, y_vals = _cdf(values)
        ax.plot(x_vals, y_vals, linewidth=2.0, color=spec.color, label=spec.label)
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No plottable mitigation series", ha="center", va="center", transform=ax.transAxes)
        return fig, MitigationCDFStats(tunable.included_runs, tunable.excluded_runs, tunable.median_threshold, 0)

    ax.set_xlim(0, max_overtuning)
    ax.set_ylim(0, 1)
    ax.axvline(1.0, linestyle=":", color="black")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0, 1.1, 0.1)])
    ax.set_xlabel("Relative Overtuning", fontsize=16)
    ax.set_ylabel("Proportion", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)

    ax.grid(True, alpha=0.3, linestyle="--")
    if show_legend:
        ax.legend(loc="lower right", fontsize=12, framealpha=0.95)
    if ax is None:
        fig.tight_layout()

    return fig, MitigationCDFStats(
        included_runs=tunable.included_runs,
        excluded_runs=tunable.excluded_runs,
        threshold=tunable.median_threshold,
        plotted_series=plotted,
    )
