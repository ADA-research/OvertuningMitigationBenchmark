"""Autorank critical difference diagram combining CV and holdout methods."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from ..filters import final_rows_per_run
from ..methods import (
    BERGMAN_LABEL,
    BERGMAN_RESHUFFLING_LABEL,
    BASELINE_LABEL,
    MLPLAN_LABEL,
    method_color,
    combined_benchmark_mask,
    combined_bergman_mask,
    combined_bergman_reshuffling_mask,
    combined_reshuffling_mask,
    combined_thresholdout_mask,
    makarova_mask,
    mlplan_mask,
    one_se_mask,
    post_hoc_ensemble_mask,
    post_hoc_surrogate_mask,
    selection_set_mask,
)


@dataclass
class CDDiagramStats:
    n_methods: int
    n_runs: int
    n_dropped_runs: int
    n_unique_datasets: int


_PAIR_COLS = ["dataset_id", "optimizer", "repetition", "outer_fold"]
_CD_BERGMAN_RESHUFFLING_LABEL = BERGMAN_RESHUFFLING_LABEL.replace("\n", " ")


def _pair_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in _PAIR_COLS if c in df.columns]


def _normalize_dataset_id(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)


def _base_method_label(label: str) -> str:
    text = str(label).strip()
    if not text:
        return ""
    lower = text.lower()
    if "baseline" in lower or "benchmark" in lower:
        return BASELINE_LABEL
    if text.startswith("Selection set"):
        return "Selection set"
    if text.startswith("1SE rule"):
        return "1SE rule"
    if text.startswith("Surrogate mean"):
        return "Surrogate mean"
    if text.startswith("Post-hoc ensembling"):
        return "Post-hoc ensembling"
    if text.startswith("Makarova"):
        return "Makarova"
    if text.startswith(MLPLAN_LABEL):
        return MLPLAN_LABEL
    if text.startswith("Reshuffling"):
        return "Reshuffling"
    if text.startswith(_CD_BERGMAN_RESHUFFLING_LABEL):
        return _CD_BERGMAN_RESHUFFLING_LABEL
    if text.startswith(BERGMAN_RESHUFFLING_LABEL):
        return _CD_BERGMAN_RESHUFFLING_LABEL
    if text.startswith(BERGMAN_LABEL):
        return BERGMAN_LABEL
    if text.startswith("Thresholdout"):
        return "Thresholdout"
    return ""


def _color_for_method_label(label: str) -> str | None:
    base = _base_method_label(label)
    if not base:
        return None
    return method_color(base, default="#202020")


def _apply_cd_method_colors(ax: plt.Axes) -> None:
    # Keep text labels black and color only method connector lines so the
    # CD diagram matches the pipeline palette without changing label text color.
    method_y_to_color: dict[tuple[str, float], str] = {}
    for txt in ax.texts:
        color = _color_for_method_label(txt.get_text())
        if not color:
            continue
        txt.set_color("black")
        method_y_to_color[(txt.get_ha(), round(float(txt.get_position()[1]), 4))] = color

    for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick.set_color("black")

    for line in ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()
        if len(x) != 3 or len(y) != 3:
            continue
        if line.get_marker() not in {None, "None", ""}:
            continue
        if line.get_linewidth() > 1.0:
            continue

        # Autorank method connectors are 3-point polylines ending on the
        # label anchors at x ~= 0.15 (left) or x ~= 0.85 (right).
        endpoint_side: str | None = None
        endpoint_y: float | None = None
        if abs(float(x[-1]) - 0.15) < 0.03:
            endpoint_side = "right"
            endpoint_y = float(y[-1])
        elif abs(float(x[-1]) - 0.85) < 0.03:
            endpoint_side = "left"
            endpoint_y = float(y[-1])
        elif abs(float(x[0]) - 0.15) < 0.03:
            endpoint_side = "right"
            endpoint_y = float(y[0])
        elif abs(float(x[0]) - 0.85) < 0.03:
            endpoint_side = "left"
            endpoint_y = float(y[0])

        if endpoint_side is None or endpoint_y is None:
            continue

        color = method_y_to_color.get((endpoint_side, round(endpoint_y, 4)))
        if color:
            line.set_color(color)


def _method_delta_per_run(
    final_df: pd.DataFrame,
    method_mask: pd.Series,
    bench_mask: pd.Series,
    method_score_col: str,
    benchmark_score_col: str,
    label: str,
    comparison_unit: Literal["run", "dataset"],
) -> tuple[pd.Series | None, dict[str, int], pd.DataFrame, pd.DataFrame]:
    """Return per-run delta Series (method - benchmark) and inclusion counts."""
    counts = {
        "method_rows": 0,
        "benchmark_rows": 0,
        "paired_rows": 0,
        "paired_runs": 0,
    }
    empty_dataset_diag = pd.DataFrame(
        columns=[
            "dataset_id",
            "method",
            "method_score_column",
            "benchmark_score_column",
            "method_rows",
            "benchmark_rows",
            "paired_runs",
        ]
    )
    empty_missing_keys = pd.DataFrame(
        columns=[
            "method",
            "method_score_column",
            "benchmark_score_column",
            "missing_kind",
            "dataset_id",
            "optimizer",
            "repetition",
            "outer_fold",
            "model",
        ]
    )
    pair_cols = _pair_cols(final_df)
    if comparison_unit != "dataset" and not pair_cols:
        return None, counts, empty_dataset_diag, empty_missing_keys

    bench_df = final_df.loc[bench_mask, pair_cols + [benchmark_score_col]].copy()
    method_df = final_df.loc[method_mask, pair_cols + [method_score_col]].copy()
    counts["method_rows"] = int(len(method_df))
    counts["benchmark_rows"] = int(len(bench_df))

    if bench_df.empty or method_df.empty:
        return None, counts, empty_dataset_diag, empty_missing_keys

    if "dataset_id" in bench_df.columns:
        bench_df["dataset_id"] = _normalize_dataset_id(bench_df["dataset_id"])
        method_df["dataset_id"] = _normalize_dataset_id(method_df["dataset_id"])

    bench_df = bench_df.rename(columns={benchmark_score_col: "bench_score"})
    method_df = method_df.rename(columns={method_score_col: "method_score"})

    if comparison_unit == "dataset":
        if "dataset_id" not in method_df.columns or "dataset_id" not in bench_df.columns:
            return None, counts, empty_dataset_diag, empty_missing_keys
        # Dataset-level CD input: average method and benchmark independently
        # over optimizer/repetition/fold, then compute one delta per dataset.
        method_dataset_mean = (
            method_df.groupby("dataset_id", dropna=False)["method_score"]
            .mean()
            .rename("method_score")
        )
        bench_dataset_mean = (
            bench_df.groupby("dataset_id", dropna=False)["bench_score"]
            .mean()
            .rename("bench_score")
        )
        paired_dataset = pd.concat([method_dataset_mean, bench_dataset_mean], axis=1, join="inner").dropna()
        if paired_dataset.empty:
            return None, counts, empty_dataset_diag, empty_missing_keys

        per_series = (paired_dataset["method_score"] - paired_dataset["bench_score"]).rename(label)
        counts["paired_rows"] = int(len(paired_dataset))

        method_ds = (
            method_df.groupby("dataset_id", dropna=False)
            .size()
            .rename("method_rows")
            .reset_index()
        )
        bench_ds = (
            bench_df.groupby("dataset_id", dropna=False)
            .size()
            .rename("benchmark_rows")
            .reset_index()
        )
        paired_ds = pd.DataFrame({
            "dataset_id": paired_dataset.index,
            "paired_runs": 1,
        })
        dataset_diag = method_ds.merge(bench_ds, on="dataset_id", how="outer").merge(
            paired_ds,
            on="dataset_id",
            how="outer",
        )
        dataset_diag[["method_rows", "benchmark_rows", "paired_runs"]] = dataset_diag[
            ["method_rows", "benchmark_rows", "paired_runs"]
        ].fillna(0).astype(int)
        dataset_diag["method"] = label
        dataset_diag["method_score_column"] = method_score_col
        dataset_diag["benchmark_score_column"] = benchmark_score_col

        missing_method_ds = (
            pd.DataFrame({"dataset_id": bench_dataset_mean.index})
            .merge(pd.DataFrame({"dataset_id": method_dataset_mean.index}), on="dataset_id", how="left", indicator=True)
            .loc[lambda d: d["_merge"].eq("left_only"), ["dataset_id"]]
            .copy()
        )
        missing_method_ds["missing_kind"] = "missing_method_for_benchmark_run"

        missing_benchmark_ds = (
            pd.DataFrame({"dataset_id": method_dataset_mean.index})
            .merge(pd.DataFrame({"dataset_id": bench_dataset_mean.index}), on="dataset_id", how="left", indicator=True)
            .loc[lambda d: d["_merge"].eq("left_only"), ["dataset_id"]]
            .copy()
        )
        missing_benchmark_ds["missing_kind"] = "missing_benchmark_for_method_run"

        missing_keys = pd.concat([missing_method_ds, missing_benchmark_ds], ignore_index=True)
        if not missing_keys.empty:
            for col in [c for c in pair_cols if c != "dataset_id"]:
                missing_keys[col] = pd.NA
            missing_keys["method"] = label
            missing_keys["method_score_column"] = method_score_col
            missing_keys["benchmark_score_column"] = benchmark_score_col
            missing_keys = missing_keys[
                [
                    "method",
                    "method_score_column",
                    "benchmark_score_column",
                    "missing_kind",
                    *pair_cols,
                ]
            ]
        else:
            missing_keys = empty_missing_keys

        counts["paired_runs"] = int(len(per_series))
        return per_series, counts, dataset_diag, missing_keys

    paired = method_df.merge(bench_df, on=pair_cols, how="inner")
    if paired.empty:
        return None, counts, empty_dataset_diag, empty_missing_keys

    paired["delta"] = paired["method_score"] - paired["bench_score"]
    counts["paired_rows"] = int(len(paired))

    # Keep run-level comparisons (no pre-averaging by dataset).
    per_series = (
        paired.groupby(pair_cols, dropna=False)["delta"]
        .mean()
        .rename(label)
    )

    counts["paired_runs"] = int(len(per_series))

    method_ds = (
        method_df.groupby("dataset_id", dropna=False)
        .size()
        .rename("method_rows")
        .reset_index()
    )
    bench_ds = (
        bench_df.groupby("dataset_id", dropna=False)
        .size()
        .rename("benchmark_rows")
        .reset_index()
    )
    paired_ds = (
        paired.groupby("dataset_id", dropna=False)
        .size()
        .rename("paired_runs")
        .reset_index()
    )
    dataset_diag = method_ds.merge(bench_ds, on="dataset_id", how="outer").merge(
        paired_ds,
        on="dataset_id",
        how="outer",
    )
    dataset_diag[["method_rows", "benchmark_rows", "paired_runs"]] = dataset_diag[
        ["method_rows", "benchmark_rows", "paired_runs"]
    ].fillna(0).astype(int)
    dataset_diag["method"] = label
    dataset_diag["method_score_column"] = method_score_col
    dataset_diag["benchmark_score_column"] = benchmark_score_col

    bench_keys = bench_df[pair_cols].drop_duplicates()
    method_keys = method_df[pair_cols].drop_duplicates()

    missing_method = (
        bench_keys.merge(method_keys, on=pair_cols, how="left", indicator=True)
        .loc[lambda d: d["_merge"].eq("left_only"), pair_cols]
        .copy()
    )
    missing_method["missing_kind"] = "missing_method_for_benchmark_run"

    missing_benchmark = (
        method_keys.merge(bench_keys, on=pair_cols, how="left", indicator=True)
        .loc[lambda d: d["_merge"].eq("left_only"), pair_cols]
        .copy()
    )
    missing_benchmark["missing_kind"] = "missing_benchmark_for_method_run"

    missing_keys = pd.concat([missing_method, missing_benchmark], ignore_index=True)
    if not missing_keys.empty:
        missing_keys["method"] = label
        missing_keys["method_score_column"] = method_score_col
        missing_keys["benchmark_score_column"] = benchmark_score_col
        missing_keys = missing_keys[
            [
                "method",
                "method_score_column",
                "benchmark_score_column",
                "missing_kind",
                *pair_cols,
            ]
        ]

    return per_series, counts, dataset_diag, missing_keys


def _build_method_specs(
    final_df: pd.DataFrame,
) -> list[tuple[str, pd.Series, pd.Series, str, str]]:
    """Return list of (label, method_mask, bench_mask, method_score_col, benchmark_score_col) tuples."""
    cv_bench = combined_benchmark_mask(final_df, "cv")
    ho_bench = combined_benchmark_mask(final_df, "holdout")
    common_bench = ho_bench
    common_bench_col = "retrain_test_loss"

    specs: list[tuple[str, pd.Series, pd.Series, str, str]] = [
        # -- Common comparator method --
        ("Baseline (holdout)", common_bench, common_bench, "retrain_test_loss", common_bench_col),
        # -- CV baseline and CV methods (ensemble score; no CV retrain path) --
        ("Baseline (CV)", cv_bench, common_bench, "ensembled_test_loss", common_bench_col),
        ("Selection set (CV)", selection_set_mask(final_df, "cv"), common_bench, "ensembled_test_loss", common_bench_col),
        ("1SE rule", one_se_mask(final_df, "cv"), common_bench, "ensembled_test_loss", common_bench_col),
        ("Surrogate mean (CV)", post_hoc_surrogate_mask(final_df, "cv"), common_bench, "ensembled_test_loss", common_bench_col),
        ("Post-hoc ensembling (CV)", post_hoc_ensemble_mask(final_df, "cv"), common_bench, "ensembled_test_loss", common_bench_col),
        ("Makarova", makarova_mask(final_df, "cv"), common_bench, "ensembled_test_loss", common_bench_col),
        (MLPLAN_LABEL, mlplan_mask(final_df), common_bench, "ensembled_test_loss", common_bench_col),
        ("Reshuffling (CV)", combined_reshuffling_mask(final_df, "cv"), common_bench, "ensembled_test_loss", common_bench_col),
        (BERGMAN_LABEL, combined_bergman_mask(final_df, "cv"), common_bench, "ensembled_test_loss", common_bench_col),
        (_CD_BERGMAN_RESHUFFLING_LABEL, combined_bergman_reshuffling_mask(final_df, "cv"), common_bench, "ensembled_test_loss", common_bench_col),
        # -- Holdout methods (retrain test loss) --
        ("Selection set (holdout)", selection_set_mask(final_df, "holdout"), common_bench, "retrain_test_loss", common_bench_col),
        ("Surrogate mean (holdout)", post_hoc_surrogate_mask(final_df, "holdout"), common_bench, "retrain_test_loss", common_bench_col),
        ("Post-hoc ensembling (holdout)", post_hoc_ensemble_mask(final_df, "holdout"), common_bench, "retrain_test_loss", common_bench_col),
        ("Reshuffling (holdout)", combined_reshuffling_mask(final_df, "holdout"), common_bench, "retrain_test_loss", common_bench_col),
        ("Thresholdout", combined_thresholdout_mask(final_df, "holdout"), common_bench, "retrain_test_loss", common_bench_col),
    ]

    return specs


def build_cd_diagram_dataframe(
    trajectories: pd.DataFrame,
    comparison_unit: Literal["run", "dataset"] = "run",
) -> tuple[pd.DataFrame, CDDiagramStats, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a wide-format DataFrame for autorank critical difference diagram.

    Each column is a method; each row is one paired run-context. Values are
    delta test loss (method - benchmark), lower is better.

    Run-contexts where any method has no data are dropped.
    """
    final_df = final_rows_per_run(trajectories)
    if final_df.empty:
        return pd.DataFrame(), CDDiagramStats(0, 0, 0, 0), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    specs = _build_method_specs(final_df)

    series_list: list[pd.Series] = []
    method_count_rows: list[dict[str, object]] = []
    dataset_diag_frames: list[pd.DataFrame] = []
    missing_key_frames: list[pd.DataFrame] = []
    for label, method_mask, bench_mask, method_score_col, benchmark_score_col in specs:
        s, counts, dataset_diag, missing_keys = _method_delta_per_run(
            final_df,
            method_mask,
            bench_mask,
            method_score_col,
            benchmark_score_col,
            label,
            comparison_unit,
        )
        method_count_rows.append(
            {
                "method": label,
                "method_score_column": method_score_col,
                "benchmark_score_column": benchmark_score_col,
                "comparison_unit": comparison_unit,
                **counts,
            }
        )
        if not dataset_diag.empty:
            dataset_diag_frames.append(dataset_diag)
        if not missing_keys.empty:
            missing_key_frames.append(missing_keys)
        if s is not None and not s.empty:
            series_list.append(s)

    dataset_counts_df = (
        pd.concat(dataset_diag_frames, ignore_index=True)
        if dataset_diag_frames
        else pd.DataFrame(
            columns=[
                "dataset_id",
                "method",
                "method_score_column",
                "benchmark_score_column",
                "method_rows",
                "benchmark_rows",
                "paired_runs",
            ]
        )
    )
    if not dataset_counts_df.empty:
        dataset_counts_df["missing_vs_benchmark"] = (
            dataset_counts_df["benchmark_rows"] - dataset_counts_df["paired_runs"]
        )

    missing_keys_df = (
        pd.concat(missing_key_frames, ignore_index=True)
        if missing_key_frames
        else pd.DataFrame(
            columns=[
                "method",
                "method_score_column",
                "benchmark_score_column",
                "missing_kind",
                "dataset_id",
                "optimizer",
                "repetition",
                "outer_fold",
                "model",
            ]
        )
    )

    if not series_list:
        counts_df = pd.DataFrame(method_count_rows)
        return pd.DataFrame(), CDDiagramStats(0, 0, 0, 0), counts_df, dataset_counts_df, missing_keys_df

    wide = pd.concat(series_list, axis=1)
    total_rows = len(wide)
    wide = wide.dropna()
    dropped = total_rows - len(wide)

    counts_df = pd.DataFrame(method_count_rows)
    if not counts_df.empty:
        complete_case_runs = int(len(wide))
        counts_df["complete_case_runs"] = complete_case_runs
    if not dataset_counts_df.empty:
        dataset_counts_df["complete_case_runs"] = int(len(wide))

    n_unique_datasets = 0
    if "dataset_id" in wide.index.names:
        n_unique_datasets = int(pd.Index(wide.index.get_level_values("dataset_id")).nunique())

    stats = CDDiagramStats(
        n_methods=len(wide.columns),
        n_runs=len(wide),
        n_dropped_runs=dropped,
        n_unique_datasets=n_unique_datasets,
    )
    return wide, stats, counts_df, dataset_counts_df, missing_keys_df


def plot_critical_difference_diagram(
    trajectories: pd.DataFrame,
    comparison_unit: Literal["run", "dataset"] = "run",
    alpha: float = 0.05,
    ax: plt.Axes | None = None,
    title: str | None = None,
    save_wide_csv_path: str | None = None,
) -> tuple[plt.Figure, CDDiagramStats, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Plot an autorank critical difference diagram combining CV and holdout methods.

    Parameters
    ----------
    trajectories:
        Full trajectory dataframe (all sources merged, with ``trajectory_source`` column).
    comparison_unit:
        Unit used to construct one paired observation for autorank.
        - ``"run"``: one row per run-context key
        - ``"dataset"``: one row per dataset (dataset-averaged deltas)
    alpha:
        Significance level for the statistical test.
    ax:
        Axes to draw on. If None, a new figure is created.
    title:
        Optional title for the axes.
    """
    from autorank import autorank, plot_stats

    wide, stats, counts_df, dataset_counts_df, missing_keys_df = build_cd_diagram_dataframe(
        trajectories,
        comparison_unit=comparison_unit,
    )

    if wide.empty or stats.n_methods < 2 or stats.n_runs < 3:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.figure
        msg = (
            f"Not enough data for CD diagram "
            f"(methods={stats.n_methods}, runs={stats.n_runs})"
        )
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=13)
        if title:
            ax.set_title(title)
        return fig, stats, counts_df, dataset_counts_df, missing_keys_df

    if save_wide_csv_path:
        export_path = Path(save_wide_csv_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(export_path, index=True)

    # autorank expects lower = better (we use delta = method - benchmark, lower = better)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = autorank(wide, alpha=alpha, verbose=False, order="ascending")

    # Keep the chart tight, but give labels slightly more vertical room.
    n_methods = stats.n_methods
    fig_height = max(2.1, 1.0 + n_methods * 0.18)

    if ax is None:
        fig, ax = plt.subplots(figsize=(18, fig_height))
    else:
        fig = ax.figure

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_stats(result, ax=ax)

    # Keep whitespace tight, but avoid clipping/overlap at dense method counts.
    fig.subplots_adjust(left=0.03, right=0.997, bottom=0.10, top=0.97)

    _apply_cd_method_colors(ax)

    if title:
        ax.set_title(title, fontsize=12, pad=2)

    return fig, stats, counts_df, dataset_counts_df, missing_keys_df
