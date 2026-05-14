from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..filters import filter_tunable_runs, final_rows_per_run, keep_baseline_unmitigated_runs


# Three size buckets: small / medium / large
SIZE_CATEGORIES = [
    ("small", 0, 2500),
    ("medium", 2500, 10000),
    ("large", 10000, float("inf")),
]

SIZE_COLORS = {
    "small": "#5e81ac",
    "medium": "#e39a61",
    "large": "#4a7c4e",
}

# Line style encodes problem type; same meaning regardless of size color
PROBLEM_TYPE_STYLES = {
    "binary":     "-",
    "multiclass": "--",
    "regression": ":",
}


@dataclass
class BreakdownCDFStats:
    included_runs: int
    excluded_runs: int
    threshold: float
    plotted_all_runs: int


def _normalize_dataset_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if re.fullmatch(r"\d{6}", text):
        return text
    match = re.search(r"(\d{6})", text)
    return match.group(1) if match else None


def _get_size_category(n_instances: int) -> str:
    if n_instances < 2500:
        return "small"
    if n_instances < 10000:
        return "medium"
    return "large"


def _load_dataset_info(dataset_csv: Path) -> dict[str, dict[str, object]]:
    info: dict[str, dict[str, object]] = {}
    if not dataset_csv.exists():
        return info

    df = pd.read_csv(dataset_csv)
    for _, row in df.iterrows():
        did = str(row["Dataset ID"])
        info[did] = {
            "n_instances": int(row["# Instances"]),
            "type": str(row["Type"]).lower(),
        }
    return info


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values)
    return values, np.arange(1, len(values) + 1) / len(values)


def plot_relative_overtuning_cdf_breakdown(
    trajectories_subset: pd.DataFrame,
    data_dir: Path,
    improvement_threshold: float = 0.001,
    score_col: str = "ensembled_test_loss",
    overtuning_col: str = "relative_ensemble_overtuning",
    max_overtuning: float = 2.0,
) -> tuple[plt.Figure, BreakdownCDFStats]:
    """Recreate the combined size and problem-type CDF breakdown plot."""
    baseline_subset = keep_baseline_unmitigated_runs(trajectories_subset)
    print(f"[cdf_breakdown] Baseline rows considered: {len(baseline_subset)}")

    tunable = filter_tunable_runs(
        baseline_subset,
        score_col=score_col,
        base_threshold=improvement_threshold,
    )

    if tunable.filtered_df.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.text(0.5, 0.5, "No tunable runs after filtering", ha="center", va="center", transform=ax.transAxes)
        return fig, BreakdownCDFStats(tunable.included_runs, tunable.excluded_runs, tunable.median_threshold, 0)

    final_rows = final_rows_per_run(tunable.filtered_df)
    if final_rows.empty or overtuning_col not in final_rows.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.text(0.5, 0.5, f"Missing column: {overtuning_col}", ha="center", va="center", transform=ax.transAxes)
        return fig, BreakdownCDFStats(tunable.included_runs, tunable.excluded_runs, tunable.median_threshold, 0)

    dataset_info = _load_dataset_info(data_dir / "datasets.csv")

    if "dataset_id" in final_rows.columns:
        final_rows = final_rows.copy()
        final_rows["dataset_id_norm"] = final_rows["dataset_id"].map(_normalize_dataset_id)
        final_rows["n_instances"] = final_rows["dataset_id_norm"].map(
            lambda did: dataset_info.get(did, {}).get("n_instances", 0)
        )
        final_rows["size_category"] = final_rows["n_instances"].map(_get_size_category)

        dataset_problem_type = final_rows["dataset_id_norm"].map(
            lambda did: dataset_info.get(did, {}).get("type", "unknown")
        )
        if "problem_type" in final_rows.columns:
            existing = final_rows["problem_type"].astype(str).str.lower()
            final_rows["problem_type"] = np.where(dataset_problem_type != "unknown", dataset_problem_type, existing)
        else:
            final_rows["problem_type"] = dataset_problem_type

    fig, ax = plt.subplots(figsize=(10, 6))

    all_values = pd.to_numeric(final_rows[overtuning_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(all_values) > 0:
        x_vals, y_vals = _cdf(all_values)
        ax.plot(x_vals, y_vals, linewidth=2.5, color="black", linestyle="-", label=f"All (n={len(all_values)})")

    # 9 lines: one per (size_category, problem_type) combination
    # Color = size, line style = problem type
    if "size_category" in final_rows.columns and "problem_type" in final_rows.columns:
        for size_name, _, _ in SIZE_CATEGORIES:
            for problem_type, linestyle in PROBLEM_TYPE_STYLES.items():
                mask = (
                    (final_rows["size_category"] == size_name) &
                    (final_rows["problem_type"] == problem_type)
                )
                values = pd.to_numeric(
                    final_rows.loc[mask, overtuning_col],
                    errors="coerce",
                ).dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                x_vals, y_vals = _cdf(values)
                ax.plot(
                    x_vals,
                    y_vals,
                    linewidth=1.5,
                    color=SIZE_COLORS[size_name],
                    linestyle=linestyle,
                    label=f"{size_name} / {problem_type} (n={len(values)})",
                    alpha=0.85,
                )

    ax.set_xlim(0, max_overtuning)
    ax.set_ylim(0, 1)
    ax.axvline(1.0, linestyle=":")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0, 1.1, 0.1)])
    ax.set_xlabel("Relative Overtuning", fontsize=16)
    ax.set_ylabel("Proportion", fontsize=16)

    total_runs = tunable.included_runs + tunable.excluded_runs
    removed_pct = (100.0 * tunable.excluded_runs / total_runs) if total_runs > 0 else 0.0
    ax.text(
        0.5,
        1.02,
        f"Removed {tunable.excluded_runs} out of {total_runs} ({removed_pct:.1f}%)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
    )

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=12, framealpha=0.95, ncol=2)
    fig.tight_layout()

    return fig, BreakdownCDFStats(
        included_runs=tunable.included_runs,
        excluded_runs=tunable.excluded_runs,
        threshold=tunable.median_threshold,
        plotted_all_runs=len(all_values),
    )
