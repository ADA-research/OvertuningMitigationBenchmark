from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Polygon
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..methods import method_color


ALPHA = 0.05
HIGHER_IS_BETTER = False
MATRIX_CORRECTION = "bonferroni"


@dataclass(frozen=True)
class PairwiseResult:
    method_a: str
    method_b: str
    raw_pvalue: float
    better_method: str
    median_difference: float
    method_a_wins: int
    ties: int
    method_b_wins: int
    corrected_pvalue: float | None = None


def choose_better_method(data: pd.DataFrame, method_a: str, method_b: str) -> tuple[str, float]:
    difference = data[method_a] - data[method_b]
    direction = float(difference.median())
    if direction == 0:
        direction = float(difference.mean())

    if direction == 0:
        return "tie", direction
    if HIGHER_IS_BETTER:
        return (method_a if direction > 0 else method_b), direction
    return (method_a if direction < 0 else method_b), direction


def count_pairwise_outcomes(
    data: pd.DataFrame, method_a: str, method_b: str
) -> tuple[int, int, int]:
    differences = data[method_a] - data[method_b]
    ties = int((differences == 0).sum())
    if HIGHER_IS_BETTER:
        method_a_wins = int((differences > 0).sum())
        method_b_wins = int((differences < 0).sum())
    else:
        method_a_wins = int((differences < 0).sum())
        method_b_wins = int((differences > 0).sum())
    return method_a_wins, ties, method_b_wins


def compute_pairwise_pvalue(data: pd.DataFrame, method_a: str, method_b: str) -> float:
    differences = data[method_a] - data[method_b]
    nonzero_differences = differences[differences != 0]
    num_trials = len(nonzero_differences)
    if num_trials == 0:
        return 1.0
    num_positive = int((nonzero_differences > 0).sum())
    return stats.binomtest(num_positive, n=num_trials, p=0.5, alternative="two-sided").pvalue


def compute_pairwise_tests(data: pd.DataFrame) -> list[PairwiseResult]:
    results: list[PairwiseResult] = []
    methods = list(data.columns)
    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1 :]:
            pvalue = compute_pairwise_pvalue(data, method_a, method_b)
            better_method, median_difference = choose_better_method(data, method_a, method_b)
            method_a_wins, ties, method_b_wins = count_pairwise_outcomes(data, method_a, method_b)
            results.append(
                PairwiseResult(
                    method_a,
                    method_b,
                    float(pvalue),
                    better_method,
                    median_difference,
                    method_a_wins,
                    ties,
                    method_b_wins,
                )
            )
    return results


def add_corrected_pvalues(pairwise_results: list[PairwiseResult], correction: str) -> list[PairwiseResult]:
    raw_pvalues = [result.raw_pvalue for result in pairwise_results]
    _, corrected_pvalues, _, _ = multipletests(raw_pvalues, alpha=ALPHA, method=correction)

    corrected_results: list[PairwiseResult] = []
    for result, corrected_pvalue in zip(pairwise_results, corrected_pvalues):
        corrected_results.append(
            PairwiseResult(
                result.method_a,
                result.method_b,
                result.raw_pvalue,
                result.better_method,
                result.median_difference,
                result.method_a_wins,
                result.ties,
                result.method_b_wins,
                float(corrected_pvalue),
            )
        )
    return corrected_results


def sort_methods_by_average_rank(data: pd.DataFrame) -> list[str]:
    ranks = data.rank(axis="columns", ascending=not HIGHER_IS_BETTER)
    return ranks.mean().sort_values(ascending=True).index.to_list()


def format_pvalue(pvalue: float) -> str:
    if pvalue < 0.001:
        return "<0.001"
    return f"{pvalue:.3f}"


def plot_pairwise_wilcoxon_matrix(
    data: pd.DataFrame,
    output_path: Path,
) -> tuple[plt.Figure, plt.Axes]:
    raw_pairwise_results = compute_pairwise_tests(data)
    corrected_results = add_corrected_pvalues(raw_pairwise_results, MATRIX_CORRECTION)

    methods_x = sort_methods_by_average_rank(data)
    methods_y = list(reversed(methods_x))
    value_matrix = np.full((len(methods_y), len(methods_x)), np.nan, dtype=float)
    annotation_matrix = [["" for _ in methods_x] for _ in methods_y]

    result_lookup = {
        frozenset((result.method_a, result.method_b)): result
        for result in corrected_results
    }

    n_methods = len(methods_x)
    for row, method_row in enumerate(methods_y):
        for col, method_col in enumerate(methods_x):
            if row + col < (n_methods - 1):
                continue

            if method_row == method_col:
                value_matrix[row, col] = 0
                annotation_matrix[row][col] = "-"
                continue

            result = result_lookup[frozenset((method_row, method_col))]
            corrected_p = float(result.corrected_pvalue)
            value_matrix[row, col] = 1 if corrected_p < ALPHA else 2
            annotation_matrix[row][col] = format_pvalue(corrected_p)

    cmap = ListedColormap(["#ffffff", "#d0d0d0", "#595959"])
    cmap.set_bad("#ffffff")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    width = max(12.5, 1.00 * len(methods_x))
    height = max(5.5, 0.35 * len(methods_y))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(value_matrix, cmap=cmap, norm=norm)
    ax.set_aspect("auto")
    ax.set_xticks(range(len(methods_x)), labels=methods_x, rotation=30, ha="right")
    ax.set_yticks(range(len(methods_y)), labels=methods_y)

    for row in range(len(methods_y)):
        for col in range(len(methods_x)):
            text_color = "#ffffff" if value_matrix[row, col] == 2 else "#1a1a1a"
            ax.text(
                col,
                row,
                annotation_matrix[row][col],
                ha="center",
                va="center",
                fontsize=11.0,
                color=text_color,
            )

    for tick in ax.get_xticklabels():
        tick.set_color(method_color(str(tick.get_text()), default="#202020"))
    for tick in ax.get_yticklabels():
        tick.set_color(method_color(str(tick.get_text()), default="#202020"))

    ax.set_xticks(np.arange(len(methods_x) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(methods_y) + 1) - 0.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add subtle horizontal guides only in the intentionally empty top-left half.
    n_methods = len(methods_x)
    for row in range(len(methods_y)):
        x_end = n_methods - row - 1.5
        if x_end <= -0.5:
            continue
        ax.hlines(row + 0.5, -0.5, x_end, colors="#d6d6d6", linewidth=0.6, zorder=2.6)
    ax.tick_params(axis="both", labelsize=13)

    # Keep the intentionally unused top-left half visually blank.
    n = len(methods_x)
    upper_triangle_mask = Polygon(
        [(-0.5, -0.5), (n - 0.5, -0.5), (-0.5, n - 0.5)],
        closed=True,
        facecolor="white",
        edgecolor="none",
        zorder=2.5,
    )
    ax.add_patch(upper_triangle_mask)

    # Remove hard outer edges on the empty side for a cleaner triangular look.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    return fig, ax