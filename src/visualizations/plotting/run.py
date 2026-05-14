import argparse
from datetime import datetime
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from .plots.cdf_mitigations import plot_relative_overtuning_cdf_mitigations
from .plots.delta_scatter import plot_relative_overtuning_delta_scatter
from .plots.critical_difference import plot_critical_difference_diagram
from .plots.critical_difference import build_cd_diagram_dataframe
from .plots.ranking_all_methods import plot_all_methods_average_rank_trajectory
from .plots.pairwise_wilcoxon_matrix import plot_pairwise_wilcoxon_matrix
from .plots.trajectories_all_methods import plot_all_methods_normalized_test_error_trajectory
from .plots.trajectories_per_dataset import plot_benchmark_dataset_trajectory
from .filters import apply_subset_filters, final_rows_per_run, normalize_dataset_id_value
from .io import load_trajectory_source, load_trajectory_sources
from .methods import (
    BERGMAN_LABEL,
    BERGMAN_RESHUFFLING_LABEL,
    MLPLAN_LABEL,
    build_all_methods_cdf,
    build_all_methods_cdf_with_benchmark,
    build_all_methods_delta,
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

import warnings

warnings.filterwarnings("ignore")


PLOTTING_USECOLS = [
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
    "iteration",
    "problem_type",
    "ensemble_overtuning",
    "retrain_overtuning",
    "relative_ensemble_overtuning",
    "relative_retrain_overtuning",
    "normalized_ensemble_overtuning",
    "normalized_retrain_overtuning",
    "ensembled_test_loss",
    "retrain_test_loss",
    "normalized_ensembled_test_loss",
    "normalized_retrain_test_loss",
]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all-method overtuning plots")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("src/visualizations/data"),
        help="Directory containing trajectories CSV files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("src/visualizations/results"),
        help="Parent directory where timestamped plotting_results_* folders are created",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Optional dataset filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--optimizer",
        action="append",
        default=[],
        help="Optional optimizer filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--problem-type",
        action="append",
        default=[],
        help="Optional problem_type filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--mitigation",
        action="append",
        default=[],
        help="Optional mitigation filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--inner-split",
        action="append",
        default=[],
        help="Optional inner_split filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.001,
        help="Base threshold used to filter non-tunable runs",
    )
    parser.add_argument(
        "--per-dataset",
        action="store_true",
        default=False,
        help="Generate per-dataset plot outputs (slow; disabled by default)",
    )
    parser.add_argument(
        "--only-pairwise-matrix",
        action="store_true",
        default=False,
        help="Only generate the all-datasets pairwise Wilcoxon matrix and exit",
    )
    parser.add_argument(
        "--only-cv-retrain-plots",
        action="store_true",
        default=False,
        help="Only generate the single-panel CV-with-retrain CDF and delta scatter plots for all_datasets and all_small_binary",
    )
    parser.add_argument(
        "--only-model-score-cd",
        action="store_true",
        default=False,
        help="Only generate model+score critical difference plots for large/small dataset groups using CV baseline rows",
    )
    parser.add_argument(
        "--only-benchmark-aggregate-selected",
        action="store_true",
        default=False,
        help="Only generate the three selected benchmark aggregate trajectory plots and exit",
    )
    parser.add_argument(
        "--only-appendix-combined-plots",
        action="store_true",
        default=False,
        help="Only generate the appendix combined normalized delta-scatter plots and exit",
    )
    parser.add_argument(
        "--only-appendix-optimizer-plots",
        action="store_true",
        default=False,
        help="Only generate appendix normalized delta-scatter plots split by problem type x optimizer and exit",
    )
    parser.add_argument(
        "--only-appendix-cdf-plots",
        action="store_true",
        default=False,
        help="Only generate appendix CDF plots split by problem type x ML algorithm and problem type x optimizer and exit",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        default=False,
        help="Production mode: remove panel debug overlays (n_datasets | n_runs) from plots",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        default=False,
        help="Partial mode: skip statistical testing plots, appendix plots, and benchmark trajectories; only generate core grouped suite with one dataset and model visualizations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"plotting_results_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_root = output_dir / "grouped"
    grouped_root.mkdir(parents=True, exist_ok=True)
    per_dataset_root = output_dir / "per_dataset"
    per_dataset_root.mkdir(parents=True, exist_ok=True)

    def scatter_cols(budget: str) -> tuple[str, str]:
        """CV uses ensemble (multiple folds), holdout uses retrain (single fold)."""
        if budget == "cv":
            return "ensemble_overtuning", "ensembled_test_loss"
        return "retrain_overtuning", "retrain_test_loss"

    def normalized_scatter_cols(budget: str) -> tuple[str, str]:
        """Normalized delta-scatter columns by budget."""
        if budget == "cv":
            return "normalized_ensemble_overtuning", "normalized_ensembled_test_loss"
        return "normalized_retrain_overtuning", "normalized_retrain_test_loss"

    def cdf_cols(budget: str) -> tuple[str, str]:
        """CV uses ensemble relative overtuning; holdout uses retrain relative overtuning."""
        if budget == "cv":
            return "ensembled_test_loss", "relative_ensemble_overtuning"
        return "retrain_test_loss", "relative_retrain_overtuning"

    def normalized_test_col(budget: str) -> str:
        """Normalized test column for trajectory plots by budget."""
        if budget == "cv":
            return "normalized_ensembled_test_loss"
        return "normalized_retrain_test_loss"

    def raw_test_col(budget: str) -> str:
        """Raw test column by budget."""
        if budget == "cv":
            return "ensembled_test_loss"
        return "retrain_test_loss"

    def _hpo_run_count(df: pd.DataFrame) -> int:
        run_cols = [
            col
            for col in ["dataset_id", "optimizer", "repetition", "outer_fold", "model"]
            if col in df.columns
        ]
        if not run_cols or df.empty:
            return 0
        return int(df[run_cols].drop_duplicates().shape[0])

    def _dataset_count(df: pd.DataFrame) -> int:
        if "dataset_id" not in df.columns or df.empty:
            return 0
        return int(df["dataset_id"].map(normalize_dataset_id_value).nunique())

    def panel_debug_text(full_df: pd.DataFrame, final_df: pd.DataFrame) -> str:
        return f"n_datasets={_dataset_count(full_df)} | n_runs={_hpo_run_count(final_df)}"

    def add_panel_debug(ax: plt.Axes, text: str) -> None:
        if args.prod:
            return
        ax.text(
            0.01,
            1.01,
            text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#4a4a4a",
            fontweight="normal",
        )

    dataset_info_path = args.data_dir / "datasets.csv"
    dataset_info = pd.read_csv(dataset_info_path) if dataset_info_path.exists() else pd.DataFrame()
    source_map = load_trajectory_sources(
        args.data_dir,
        sources=["default", "one_se", "post_hoc_surrogate", "post_hoc_ensemble", "makarova", "mlplan"],
        usecols=PLOTTING_USECOLS,
    )

    incumbent_frames: list[pd.DataFrame] = []
    for incumbent_source_name, incumbent_df in source_map.items():
        filtered_df = apply_subset_filters(
            incumbent_df,
            dataset_ids=args.dataset_id,
            optimizers=args.optimizer,
            problem_types=args.problem_type,
            mitigations=args.mitigation,
            inner_splits=args.inner_split,
        )
        if filtered_df.empty:
            continue
        filtered_df = filtered_df.copy()
        filtered_df["trajectory_source"] = incumbent_source_name
        incumbent_frames.append(filtered_df)

    if not incumbent_frames:
        raise ValueError("No rows available after loading and filtering trajectory sources")
    incumbent_subset = pd.concat(incumbent_frames, ignore_index=True)
    print(f"[plotting] Loaded {len(incumbent_subset)} trajectory rows from {len(source_map)} sources")
    print(f"[plotting] Output directory: {output_dir}")

    final_incumbent_rows = final_rows_per_run(incumbent_subset)

    benchmark_target_name_overrides: dict[str, str] = {
        "363631": "diamonds",
        "363682": "Is-this-a-good-customer",
        "363693": "physiochemical_protein",
    }

    dataset_name_map: dict[str, str] = {}
    if not dataset_info.empty and {"Dataset ID", "Name"}.issubset(dataset_info.columns):
        dataset_name_series = dataset_info["Dataset ID"].map(normalize_dataset_id_value)
        dataset_name_map = {
            dataset_id: str(name)
            for dataset_id, name in zip(dataset_name_series, dataset_info["Name"])
            if dataset_id
        }
    dataset_name_map.update(benchmark_target_name_overrides)

    def _dataset_sort_key(dataset_id: str) -> tuple[int, str]:
        value = str(dataset_id)
        return (0, f"{int(value):09d}") if value.isdigit() else (1, value)

    def _single_line_label(label: str) -> str:
        return str(label).replace("\n", " ")

    def _format_mean_se(values: pd.Series) -> str:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return ""
        mean_val = float(numeric.mean())
        if len(numeric) <= 1:
            se_val = 0.0
        else:
            se_val = float(numeric.std(ddof=1) / (len(numeric) ** 0.5))
        return f"{mean_val:.6f} +- {se_val:.6f}"

    def _summary_method_specs(df: pd.DataFrame) -> list[tuple[str, pd.Series, str, str]]:
        return [
            ("Baseline (CV)", combined_benchmark_mask(df, "cv"), "ensembled_test_loss", "ensemble_overtuning"),
            ("Baseline (holdout)", combined_benchmark_mask(df, "holdout"), "retrain_test_loss", "retrain_overtuning"),
            ("Selection set (CV)", selection_set_mask(df, "cv"), "ensembled_test_loss", "ensemble_overtuning"),
            ("Selection set (holdout)", selection_set_mask(df, "holdout"), "retrain_test_loss", "retrain_overtuning"),
            ("1SE rule", one_se_mask(df, "cv"), "ensembled_test_loss", "ensemble_overtuning"),
            ("Surrogate mean (CV)", post_hoc_surrogate_mask(df, "cv"), "ensembled_test_loss", "ensemble_overtuning"),
            ("Surrogate mean (holdout)", post_hoc_surrogate_mask(df, "holdout"), "retrain_test_loss", "retrain_overtuning"),
            ("Post-hoc ensembling (CV)", post_hoc_ensemble_mask(df, "cv"), "ensembled_test_loss", "ensemble_overtuning"),
            (
                "Post-hoc ensembling (holdout)",
                post_hoc_ensemble_mask(df, "holdout"),
                "retrain_test_loss",
                "retrain_overtuning",
            ),
            ("Makarova", makarova_mask(df, "cv"), "ensembled_test_loss", "ensemble_overtuning"),
            (MLPLAN_LABEL, mlplan_mask(df), "ensembled_test_loss", "ensemble_overtuning"),
            ("Reshuffling (CV)", combined_reshuffling_mask(df, "cv"), "ensembled_test_loss", "ensemble_overtuning"),
            (
                "Reshuffling (holdout)",
                combined_reshuffling_mask(df, "holdout"),
                "retrain_test_loss",
                "retrain_overtuning",
            ),
            (_single_line_label(BERGMAN_LABEL), combined_bergman_mask(df, "cv"), "ensembled_test_loss", "ensemble_overtuning"),
            (
                _single_line_label(BERGMAN_RESHUFFLING_LABEL),
                combined_bergman_reshuffling_mask(df, "cv"),
                "ensembled_test_loss",
                "ensemble_overtuning",
            ),
            ("Thresholdout", combined_thresholdout_mask(df, "holdout"), "retrain_test_loss", "retrain_overtuning"),
        ]

    def _build_metric_table_for_model(
        model_df: pd.DataFrame,
        method_specs: list[tuple[str, pd.Series, str, str]],
        metric_kind: str,
    ) -> pd.DataFrame:
        if model_df.empty or "dataset_id" not in model_df.columns:
            columns = ["dataset_id", "dataset_name", *[spec[0] for spec in method_specs]]
            return pd.DataFrame(columns=columns)

        dataset_ids = sorted(
            model_df["dataset_id"].map(normalize_dataset_id_value).dropna().unique().tolist(),
            key=_dataset_sort_key,
        )

        table_data: dict[str, dict[str, str]] = {
            method_label: {dataset_id: "" for dataset_id in dataset_ids}
            for method_label, _, _, _ in method_specs
        }

        for method_label, method_mask, score_col, overtuning_col in method_specs:
            metric_col = score_col if metric_kind == "performance" else overtuning_col
            if metric_col not in model_df.columns:
                continue

            subset = model_df.loc[method_mask, ["dataset_id", metric_col]].copy()
            if subset.empty:
                continue
            subset["dataset_id_norm"] = subset["dataset_id"].map(normalize_dataset_id_value)
            grouped = subset.groupby("dataset_id_norm", dropna=False)[metric_col]

            for dataset_id, values in grouped:
                if dataset_id in table_data[method_label]:
                    table_data[method_label][dataset_id] = _format_mean_se(values)

        rows: list[dict[str, str]] = []
        for dataset_id in dataset_ids:
            row: dict[str, str] = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name_map.get(dataset_id, dataset_id),
            }
            for method_label, _, _, _ in method_specs:
                row[method_label] = table_data[method_label].get(dataset_id, "")
            rows.append(row)

        return pd.DataFrame(rows)

    model_file_prefix = {
        "FastAIMLP": "fastaimlp",
        "LGBM": "lgbm",
        "RealMLP": "realmlp",
    }

    for model_name, file_prefix in model_file_prefix.items():
        model_df = final_incumbent_rows.loc[
            final_incumbent_rows["model"].astype(str).eq(model_name)
        ].copy()
        method_specs = _summary_method_specs(model_df)

        performance_table = _build_metric_table_for_model(
            model_df,
            method_specs,
            metric_kind="performance",
        )
        performance_csv = output_dir / f"{file_prefix}_performances_over_datasets.csv"
        performance_table.to_csv(performance_csv, index=False)
        print(f"[plotting] [✓] {performance_csv.name} ({len(performance_table)} datasets)")

        overtuning_table = _build_metric_table_for_model(
            model_df,
            method_specs,
            metric_kind="overtuning",
        )
        overtuning_csv = output_dir / f"{file_prefix}_overtuning_over_datasets.csv"
        overtuning_table.to_csv(overtuning_csv, index=False)
        print(f"[plotting] [✓] {overtuning_csv.name} ({len(overtuning_table)} datasets)")

    def generate_group_suite(suite_df: pd.DataFrame, suite_output_dir: Path, skip_pairwise: bool = False) -> None:
        if suite_df.empty:
            print(f"[plotting] Skip empty suite: {suite_output_dir.name}")
            return

        print(f"[plotting] Generating grouped suite: {suite_output_dir.name} ({len(suite_df)} rows)")
        suite_output_dir.mkdir(parents=True, exist_ok=True)
        suite_final_for_scatter = final_rows_per_run(suite_df)

        if suite_output_dir.name in {"all_datasets", "all_small_binary"}:
            generate_cv_retrain_single_plots(suite_df, suite_output_dir)

        if suite_output_dir.name in {"all_datasets", "all_small_datasets"}:
            include_realmlp = suite_output_dir.name == "all_small_datasets"
            generate_model_score_cd_plot(
                suite_df,
                suite_output_dir,
                include_realmlp=include_realmlp,
            )

        fig_cdf, cdf_axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
        for i, budget in enumerate(["cv", "holdout"]):
            score_col, overtuning_col = cdf_cols(budget)
            plot_relative_overtuning_cdf_mitigations(
                suite_df,
                budget=budget,
                improvement_threshold=args.improvement_threshold,
                method_builder=build_all_methods_cdf_with_benchmark,
                score_col=score_col,
                overtuning_col=overtuning_col,
                ax=cdf_axes[i],
                show_legend=True,
                panel_title=None,
            )
            add_panel_debug(
                cdf_axes[i],
                panel_debug_text(
                    suite_df,
                    suite_final_for_scatter,
                ),
            )
        cdf_axes[1].set_ylabel("")
        cdf_axes[1].tick_params(axis="y", left=False, labelleft=False)
        fig_cdf.tight_layout()
        fig_cdf.savefig(suite_output_dir / "relative_overtuning_cdf_cv_holdout.pdf", bbox_inches="tight", dpi=200)
        plt.close(fig_cdf)
        print(f"  [✓] relative_overtuning_cdf_cv_holdout.pdf")

        fig_scatter, scatter_axes = plt.subplots(1, 2, figsize=(14, 5.3))
        for i, budget in enumerate(["cv", "holdout"]):
            ot_col, tl_col = normalized_scatter_cols(budget)
            _, scatter_stats = plot_relative_overtuning_delta_scatter(
                suite_final_for_scatter,
                budget=budget,
                method_builder=build_all_methods_delta,
                benchmark_mask_builder=combined_benchmark_mask,
                show_raw_points=False,
                summary_by_problem_type=False,
                clamp_cv_axes=False,
                overtuning_col=ot_col,
                test_loss_col=tl_col,
                split_method_problem_legend=False,
                ax=scatter_axes[i],
                show_legend=True,
                panel_title=None,
                precomputed_final_rows=suite_final_for_scatter,
                aggregate_per_dataset_first=True,
            )
            add_panel_debug(
                scatter_axes[i],
                panel_debug_text(suite_df, suite_final_for_scatter),
            )
        fig_scatter.tight_layout()
        fig_scatter.savefig(suite_output_dir / "normalized_delta_scatter_cv_holdout.pdf", bbox_inches="tight", dpi=200)
        plt.close(fig_scatter)
        print(f"  [✓] normalized_delta_scatter_cv_holdout.pdf")

        fig_traj, traj_axes = plt.subplots(1, 2, figsize=(14, 5.2))
        for i, budget in enumerate(["cv", "holdout"]):
            plot_all_methods_normalized_test_error_trajectory(
                suite_df,
                budget=budget,
                test_col=normalized_test_col(budget),
                ax=traj_axes[i],
                show_legend=True,
                aggregate_per_dataset_first=True,
            )
            add_panel_debug(
                traj_axes[i],
                panel_debug_text(suite_df, suite_final_for_scatter),
            )
        fig_traj.tight_layout()
        fig_traj.savefig(suite_output_dir / "normalized_test_loss_trajectory_cv_holdout.pdf", bbox_inches="tight", dpi=200)
        plt.close(fig_traj)
        print(f"  [✓] normalized_test_loss_trajectory_cv_holdout.pdf")

        fig_rank, rank_axes = plt.subplots(1, 2, figsize=(14, 5.2))
        for i, budget in enumerate(["cv", "holdout"]):
            plot_all_methods_average_rank_trajectory(
                suite_df,
                budget=budget,
                metric_col=raw_test_col(budget),
                metric_label="test loss",
                absolute_metric=False,
                ax=rank_axes[i],
                show_legend=True,
                aggregate_per_dataset_first=True,
            )
            add_panel_debug(
                rank_axes[i],
                panel_debug_text(suite_df, suite_final_for_scatter),
            )
        fig_rank.tight_layout()
        fig_rank.savefig(suite_output_dir / "average_rank_test_loss_cv_holdout.pdf", bbox_inches="tight", dpi=200)
        plt.close(fig_rank)
        print(f"  [✓] average_rank_test_loss_cv_holdout.pdf")

        # Critical difference diagrams (dataset-averaged), one per model.
        cd_model_order = ["FastAIMLP", "LGBM", "RealMLP"]
        for cd_model_name in cd_model_order:
            if "model" not in suite_df.columns:
                break

            model_subset = suite_df.loc[
                suite_df["model"].astype(str).eq(cd_model_name)
            ].copy()
            if model_subset.empty:
                print(f"  [plotting] Skip CD for {cd_model_name}: no rows")
                continue

            lgbm_matrix_export_path = None
            if cd_model_name == "LGBM":
                lgbm_matrix_export_path = str(suite_output_dir / "critical_difference_matrix_LGBM.csv")

            fig_cd, cd_stats, cd_counts, cd_dataset_counts, cd_missing_keys = plot_critical_difference_diagram(
                model_subset,
                comparison_unit="dataset",
                save_wide_csv_path=lgbm_matrix_export_path,
            )
            fig_cd.tight_layout()
            cd_file_name = f"critical_difference_diagram_{cd_model_name}.pdf"
            fig_cd.savefig(suite_output_dir / cd_file_name, bbox_inches="tight", dpi=200)
            plt.close(fig_cd)

            if lgbm_matrix_export_path is not None:
                print(f"  [✓] critical_difference_matrix_LGBM.csv")

            cd_total_candidates = cd_stats.n_runs + cd_stats.n_dropped_runs
            cd_retention = (100.0 * cd_stats.n_runs / cd_total_candidates) if cd_total_candidates > 0 else 0.0
            if not cd_counts.empty and "paired_runs" in cd_counts.columns:
                cd_bottleneck = cd_counts.sort_values("paired_runs").iloc[0]
                cd_bottleneck_text = f", bottleneck_method={cd_bottleneck['method']} ({int(cd_bottleneck['paired_runs'])} paired runs)"
            else:
                cd_bottleneck_text = ""
            print(
                f"  [✓] {cd_file_name}  "
                f"(model={cd_model_name}, unit=dataset, complete_case_observations={cd_stats.n_runs} datasets, "
                f"candidate_observations={cd_total_candidates}, "
                f"incomplete_observations_dropped={cd_stats.n_dropped_runs}, retention={cd_retention:.1f}%, "
                f"n_datasets={cd_stats.n_unique_datasets}, n_methods={cd_stats.n_methods}{cd_bottleneck_text})"
            )

        if suite_output_dir.name == "all_datasets" and not skip_pairwise:
            all_models_wide, all_models_stats, _, _, _ = build_cd_diagram_dataframe(
                suite_df,
                comparison_unit="dataset",
            )
            if all_models_wide.empty:
                print("  [plotting] Skip pairwise matrix (all-models): no complete-case dataset rows")
            else:
                matrix_csv_path = suite_output_dir / "critical_difference_matrix_all_models.csv"
                all_models_wide_reset = all_models_wide.reset_index()
                all_models_wide_reset.to_csv(matrix_csv_path, index=False)

                matrix_plot_path = suite_output_dir / "pairwise_wilcoxon_matrix_all_models.pdf"
                fig_pw, _ = plot_pairwise_wilcoxon_matrix(all_models_wide, matrix_plot_path)
                plt.close(fig_pw)
                print(
                    f"  [✓] {matrix_csv_path.name} "
                    f"({len(all_models_wide)} datasets x {all_models_wide.shape[1]} methods)"
                )
                print(
                    f"  [✓] {matrix_plot_path.name} "
                    f"(comparison_unit=dataset, n_datasets={all_models_stats.n_runs}, n_methods={all_models_stats.n_methods})"
                )

    def generate_cv_retrain_single_plots(suite_df: pd.DataFrame, suite_output_dir: Path) -> None:
        suite_output_dir.mkdir(parents=True, exist_ok=True)
        cdf_df = suite_df.copy()
        if "inner_split" in cdf_df.columns:
            cdf_df["inner_split"] = cdf_df["inner_split"].astype("string").fillna("").astype(str)
        suite_final_for_scatter = final_rows_per_run(suite_df)

        fig_cdf_single, ax_cdf_single = plt.subplots(figsize=(8, 6.7))
        fig_cdf, _ = plot_relative_overtuning_cdf_mitigations(
            cdf_df,
            budget="cv",
            improvement_threshold=args.improvement_threshold,
            method_builder=build_all_methods_cdf_with_benchmark,
            score_col="retrain_test_loss",
            overtuning_col="relative_retrain_overtuning",
            ax=ax_cdf_single,
            show_legend=True,
            panel_title=None,
        )
        add_panel_debug(fig_cdf.axes[0], panel_debug_text(cdf_df, suite_final_for_scatter))
        cdf_path = suite_output_dir / "relative_overtuning_cdf_cv_retrain.pdf"
        fig_cdf.tight_layout()
        fig_cdf.savefig(cdf_path, bbox_inches="tight", dpi=200)
        plt.close(fig_cdf)
        print(f"  [✓] {cdf_path.name}")

        fig_scatter_single, ax_scatter_single = plt.subplots(figsize=(8, 6.03))
        fig_scatter, _ = plot_relative_overtuning_delta_scatter(
            suite_final_for_scatter,
            budget="cv",
            method_builder=build_all_methods_delta,
            benchmark_mask_builder=combined_benchmark_mask,
            show_raw_points=False,
            summary_by_problem_type=False,
            clamp_cv_axes=False,
            overtuning_col="normalized_retrain_overtuning",
            test_loss_col="normalized_retrain_test_loss",
            split_method_problem_legend=False,
            ax=ax_scatter_single,
            show_legend=True,
            panel_title=None,
            precomputed_final_rows=suite_final_for_scatter,
            aggregate_per_dataset_first=True,
        )
        add_panel_debug(fig_scatter.axes[0], panel_debug_text(suite_df, suite_final_for_scatter))
        scatter_path = suite_output_dir / "normalized_delta_scatter_cv_retrain.pdf"
        fig_scatter.tight_layout()
        fig_scatter.savefig(scatter_path, bbox_inches="tight", dpi=200)
        plt.close(fig_scatter)
        print(f"  [✓] {scatter_path.name}")

    def generate_model_score_cd_plot(
        suite_df: pd.DataFrame,
        suite_output_dir: Path,
        include_realmlp: bool,
    ) -> None:
        from autorank import autorank, plot_stats

        suite_output_dir.mkdir(parents=True, exist_ok=True)
        final_df = final_rows_per_run(suite_df)
        if final_df.empty:
            print(f"  [plotting] Skip model-score CD for {suite_output_dir.name}: no final rows")
            return

        baseline_cv = final_df.loc[combined_benchmark_mask(final_df, "cv")].copy()
        if baseline_cv.empty:
            print(f"  [plotting] Skip model-score CD for {suite_output_dir.name}: no CV baseline rows")
            return

        baseline_cv["dataset_id_norm"] = baseline_cv["dataset_id"].map(normalize_dataset_id_value)
        model_specs: list[tuple[str, set[str], str]] = [
            ("FastAIMLP + CV ensembling", {"FastAIMLP"}, "ensembled_test_loss"),
            ("FastAIMLP + retraining", {"FastAIMLP"}, "retrain_test_loss"),
            ("LightGBM + CV ensembling", {"LightGBM", "LGBM"}, "ensembled_test_loss"),
            ("LightGBM + retraining", {"LightGBM", "LGBM"}, "retrain_test_loss"),
        ]
        if include_realmlp:
            model_specs.extend(
                [
                    ("RealMLP + CV ensembling", {"RealMLP"}, "ensembled_test_loss"),
                    ("RealMLP + retraining", {"RealMLP"}, "retrain_test_loss"),
                ]
            )

        series_list: list[pd.Series] = []
        for method_label, model_names, score_col in model_specs:
            method_rows = baseline_cv.loc[
                baseline_cv["model"].astype(str).isin(model_names),
                ["dataset_id_norm", score_col],
            ].copy()
            method_rows[score_col] = pd.to_numeric(method_rows[score_col], errors="coerce")
            method_series = (
                method_rows.dropna(subset=["dataset_id_norm", score_col])
                .groupby("dataset_id_norm", dropna=False)[score_col]
                .mean()
                .rename(method_label)
            )
            if not method_series.empty:
                series_list.append(method_series)

        if len(series_list) < 2:
            print(f"  [plotting] Skip model-score CD for {suite_output_dir.name}: insufficient method columns")
            return

        wide = pd.concat(series_list, axis=1, join="inner").dropna()
        if wide.shape[0] < 3 or wide.shape[1] < 2:
            print(
                f"  [plotting] Skip model-score CD for {suite_output_dir.name}: "
                f"insufficient complete-case rows (datasets={wide.shape[0]}, methods={wide.shape[1]})"
            )
            return

        csv_path = suite_output_dir / "critical_difference_matrix_model_score_cv_baseline.csv"
        wide.reset_index().to_csv(csv_path, index=False)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = autorank(wide, alpha=0.05, verbose=False, order="ascending")

        fig_height = max(2.3, 1.0 + 0.20 * wide.shape[1])
        fig, ax = plt.subplots(figsize=(13, fig_height))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_stats(result, ax=ax)
        fig.subplots_adjust(left=0.03, right=0.997, bottom=0.10, top=0.97)

        cd_path = suite_output_dir / "critical_difference_model_score_cv_baseline.pdf"
        fig.savefig(cd_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

        size_label = "small" if include_realmlp else "large"
        print(
            f"  [✓] {csv_path.name} "
            f"({size_label}_datasets, datasets={wide.shape[0]}, methods={wide.shape[1]})"
        )
        print(
            f"  [✓] {cd_path.name} "
            f"({size_label}_datasets, unit=dataset, cv_baseline_only=True)"
        )

    def generate_all_datasets_pairwise_matrix_only(suite_df: pd.DataFrame, suite_output_dir: Path) -> None:
        suite_output_dir.mkdir(parents=True, exist_ok=True)
        all_models_wide, all_models_stats, _, _, _ = build_cd_diagram_dataframe(
            suite_df,
            comparison_unit="dataset",
        )
        if all_models_wide.empty:
            raise ValueError("No complete-case dataset rows for all-model pairwise matrix")

        matrix_csv_path = suite_output_dir / "critical_difference_matrix_all_models.csv"
        all_models_wide.reset_index().to_csv(matrix_csv_path, index=False)

        matrix_plot_path = suite_output_dir / "pairwise_wilcoxon_matrix_all_models.pdf"
        fig_pw, _ = plot_pairwise_wilcoxon_matrix(all_models_wide, matrix_plot_path)
        plt.close(fig_pw)
        print(
            f"[plotting] [✓] {matrix_csv_path.name} "
            f"({len(all_models_wide)} datasets x {all_models_wide.shape[1]} methods)"
        )
        print(
            f"[plotting] [✓] {matrix_plot_path.name} "
            f"(comparison_unit=dataset, n_datasets={all_models_stats.n_runs}, n_methods={all_models_stats.n_methods})"
        )

    if args.only_pairwise_matrix:
        pairwise_dir = grouped_root / "all_datasets"
        generate_all_datasets_pairwise_matrix_only(incumbent_subset, pairwise_dir)
        print(f"[plotting] Output directory: {output_dir}")
        return

    def generate_appendix_combined_delta_scatter(
        suite_name: str,
        suite_df: pd.DataFrame,
        suite_output_dir: Path,
        model_labels: list[str],
    ) -> None:
        if suite_df.empty:
            print(f"[plotting] Skip appendix combined plot: {suite_name} has no rows")
            return

        suite_output_dir.mkdir(parents=True, exist_ok=True)

        problem_type_specs = [
            ("binary", "Binary"),
            ("multiclass", "Multiclass"),
            ("regression", "Regression"),
        ]
        model_aliases = {
            "FastAIMLP": {"FastAIMLP"},
            "LightGBM": {"LightGBM", "LGBM"},
            "RealMLP": {"RealMLP"},
        }
        n_rows = len(problem_type_specs)
        n_cols = len(model_labels)
        base_height = max(13.2, 4.6 * n_rows)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(max(8.8, 4.4 * n_cols), base_height * 1.20),
            squeeze=False,
        )
        print(
            "[plotting] Generating appendix combined delta scatter: "
            f"{suite_name} ({n_rows} rows x {n_cols} cols)"
        )

        for row_idx, (problem_type, problem_type_label) in enumerate(problem_type_specs):
            problem_df = suite_df.loc[
                suite_df["problem_type"].astype(str).str.strip().str.lower().eq(problem_type)
            ].copy()
            for col_idx, model_label in enumerate(model_labels):
                ax = axes[row_idx, col_idx]
                model_df = problem_df.loc[
                    problem_df["model"].astype(str).isin(model_aliases[model_label])
                ].copy()
                model_final_rows = final_rows_per_run(model_df)
                plot_relative_overtuning_delta_scatter(
                    model_final_rows,
                    budget="cv",
                    method_builder=build_all_methods_delta,
                    benchmark_mask_builder=combined_benchmark_mask,
                    show_raw_points=False,
                    summary_by_problem_type=False,
                    clamp_cv_axes=False,
                    overtuning_col="normalized_ensemble_overtuning",
                    test_loss_col="normalized_ensembled_test_loss",
                    split_method_problem_legend=False,
                    ax=ax,
                    show_legend=False,
                    panel_title=None,
                    precomputed_final_rows=model_final_rows,
                    aggregate_per_dataset_first=True,
                )
                ax.set_xlabel(ax.get_xlabel(), fontsize=14)
                ax.set_ylabel(ax.get_ylabel(), fontsize=14)
                ax.tick_params(axis="both", labelsize=11)
                add_panel_debug(ax, panel_debug_text(model_df, model_final_rows))

                if row_idx == 0:
                    ax.text(
                        0.5,
                        1.18,
                        model_label,
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=13,
                        fontweight="semibold",
                    )
                if col_idx == 0:
                    ax.text(
                        -0.34,
                        0.5,
                        problem_type_label,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=13,
                        fontweight="semibold",
                    )

        legend_rows = final_rows_per_run(suite_df)
        legend_handles: list[Line2D] = []
        for spec, mask in build_all_methods_delta(legend_rows, budget="cv"):
            if not bool(mask.any()):
                continue
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=spec.color,
                    markeredgecolor=spec.color,
                    markeredgewidth=1.0,
                    markersize=7,
                    label=spec.label.replace("\n", " "),
                )
            )
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.03),
                ncol=4,
                fontsize=11,
                framealpha=0.9,
                numpoints=1,
                handlelength=1.0,
                columnspacing=1.4,
                handletextpad=0.6,
            )

        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.90, wspace=0.30, hspace=0.42)
        output_path = suite_output_dir / f"appendix_normalized_delta_scatter_{suite_name}.pdf"
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"  [✓] {output_path.name}")

    def _ordered_optimizer_labels(df: pd.DataFrame) -> list[str]:
        if "optimizer" not in df.columns or df.empty:
            return []
        optimizer_series = df["optimizer"].astype("object").where(pd.notna(df["optimizer"]), "").astype(str).str.strip()
        optimizer_series = optimizer_series[optimizer_series.ne("")]
        if optimizer_series.empty:
            return []
        counts = optimizer_series.value_counts()
        return sorted(counts.index.tolist(), key=lambda opt: (-int(counts.loc[opt]), opt.lower()))

    def _display_optimizer_label(value: str) -> str:
        optimizer_norm = str(value).strip().lower()
        if optimizer_norm == "hebo":
            return "HEBO"
        if optimizer_norm == "smac":
            return "SMAC"
        return str(value)

    def generate_appendix_optimizer_delta_scatter(
        suite_name: str,
        suite_df: pd.DataFrame,
        suite_output_dir: Path,
    ) -> None:
        if suite_df.empty:
            print(f"[plotting] Skip appendix optimizer plot: {suite_name} has no rows")
            return

        optimizer_labels = _ordered_optimizer_labels(suite_df)
        if not optimizer_labels:
            print(f"[plotting] Skip appendix optimizer plot: {suite_name} has no optimizer labels")
            return

        problem_type_specs = [
            ("binary", "Binary"),
            ("multiclass", "Multiclass"),
            ("regression", "Regression"),
        ]
        n_rows = len(problem_type_specs)
        n_cols = len(optimizer_labels)
        base_height = max(13.2, 4.6 * n_rows)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(max(8.8, 4.4 * n_cols), base_height * 1.20),
            squeeze=False,
        )
        print(
            "[plotting] Generating appendix optimizer delta scatter: "
            f"{suite_name} ({n_rows} rows x {n_cols} cols)"
        )

        for row_idx, (problem_type, problem_type_label) in enumerate(problem_type_specs):
            problem_df = suite_df.loc[
                suite_df["problem_type"].astype(str).str.strip().str.lower().eq(problem_type)
            ].copy()
            for col_idx, optimizer_label in enumerate(optimizer_labels):
                ax = axes[row_idx, col_idx]
                optimizer_series = problem_df["optimizer"].astype("object").where(pd.notna(problem_df["optimizer"]), "").astype(str).str.strip()
                optimizer_df = problem_df.loc[
                    optimizer_series.eq(optimizer_label)
                ].copy()
                optimizer_final_rows = final_rows_per_run(optimizer_df)
                plot_relative_overtuning_delta_scatter(
                    optimizer_final_rows,
                    budget="cv",
                    method_builder=build_all_methods_delta,
                    benchmark_mask_builder=combined_benchmark_mask,
                    show_raw_points=False,
                    summary_by_problem_type=False,
                    clamp_cv_axes=False,
                    overtuning_col="normalized_ensemble_overtuning",
                    test_loss_col="normalized_ensembled_test_loss",
                    split_method_problem_legend=False,
                    ax=ax,
                    show_legend=False,
                    panel_title=None,
                    precomputed_final_rows=optimizer_final_rows,
                    aggregate_per_dataset_first=True,
                )
                ax.set_xlabel(ax.get_xlabel(), fontsize=14)
                ax.set_ylabel(ax.get_ylabel(), fontsize=14)
                ax.tick_params(axis="both", labelsize=11)
                add_panel_debug(ax, panel_debug_text(optimizer_df, optimizer_final_rows))

                if row_idx == 0:
                    ax.text(
                        0.5,
                        1.18,
                        _display_optimizer_label(optimizer_label),
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=13,
                        fontweight="semibold",
                    )
                if col_idx == 0:
                    ax.text(
                        -0.34,
                        0.5,
                        problem_type_label,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=13,
                        fontweight="semibold",
                    )

        legend_rows = final_rows_per_run(suite_df)
        legend_handles: list[Line2D] = []
        for spec, mask in build_all_methods_delta(legend_rows, budget="cv"):
            if not bool(mask.any()):
                continue
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=spec.color,
                    markeredgecolor=spec.color,
                    markeredgewidth=1.0,
                    markersize=7,
                    label=spec.label.replace("\n", " "),
                )
            )
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.03),
                ncol=4,
                fontsize=11,
                framealpha=0.9,
                numpoints=1,
                handlelength=1.0,
                columnspacing=1.4,
                handletextpad=0.6,
            )

        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.90, wspace=0.30, hspace=0.42)
        output_path = suite_output_dir / f"appendix_normalized_delta_scatter_by_optimizer_{suite_name}.pdf"
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"  [✓] {output_path.name}")

    def _make_cdf_legend_handles(suite_df: pd.DataFrame) -> list[Line2D]:
        """Shared legend handles for CDF appendix plots (line style, no markers)."""
        final_rows = final_rows_per_run(suite_df)
        handles: list[Line2D] = []
        for spec, mask in build_all_methods_cdf_with_benchmark(final_rows, budget="cv"):
            if not bool(mask.any()):
                continue
            handles.append(
                Line2D(
                    [0], [0],
                    color=spec.color,
                    linewidth=2.0,
                    linestyle="-",
                    label=spec.label.replace("\n", " "),
                )
            )
        return handles

    def _attach_shared_cdf_legend(fig: plt.Figure, legend_handles: list[Line2D]) -> None:
        if not legend_handles:
            return
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=4,
            fontsize=11,
            framealpha=0.9,
            handlelength=2.0,
            columnspacing=1.4,
            handletextpad=0.6,
        )

    def generate_appendix_cdf_by_model(
        suite_name: str,
        suite_df: pd.DataFrame,
        suite_output_dir: Path,
        model_specs: list[tuple[str, set[str], str]],
    ) -> None:
        """CDF grid: rows=problem_type, columns=ML algorithm (no baseline)."""
        if suite_df.empty:
            print(f"[plotting] Skip appendix CDF by model: {suite_name} has no rows")
            return
        problem_type_specs = [
            ("binary", "Binary"),
            ("multiclass", "Multiclass"),
            ("regression", "Regression"),
        ]
        score_col, overtuning_col = cdf_cols("cv")
        n_rows = len(problem_type_specs)
        n_cols = len(model_specs)
        base_height = max(13.2, 4.6 * n_rows)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(max(8.8, 4.4 * n_cols), base_height * 1.20),
            squeeze=False,
        )
        print(f"[plotting] Generating appendix CDF by model: {suite_name} ({n_rows} rows x {n_cols} cols)")
        for row_idx, (problem_type, problem_type_label) in enumerate(problem_type_specs):
            problem_df = suite_df.loc[
                suite_df["problem_type"].astype(str).str.strip().str.lower().eq(problem_type)
            ].copy()
            for col_idx, (model_col_label, model_aliases_set, _) in enumerate(model_specs):
                ax = axes[row_idx, col_idx]
                model_df = problem_df.loc[
                    problem_df["model"].astype(str).isin(model_aliases_set)
                ].copy()
                if "inner_split" in model_df.columns:
                    model_df["inner_split"] = model_df["inner_split"].astype(object).where(
                        pd.notna(model_df["inner_split"]), None
                    ).astype(str)
                plot_relative_overtuning_cdf_mitigations(
                    model_df,
                    budget="cv",
                    improvement_threshold=args.improvement_threshold,
                    method_builder=build_all_methods_cdf_with_benchmark,
                    score_col=score_col,
                    overtuning_col=overtuning_col,
                    ax=ax,
                    show_legend=False,
                    panel_title=None,
                )
                ax.set_xlabel(ax.get_xlabel(), fontsize=14)
                ax.set_ylabel(ax.get_ylabel(), fontsize=14)
                ax.tick_params(axis="both", labelsize=11)
                add_panel_debug(ax, panel_debug_text(model_df, final_rows_per_run(model_df)))
                if row_idx == 0:
                    ax.text(0.5, 1.18, model_col_label, transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=13, fontweight="semibold")
                if col_idx == 0:
                    ax.text(-0.34, 0.5, problem_type_label, transform=ax.transAxes,
                            rotation=90, ha="center", va="center", fontsize=13, fontweight="semibold")
        _attach_shared_cdf_legend(fig, _make_cdf_legend_handles(suite_df))
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.90, wspace=0.30, hspace=0.42)
        output_path = suite_output_dir / f"appendix_cdf_by_model_{suite_name}.pdf"
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"  [✓] {output_path.name}")

    def generate_appendix_cdf_by_optimizer(
        suite_name: str,
        suite_df: pd.DataFrame,
        suite_output_dir: Path,
    ) -> None:
        """CDF grid: rows=problem_type, columns=optimizer (no baseline)."""
        if suite_df.empty:
            print(f"[plotting] Skip appendix CDF by optimizer: {suite_name} has no rows")
            return
        optimizer_labels = _ordered_optimizer_labels(suite_df)
        if not optimizer_labels:
            print(f"[plotting] Skip appendix CDF by optimizer: {suite_name} has no optimizer labels")
            return
        problem_type_specs = [
            ("binary", "Binary"),
            ("multiclass", "Multiclass"),
            ("regression", "Regression"),
        ]
        score_col, overtuning_col = cdf_cols("cv")
        n_rows = len(problem_type_specs)
        n_cols = len(optimizer_labels)
        base_height = max(13.2, 4.6 * n_rows)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(max(8.8, 4.4 * n_cols), base_height * 1.20),
            squeeze=False,
        )
        print(f"[plotting] Generating appendix CDF by optimizer: {suite_name} ({n_rows} rows x {n_cols} cols)")
        for row_idx, (problem_type, problem_type_label) in enumerate(problem_type_specs):
            problem_df = suite_df.loc[
                suite_df["problem_type"].astype(str).str.strip().str.lower().eq(problem_type)
            ].copy()
            for col_idx, optimizer_label in enumerate(optimizer_labels):
                ax = axes[row_idx, col_idx]
                optimizer_series = problem_df["optimizer"].astype("object").where(
                    pd.notna(problem_df["optimizer"]), ""
                ).astype(str).str.strip()
                optimizer_df = problem_df.loc[optimizer_series.eq(optimizer_label)].copy()
                if "inner_split" in optimizer_df.columns:
                    optimizer_df["inner_split"] = optimizer_df["inner_split"].astype(object).where(
                        pd.notna(optimizer_df["inner_split"]), None
                    ).astype(str)
                plot_relative_overtuning_cdf_mitigations(
                    optimizer_df,
                    budget="cv",
                    improvement_threshold=args.improvement_threshold,
                    method_builder=build_all_methods_cdf_with_benchmark,
                    score_col=score_col,
                    overtuning_col=overtuning_col,
                    ax=ax,
                    show_legend=False,
                    panel_title=None,
                )
                ax.set_xlabel(ax.get_xlabel(), fontsize=14)
                ax.set_ylabel(ax.get_ylabel(), fontsize=14)
                ax.tick_params(axis="both", labelsize=11)
                add_panel_debug(ax, panel_debug_text(optimizer_df, final_rows_per_run(optimizer_df)))
                if row_idx == 0:
                    ax.text(0.5, 1.18, _display_optimizer_label(optimizer_label), transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=13, fontweight="semibold")
                if col_idx == 0:
                    ax.text(-0.34, 0.5, problem_type_label, transform=ax.transAxes,
                            rotation=90, ha="center", va="center", fontsize=13, fontweight="semibold")
        _attach_shared_cdf_legend(fig, _make_cdf_legend_handles(suite_df))
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.90, wspace=0.30, hspace=0.42)
        output_path = suite_output_dir / f"appendix_cdf_by_optimizer_{suite_name}.pdf"
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"  [✓] {output_path.name}")

    if args.partial:
        all_datasets_suite_dir = grouped_root / "all_datasets"
        generate_group_suite(incumbent_subset, all_datasets_suite_dir, skip_pairwise=True)
        print(f"[plotting] Generated partial grouped suite (no statistical testing or appendix plots)")
        print(f"[plotting] Output directory: {output_dir}")
        return

    group_specs: list[tuple[str, pd.DataFrame]] = [("all_datasets", incumbent_subset)]
    appendix_plot_groups: list[tuple[str, pd.DataFrame, list[str]]] = [("all_datasets", incumbent_subset, ["FastAIMLP", "LightGBM"])]
    appendix_optimizer_plot_groups: list[tuple[str, pd.DataFrame]] = [("all_datasets", incumbent_subset)]
    _MODEL_ALIASES: dict[str, set[str]] = {
        "FastAIMLP": {"FastAIMLP"},
        "LightGBM": {"LightGBM", "LGBM"},
        "RealMLP": {"RealMLP"},
    }
    appendix_cdf_model_groups: list[tuple[str, pd.DataFrame, list[tuple[str, set[str], str]]]] = [
        ("all_datasets", incumbent_subset, [
            ("FastAIMLP", _MODEL_ALIASES["FastAIMLP"], "FastAIMLP"),
            ("LightGBM", _MODEL_ALIASES["LightGBM"], "LightGBM"),
        ]),
    ]
    appendix_cdf_optimizer_groups: list[tuple[str, pd.DataFrame]] = [("all_datasets", incumbent_subset)]
    if not dataset_info.empty and {"Dataset ID", "# Instances", "Type"}.issubset(dataset_info.columns) and "dataset_id" in incumbent_subset.columns:
        binary_metadata = dataset_info.copy()
        binary_metadata["type_norm"] = binary_metadata["Type"].fillna("").astype(str).str.strip().str.lower()
        binary_metadata["instances_num"] = pd.to_numeric(binary_metadata["# Instances"], errors="coerce")
        binary_metadata["dataset_norm"] = binary_metadata["Dataset ID"].map(normalize_dataset_id_value)
        incumbent_dataset_norm = incumbent_subset["dataset_id"].map(normalize_dataset_id_value)

        small_dataset_ids = set(binary_metadata.loc[binary_metadata["instances_num"].lt(2500), "dataset_norm"])
        large_dataset_ids = set(binary_metadata.loc[binary_metadata["instances_num"].ge(2500), "dataset_norm"])
        binary_small_ids = set(
            binary_metadata.loc[
                binary_metadata["type_norm"].eq("binary")
                & binary_metadata["instances_num"].lt(2500),
                "dataset_norm",
            ]
        )

        all_small_datasets_df = incumbent_subset.loc[incumbent_dataset_norm.isin(small_dataset_ids)].copy()
        all_large_datasets_df = incumbent_subset.loc[incumbent_dataset_norm.isin(large_dataset_ids)].copy()
        all_small_binary_df = incumbent_subset.loc[incumbent_dataset_norm.isin(binary_small_ids)].copy()

        group_specs.append(("all_small_datasets", all_small_datasets_df))
        group_specs.append(("all_large_datasets", all_large_datasets_df))
        group_specs.append(("all_small_binary", incumbent_subset.loc[incumbent_dataset_norm.isin(binary_small_ids)].copy()))

        appendix_plot_groups.append(("all_small_datasets", all_small_datasets_df, ["FastAIMLP", "LightGBM", "RealMLP"]))
        appendix_optimizer_plot_groups.append(("all_small_datasets", all_small_datasets_df))
        appendix_cdf_model_groups.append(("all_small_datasets", all_small_datasets_df, [
            ("FastAIMLP", _MODEL_ALIASES["FastAIMLP"], "FastAIMLP"),
            ("LightGBM", _MODEL_ALIASES["LightGBM"], "LightGBM"),
            ("RealMLP\n(small datasets only)", _MODEL_ALIASES["RealMLP"], "RealMLP"),
        ]))
        appendix_cdf_optimizer_groups.append(("all_small_datasets", all_small_datasets_df))
    else:
        print("[plotting] Missing dataset metadata. Only all_datasets group will be generated.")

    if args.only_cv_retrain_plots:
        group_lookup = {group_name: group_df for group_name, group_df in group_specs}
        for group_name in ["all_datasets", "all_small_binary"]:
            group_df = group_lookup.get(group_name)
            if group_df is None or group_df.empty:
                print(f"[plotting] Skip {group_name}: no rows available")
                continue
            print(f"[plotting] Generating isolated CV-with-retrain plots: {group_name}")
            generate_cv_retrain_single_plots(group_df, grouped_root / group_name)
        print(f"[plotting] Output directory: {output_dir}")
        return

    if args.only_model_score_cd:
        group_lookup = {group_name: group_df for group_name, group_df in group_specs}
        for group_name, include_realmlp in [
            ("all_datasets", False),
            ("all_small_datasets", True),
        ]:
            group_df = group_lookup.get(group_name)
            if group_df is None or group_df.empty:
                print(f"[plotting] Skip {group_name}: no rows available")
                continue
            print(f"[plotting] Generating isolated model-score CD: {group_name}")
            generate_model_score_cd_plot(
                group_df,
                grouped_root / group_name,
                include_realmlp=include_realmlp,
            )
        print(f"[plotting] Output directory: {output_dir}")
        return

    if args.only_appendix_combined_plots:
        appendix_dir = grouped_root / "appendix_overview"
        appendix_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ["overview_*.pdf", "appendix_normalized_delta_scatter_*.pdf"]:
            for existing_plot in appendix_dir.glob(pattern):
                existing_plot.unlink()
        for suite_name, suite_df, model_labels in appendix_plot_groups:
            generate_appendix_combined_delta_scatter(
                suite_name,
                suite_df,
                appendix_dir,
                model_labels,
            )
        print(f"[plotting] Output directory: {output_dir}")
        return

    if args.only_appendix_optimizer_plots:
        optimizer_appendix_dir = grouped_root / "appendix_overview_by_optimizer"
        optimizer_appendix_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ["appendix_normalized_delta_scatter_by_optimizer_*.pdf"]:
            for existing_plot in optimizer_appendix_dir.glob(pattern):
                existing_plot.unlink()
        for suite_name, suite_df in appendix_optimizer_plot_groups:
            generate_appendix_optimizer_delta_scatter(
                suite_name,
                suite_df,
                optimizer_appendix_dir,
            )
        print(f"[plotting] Output directory: {output_dir}")
        return

    if args.only_appendix_cdf_plots:
        cdf_appendix_dir = grouped_root / "appendix_cdf"
        cdf_appendix_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ["appendix_cdf_*.pdf"]:
            for existing_plot in cdf_appendix_dir.glob(pattern):
                existing_plot.unlink()
        for suite_name, suite_df, model_specs in appendix_cdf_model_groups:
            generate_appendix_cdf_by_model(suite_name, suite_df, cdf_appendix_dir, model_specs)
        for suite_name, suite_df in appendix_cdf_optimizer_groups:
            generate_appendix_cdf_by_optimizer(suite_name, suite_df, cdf_appendix_dir)
        print(f"[plotting] Output directory: {output_dir}")
        return

    if args.only_benchmark_aggregate_selected:
        def _generate_benchmark_aggregate_selected():
            benchmark_targets: list[tuple[str, str, str]] = [
                ("363631", "LightGBM", "holdout"),
                ("363693", "LightGBM", "holdout"),
                ("363682", "LightGBM", "holdout"),
            ]

            model_aliases: dict[str, set[str]] = {
                "LightGBM": {"LightGBM", "LGBM"},
                "LGBM": {"LightGBM", "LGBM"},
            }

            benchmark_target_dir = output_dir / "benchmark_aggregate_selected"
            benchmark_target_dir.mkdir(parents=True, exist_ok=True)
            benchmark_df = load_trajectory_source(
                args.data_dir,
                source="default",
                usecols=[
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
                    "iteration",
                    "val_performance",
                    "retrain_test_loss",
                    "ensembled_test_loss",
                ],
            )
            benchmark_df = benchmark_df.copy()
            benchmark_df["dataset_id_norm"] = benchmark_df["dataset_id"].map(normalize_dataset_id_value)

            benchmark_target_name_overrides: dict[str, str] = {
                "363631": "diamonds",
                "363682": "Is-this-a-good-customer",
                "363693": "physiochemical_protein",
            }

            local_dataset_name_map: dict[str, str] = {}
            if not dataset_info.empty and {"Dataset ID", "Name"}.issubset(dataset_info.columns):
                dataset_name_series = dataset_info["Dataset ID"].map(normalize_dataset_id_value)
                local_dataset_name_map = {
                    dataset_id: str(name)
                    for dataset_id, name in zip(dataset_name_series, dataset_info["Name"])
                    if dataset_id
                }
            local_dataset_name_map.update(benchmark_target_name_overrides)

            def _filename_slug(value: str) -> str:
                slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
                return slug or "dataset"

            generated_benchmark_targets = 0
            for dataset_id, model_name, budget in benchmark_targets:
                allowed_model_names = model_aliases.get(model_name, {model_name})
                target_mask = (
                    benchmark_df["dataset_id_norm"].eq(dataset_id)
                    & benchmark_df["model"].astype(str).isin(allowed_model_names)
                    & benchmark_df["inner_split"].astype(str).str.strip().str.lower().eq(budget)
                )
                subset = benchmark_df.loc[target_mask].copy()
                if subset.empty:
                    print(f"[plotting] Skip benchmark aggregate target {dataset_id}/{model_name}/{budget}: no matching rows")
                    continue

                dataset_label = local_dataset_name_map.get(dataset_id, dataset_id)
                dataset_slug = _filename_slug(dataset_label)

                fig_benchmark_retrain, benchmark_stats_retrain = plot_benchmark_dataset_trajectory(
                    subset,
                    dataset_label=dataset_label,
                    budget=budget,
                    test_col="retrain_test_loss",
                )
                retrain_pdf = benchmark_target_dir / f"benchmark_trajectory_{dataset_slug}_{model_name}_{budget}_retrain.pdf"
                fig_benchmark_retrain.savefig(retrain_pdf, bbox_inches="tight", dpi=200)
                plt.close(fig_benchmark_retrain)

                fig_benchmark_ensemble, benchmark_stats_ensemble = plot_benchmark_dataset_trajectory(
                    subset,
                    dataset_label=dataset_label,
                    budget=budget,
                    test_col="ensembled_test_loss",
                )
                ensemble_pdf = benchmark_target_dir / f"benchmark_trajectory_{dataset_slug}_{model_name}_{budget}_ensemble.pdf"
                fig_benchmark_ensemble.savefig(ensemble_pdf, bbox_inches="tight", dpi=200)
                plt.close(fig_benchmark_ensemble)

                generated_benchmark_targets += 1
                print(
                    f"[plotting] [✓] {retrain_pdf.name} "
                    f"(dataset={dataset_label}, dataset_id={dataset_id}, model={model_name}, budget={budget}, runs={benchmark_stats_retrain.runs}, points={benchmark_stats_retrain.points})"
                )
                print(
                    f"[plotting] [✓] {ensemble_pdf.name} "
                    f"(dataset={dataset_label}, dataset_id={dataset_id}, model={model_name}, budget={budget}, runs={benchmark_stats_ensemble.runs}, points={benchmark_stats_ensemble.points})"
                )

            print(
                f"[plotting] Generated {generated_benchmark_targets}/{len(benchmark_targets)} "
                f"targeted benchmark aggregate trajectories"
            )
        
        _generate_benchmark_aggregate_selected()
        print(f"[plotting] Output directory: {output_dir}")
        return

    for group_name, group_df in group_specs:
        generate_group_suite(group_df, grouped_root / group_name)

    print(f"[plotting] Generated {len(group_specs)} grouped suites")

    appendix_dir = grouped_root / "appendix_overview"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["overview_*.pdf", "appendix_normalized_delta_scatter_*.pdf"]:
        for existing_plot in appendix_dir.glob(pattern):
            existing_plot.unlink()
    for suite_name, suite_df, model_labels in appendix_plot_groups:
        generate_appendix_combined_delta_scatter(
            suite_name,
            suite_df,
            appendix_dir,
            model_labels,
        )

    optimizer_appendix_dir = grouped_root / "appendix_overview_by_optimizer"
    optimizer_appendix_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["appendix_normalized_delta_scatter_by_optimizer_*.pdf"]:
        for existing_plot in optimizer_appendix_dir.glob(pattern):
            existing_plot.unlink()
    for suite_name, suite_df in appendix_optimizer_plot_groups:
        generate_appendix_optimizer_delta_scatter(
            suite_name,
            suite_df,
            optimizer_appendix_dir,
        )

    cdf_appendix_dir = grouped_root / "appendix_cdf"
    cdf_appendix_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["appendix_cdf_*.pdf"]:
        for existing_plot in cdf_appendix_dir.glob(pattern):
            existing_plot.unlink()
    for suite_name, suite_df, model_specs in appendix_cdf_model_groups:
        generate_appendix_cdf_by_model(suite_name, suite_df, cdf_appendix_dir, model_specs)
    for suite_name, suite_df in appendix_cdf_optimizer_groups:
        generate_appendix_cdf_by_optimizer(suite_name, suite_df, cdf_appendix_dir)

    benchmark_targets: list[tuple[str, str, str]] = [
        ("363631", "LightGBM", "holdout"),
        ("363693", "LightGBM", "holdout"),
        ("363682", "LightGBM", "holdout"),
    ]

    model_aliases: dict[str, set[str]] = {
        "LightGBM": {"LightGBM", "LGBM"},
        "LGBM": {"LightGBM", "LGBM"},
    }

    benchmark_target_dir = output_dir / "benchmark_aggregate_selected"
    benchmark_target_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df = load_trajectory_source(
        args.data_dir,
        source="default",
        usecols=[
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
            "iteration",
            "val_performance",
            "retrain_test_loss",
            "ensembled_test_loss",
        ],
    )
    benchmark_df = benchmark_df.copy()
    benchmark_df["dataset_id_norm"] = benchmark_df["dataset_id"].map(normalize_dataset_id_value)

    dataset_name_map: dict[str, str] = {}
    if not dataset_info.empty and {"Dataset ID", "Name"}.issubset(dataset_info.columns):
        dataset_name_series = dataset_info["Dataset ID"].map(normalize_dataset_id_value)
        dataset_name_map = {
            dataset_id: str(name)
            for dataset_id, name in zip(dataset_name_series, dataset_info["Name"])
            if dataset_id
        }
    dataset_name_map.update(benchmark_target_name_overrides)

    def _filename_slug(value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
        return slug or "dataset"

    generated_benchmark_targets = 0
    for dataset_id, model_name, budget in benchmark_targets:
        allowed_model_names = model_aliases.get(model_name, {model_name})
        target_mask = (
            benchmark_df["dataset_id_norm"].eq(dataset_id)
            & benchmark_df["model"].astype(str).isin(allowed_model_names)
            & benchmark_df["inner_split"].astype(str).str.strip().str.lower().eq(budget)
        )
        subset = benchmark_df.loc[target_mask].copy()
        if subset.empty:
            print(f"[plotting] Skip benchmark aggregate target {dataset_id}/{model_name}/{budget}: no matching rows")
            continue

        dataset_label = dataset_name_map.get(dataset_id, dataset_id)
        dataset_slug = _filename_slug(dataset_label)

        fig_benchmark_retrain, benchmark_stats_retrain = plot_benchmark_dataset_trajectory(
            subset,
            dataset_label=dataset_label,
            budget=budget,
            test_col="retrain_test_loss",
        )
        retrain_pdf = benchmark_target_dir / f"benchmark_trajectory_{dataset_slug}_{model_name}_{budget}_retrain.pdf"
        fig_benchmark_retrain.savefig(retrain_pdf, bbox_inches="tight", dpi=200)
        plt.close(fig_benchmark_retrain)

        fig_benchmark_ensemble, benchmark_stats_ensemble = plot_benchmark_dataset_trajectory(
            subset,
            dataset_label=dataset_label,
            budget=budget,
            test_col="ensembled_test_loss",
        )
        ensemble_pdf = benchmark_target_dir / f"benchmark_trajectory_{dataset_slug}_{model_name}_{budget}_ensemble.pdf"
        fig_benchmark_ensemble.savefig(ensemble_pdf, bbox_inches="tight", dpi=200)
        plt.close(fig_benchmark_ensemble)

        generated_benchmark_targets += 1
        print(
            f"[plotting] [✓] {retrain_pdf.name} "
            f"(dataset={dataset_label}, dataset_id={dataset_id}, model={model_name}, budget={budget}, runs={benchmark_stats_retrain.runs}, points={benchmark_stats_retrain.points})"
        )
        print(
            f"[plotting] [✓] {ensemble_pdf.name} "
            f"(dataset={dataset_label}, dataset_id={dataset_id}, model={model_name}, budget={budget}, runs={benchmark_stats_ensemble.runs}, points={benchmark_stats_ensemble.points})"
        )

    print(
        f"[plotting] Generated {generated_benchmark_targets}/{len(benchmark_targets)} "
        f"targeted benchmark aggregate trajectories"
    )

    if not args.per_dataset:
        print("[plotting] Per-dataset outputs disabled (pass --per-dataset to enable)")
        print(f"[plotting] Output directory: {output_dir}")
        return

    if "dataset_id" not in incumbent_subset.columns:
        print("[plotting] No dataset_id column found; skipping per-dataset outputs")
        print(f"[plotting] Output directory: {output_dir}")
        return

    dataset_norm_series = incumbent_subset["dataset_id"].map(normalize_dataset_id_value)
    grouped_dataset_indices = incumbent_subset.groupby(dataset_norm_series, sort=True).groups
    incumbent_final_for_scatter = final_rows_per_run(incumbent_subset)
    final_norm_series = incumbent_final_for_scatter["dataset_id"].map(normalize_dataset_id_value)
    final_groups = incumbent_final_for_scatter.groupby(final_norm_series, sort=True).groups

    valid_dataset_ids = [
        dataset_id for dataset_id, index_values in grouped_dataset_indices.items()
        if dataset_id != "" and dataset_id in final_groups
    ]
    print(f"[plotting] Generating per-dataset outputs ({len(valid_dataset_ids)} datasets)")

    for dataset_num, dataset_id in enumerate(valid_dataset_ids, 1):
        index_values = grouped_dataset_indices[dataset_id]
        dataset_traj_subset = incumbent_subset.loc[index_values].copy()
        dataset_scatter_subset = incumbent_final_for_scatter.loc[final_groups[dataset_id]].copy()
        if dataset_traj_subset.empty or dataset_scatter_subset.empty:
            continue

        print(f"[plotting] [{dataset_num}/{len(valid_dataset_ids)}] Processing dataset_{dataset_id} ({len(dataset_traj_subset)} rows)")
        dataset_dir = per_dataset_root / f"dataset_{dataset_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        fig_scatter, scatter_axes = plt.subplots(1, 2, figsize=(14, 5.3))
        for i, budget in enumerate(["cv", "holdout"]):
            ot_col, tl_col = normalized_scatter_cols(budget)
            plot_relative_overtuning_delta_scatter(
                dataset_scatter_subset,
                budget=budget,
                method_builder=build_all_methods_delta,
                benchmark_mask_builder=combined_benchmark_mask,
                show_raw_points=False,
                summary_by_problem_type=False,
                overtuning_col=ot_col,
                test_loss_col=tl_col,
                clamp_cv_axes=False,
                split_method_problem_legend=False,
                ax=scatter_axes[i],
                show_legend=True,
                panel_title=None,
                precomputed_final_rows=dataset_scatter_subset,
                aggregate_per_dataset_first=False,
            )
            add_panel_debug(
                scatter_axes[i],
                panel_debug_text(dataset_traj_subset, dataset_scatter_subset),
            )
        fig_scatter.tight_layout()
        fig_scatter.savefig(dataset_dir / "delta_scatter_cv_holdout.pdf", dpi=200)
        plt.close(fig_scatter)

        fig_traj, traj_axes = plt.subplots(1, 2, figsize=(14, 5.2))
        for i, budget in enumerate(["cv", "holdout"]):
            plot_all_methods_normalized_test_error_trajectory(
                dataset_traj_subset,
                budget=budget,
                test_col=raw_test_col(budget),
                ax=traj_axes[i],
                show_legend=True,
                aggregate_per_dataset_first=False,
            )
            add_panel_debug(
                traj_axes[i],
                panel_debug_text(dataset_traj_subset, dataset_scatter_subset),
            )
        fig_traj.tight_layout()
        fig_traj.savefig(dataset_dir / "test_loss_trajectory_cv_holdout.pdf", dpi=200)
        plt.close(fig_traj)
        print(f"  [✓] dataset_{dataset_id} complete")

    print(f"[plotting] Generated {len(valid_dataset_ids)} per-dataset outputs")
    print(f"[plotting] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
