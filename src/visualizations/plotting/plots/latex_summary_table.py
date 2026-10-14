"""Generate per-dataset LaTeX summary tables: one mini-table per dataset, methods as rows."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..filters import final_rows_per_run, normalize_dataset_id_value
from ..methods import (
    BASELINE_LABEL,
    BERGMAN_LABEL,
    BERGMAN_RESHUFFLING_LABEL,
    MLPLAN_LABEL,
    SURROGATE_RESHUFFLING_LABEL,
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
    post_hoc_surrogate_reshuffling_mask,
    selection_set_mask,
)


def _hex_to_rgb_norm(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _latex_color_def(name: str, hex_color: str) -> str:
    r, g, b = _hex_to_rgb_norm(hex_color)
    return f"\\definecolor{{{name}}}{{rgb}}{{{r:.3f},{g:.3f},{b:.3f}}}"


# (display_name, method_color_key, budget, perf_col, ot_col)
_ALL_METHODS: list[tuple[str, str, str, str, str]] = [
    # CV methods
    ("Baseline (CV)", BASELINE_LABEL, "cv", "ensembled_test_loss", "ensemble_overtuning"),
    ("Selection set (CV)", "Selection set", "cv", "ensembled_test_loss", "ensemble_overtuning"),
    ("1SE rule", "1SE rule", "cv", "ensembled_test_loss", "ensemble_overtuning"),
    ("Surrogate mean (CV)", "Surrogate mean", "cv", "ensembled_test_loss", "ensemble_overtuning"),
    ("Surr. mean+reshuf. (CV)", SURROGATE_RESHUFFLING_LABEL, "cv", "ensembled_test_loss", "ensemble_overtuning"),
    ("Post-hoc ensembling (CV)", "Post-hoc ensembling", "cv", "ensembled_test_loss", "ensemble_overtuning"),
    ("Makarova", "Makarova", "cv", "ensembled_test_loss", "ensemble_overtuning"),
    ("ML-Plan", MLPLAN_LABEL, "cv", "ensembled_test_loss", "ensemble_overtuning"),
    ("Reshuffling (CV)", "Reshuffling", "cv", "ensembled_test_loss", "ensemble_overtuning"),
    ("5$\\times$5CV+Bergm.", BERGMAN_LABEL, "cv", "ensembled_test_loss", "ensemble_overtuning"),
    ("5$\\times$5CV+Bergm.+reshuf.", BERGMAN_RESHUFFLING_LABEL, "cv", "ensembled_test_loss", "ensemble_overtuning"),
    # Holdout methods
    ("Baseline (H)", BASELINE_LABEL, "holdout", "retrain_test_loss", "retrain_overtuning"),
    ("Selection set (H)", "Selection set", "holdout", "retrain_test_loss", "retrain_overtuning"),
    ("Surrogate mean (H)", "Surrogate mean", "holdout", "retrain_test_loss", "retrain_overtuning"),
    ("Surr. mean+reshuf. (H)", SURROGATE_RESHUFFLING_LABEL, "holdout", "retrain_test_loss", "retrain_overtuning"),
    ("Post-hoc ensembling (H)", "Post-hoc ensembling", "holdout", "retrain_test_loss", "retrain_overtuning"),
    ("Reshuffling (H)", "Reshuffling", "holdout", "retrain_test_loss", "retrain_overtuning"),
    ("Thresholdout", "Thresholdout", "holdout", "retrain_test_loss", "retrain_overtuning"),
]


def _mask_for_method(df: pd.DataFrame, color_key: str, budget: str) -> pd.Series:
    dispatch = {
        (BASELINE_LABEL, "cv"): lambda: combined_benchmark_mask(df, "cv"),
        (BASELINE_LABEL, "holdout"): lambda: combined_benchmark_mask(df, "holdout"),
        ("Selection set", "cv"): lambda: selection_set_mask(df, "cv"),
        ("Selection set", "holdout"): lambda: selection_set_mask(df, "holdout"),
        ("1SE rule", "cv"): lambda: one_se_mask(df, "cv"),
        ("Surrogate mean", "cv"): lambda: post_hoc_surrogate_mask(df, "cv"),
        ("Surrogate mean", "holdout"): lambda: post_hoc_surrogate_mask(df, "holdout"),
        (SURROGATE_RESHUFFLING_LABEL, "cv"): lambda: post_hoc_surrogate_reshuffling_mask(df, "cv"),
        (SURROGATE_RESHUFFLING_LABEL, "holdout"): lambda: post_hoc_surrogate_reshuffling_mask(df, "holdout"),
        ("Post-hoc ensembling", "cv"): lambda: post_hoc_ensemble_mask(df, "cv"),
        ("Post-hoc ensembling", "holdout"): lambda: post_hoc_ensemble_mask(df, "holdout"),
        ("Makarova", "cv"): lambda: makarova_mask(df, "cv"),
        (MLPLAN_LABEL, "cv"): lambda: mlplan_mask(df),
        ("Reshuffling", "cv"): lambda: combined_reshuffling_mask(df, "cv"),
        ("Reshuffling", "holdout"): lambda: combined_reshuffling_mask(df, "holdout"),
        (BERGMAN_LABEL, "cv"): lambda: combined_bergman_mask(df, "cv"),
        (BERGMAN_RESHUFFLING_LABEL, "cv"): lambda: combined_bergman_reshuffling_mask(df, "cv"),
        ("Thresholdout", "holdout"): lambda: combined_thresholdout_mask(df, "holdout"),
    }
    return dispatch[(color_key, budget)]()


_METRIC_DISPLAY: dict[str, tuple[str, bool, bool]] = {
    # metric_name -> (display_name, lower_is_better_for_display, flip_sign)
    "roc_auc": ("AUC", False, True),       # stored as 1-AUC loss; flip to AUC ↑
    "neg_log_loss": ("logloss", True, False),
    "neg_root_mean_squared_error": ("RMSE", True, False),
}


def _decimals_for_scale(representative: float) -> int:
    """Choose decimal places based on a representative value for an entire column."""
    v = abs(representative)
    if v < 0.01:
        return 4
    if v < 1.0:
        return 3
    if v < 100.0:
        return 2
    if v < 10000.0:
        return 1
    return 0


def _fmt_val(mean: float, std: float, decimals: int | None = None) -> str:
    if np.isnan(mean):
        return "--"
    if decimals is None:
        decimals = _decimals_for_scale(mean)
    return f"{mean:.{decimals}f}\\tiny{{±{std:.{decimals}f}}}"


def _latex_safe(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def generate_latex_summary_table(
    df: pd.DataFrame,
    dataset_name_map: dict[str, str],
    output_dir: Path,
) -> None:
    """Generate a CSV cache and a LaTeX file with two per-dataset tables side-by-side per page."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_df = final_rows_per_run(df)
    if final_df.empty:
        print("[latex_table] No final rows, skipping")
        return

    norm_did = final_df["dataset_id"].map(normalize_dataset_id_value)
    all_dids = sorted(norm_did.unique())

    dataset_meta: dict[str, tuple[str, str]] = {}
    for did in all_dids:
        sub = final_df.loc[norm_did == did]
        metric = str(sub["metric"].iloc[0]) if "metric" in sub.columns else ""
        problem = str(sub["problem_type"].iloc[0]) if "problem_type" in sub.columns else ""
        dataset_meta[did] = (metric, problem)

    masks: dict[tuple[str, str], pd.Series] = {}
    for _, color_key, budget, _, _ in _ALL_METHODS:
        key = (color_key, budget)
        if key not in masks:
            masks[key] = _mask_for_method(final_df, color_key, budget)

    # --- Phase 1: build CSV ---
    csv_rows: list[dict] = []
    for did in all_dids:
        did_mask = norm_did == did
        dname = dataset_name_map.get(did, did)
        metric_raw, problem_type = dataset_meta[did]
        metric_display, lower_is_better, flip_sign = _METRIC_DISPLAY.get(metric_raw, (metric_raw, True, False))
        for display_name, color_key, budget, perf_col, ot_col in _ALL_METHODS:
            mask = did_mask & masks[(color_key, budget)]
            subset = final_df.loc[mask]
            if subset.empty:
                continue
            perf = pd.to_numeric(subset[perf_col], errors="coerce").dropna()
            ot = pd.to_numeric(subset[ot_col], errors="coerce").dropna()
            if perf.empty:
                continue
            pm, ps = float(perf.mean()), float(perf.std()) if len(perf) > 1 else 0.0
            om = float(ot.mean()) if len(ot) else np.nan
            os_ = float(ot.std()) if len(ot) > 1 else 0.0
            csv_rows.append({
                "dataset_id": did,
                "dataset_name": dname,
                "metric": metric_raw,
                "metric_display": metric_display,
                "problem_type": problem_type,
                "method": display_name,
                "color_key": color_key,
                "budget": budget,
                "perf_mean": pm,
                "perf_std": ps,
                "ot_mean": om,
                "ot_std": os_,
            })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = output_dir / "summary_tables_per_dataset.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  [✓] {csv_path.name} ({len(csv_df)} rows)")

    # --- Phase 2: build LaTeX from CSV ---
    _render_latex_from_csv(csv_df, dataset_name_map, dataset_meta, output_dir)


def _render_latex_from_csv(
    csv_df: pd.DataFrame,
    dataset_name_map: dict[str, str],
    dataset_meta: dict[str, tuple[str, str]],
    output_dir: Path,
) -> None:
    """Render per-dataset LaTeX tables (two side-by-side per page) from the CSV data."""
    # Build color definitions
    color_defs: list[str] = []
    color_names: dict[str, str] = {}
    for _, color_key, _, _, _ in _ALL_METHODS:
        if color_key in color_names:
            continue
        safe = "mc" + color_key.replace(" ", "").replace("+", "p").replace("-", "m").replace(".", "")[:20]
        hex_c = method_color(color_key)
        color_defs.append(_latex_color_def(safe, hex_c))
        color_names[color_key] = safe

    all_dids = sorted(csv_df["dataset_id"].unique())

    # Build one mini-table string per dataset
    mini_tables: list[tuple[str, str]] = []  # (did, latex_str)
    for did in all_dids:
        dname = dataset_name_map.get(did, did)
        metric_raw, _problem = dataset_meta.get(did, ("", ""))
        metric_display, lower_is_better, flip_sign = _METRIC_DISPLAY.get(metric_raw, (metric_raw, True, False))
        arrow = r"$\uparrow$" if not lower_is_better else r"$\downarrow$"
        caption_name = _latex_safe(dname)
        title = f"{caption_name} ({metric_display} {arrow})"

        dsub = csv_df[csv_df["dataset_id"] == did]
        if dsub.empty:
            continue

        rows: list[tuple[str, str, float, float, float, float]] = []
        for _, r in dsub.iterrows():
            pm, ps = float(r["perf_mean"]), float(r["perf_std"])
            if flip_sign:
                pm = 1.0 - pm
            rows.append((
                str(r["method"]),
                str(r["color_key"]),
                pm, ps,
                float(r["ot_mean"]), float(r["ot_std"]),
            ))

        # Best: for flipped metrics higher is better, otherwise lowest is best
        valid_perf = [(i, r[2]) for i, r in enumerate(rows) if not np.isnan(r[2])]
        valid_ot = [(i, r[4]) for i, r in enumerate(rows) if not np.isnan(r[4])]
        if flip_sign:
            best_perf_idx = max(valid_perf, key=lambda x: x[1])[0] if valid_perf else -1
        else:
            best_perf_idx = min(valid_perf, key=lambda x: x[1])[0] if valid_perf else -1
        best_ot_idx = min(valid_ot, key=lambda x: x[1])[0] if valid_ot else -1

        # Determine consistent decimal places per column using median value
        perf_vals = [r[2] for r in rows if not np.isnan(r[2])]
        ot_vals = [r[4] for r in rows if not np.isnan(r[4])]
        perf_dec = _decimals_for_scale(float(np.median(perf_vals))) if perf_vals else 3
        ot_dec = _decimals_for_scale(float(np.median(ot_vals))) if ot_vals else 4

        tbl: list[str] = []
        tbl.append(r"\begin{tabular}{lrr}")
        tbl.append(r"\toprule")
        tbl.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{title}}}}} \\\\")
        tbl.append(r"\midrule")
        tbl.append(r"Method & Test perform. & Overtuning \\")
        tbl.append(r"\midrule")

        cv_ended = False
        for i, (display_name, ck, pm, ps, om, os_) in enumerate(rows):
            if not cv_ended and "(H)" in display_name:
                tbl.append(r"\midrule")
                cv_ended = True
            cname = color_names.get(ck, "black")
            perf_str = _fmt_val(pm, ps, perf_dec)
            ot_str = _fmt_val(om, os_, ot_dec)
            if i == best_perf_idx:
                perf_str = f"\\underline{{{perf_str}}}"
            if i == best_ot_idx:
                ot_str = f"\\underline{{{ot_str}}}"
            safe_dn = _latex_safe(display_name)
            tbl.append(f"\\textcolor{{{cname}}}{{{safe_dn}}} & {perf_str} & {ot_str} \\\\")

        tbl.append(r"\bottomrule")
        tbl.append(r"\end{tabular}")
        mini_tables.append((did, "\n".join(tbl)))

    # Assemble final document: two tables side-by-side per page
    lines: list[str] = []
    lines.append("% Auto-generated per-dataset summary tables")
    lines.append("% Requires: \\usepackage{booktabs,xcolor,caption}")
    lines.extend(color_defs)
    lines.append("")

    for i in range(0, len(mini_tables), 2):
        left_did, left_tbl = mini_tables[i]
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\footnotesize")
        lines.append(r"\begin{minipage}[t]{0.47\textwidth}")
        lines.append(r"\centering")
        lines.append(left_tbl)
        lines.append(r"\end{minipage}")

        if i + 1 < len(mini_tables):
            right_did, right_tbl = mini_tables[i + 1]
            lines.append(r"\hfill")
            lines.append(r"\begin{minipage}[t]{0.47\textwidth}")
            lines.append(r"\centering")
            lines.append(right_tbl)
            lines.append(r"\end{minipage}")

        lines.append(r"\end{table}")
        lines.append("")

    output_path = output_dir / "summary_tables_per_dataset.tex"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [✓] {output_path.name} ({len(mini_tables)} dataset tables, {(len(mini_tables)+1)//2} pages)")


def generate_latex_detailed_tables(
    df: pd.DataFrame,
    dataset_name_map: dict[str, str],
    output_dir: Path,
) -> None:
    """Generate per-dataset/model/optimizer LaTeX tables (split by all dimensions)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_df = final_rows_per_run(df)
    if final_df.empty:
        print("[latex_table_detailed] No final rows, skipping")
        return

    norm_did = final_df["dataset_id"].map(normalize_dataset_id_value)

    masks: dict[tuple[str, str], pd.Series] = {}
    for _, color_key, budget, _, _ in _ALL_METHODS:
        key = (color_key, budget)
        if key not in masks:
            masks[key] = _mask_for_method(final_df, color_key, budget)

    # Color definitions
    color_defs: list[str] = []
    color_names: dict[str, str] = {}
    for _, color_key, _, _, _ in _ALL_METHODS:
        if color_key in color_names:
            continue
        safe = "mc" + color_key.replace(" ", "").replace("+", "p").replace("-", "m").replace(".", "")[:20]
        hex_c = method_color(color_key)
        color_defs.append(_latex_color_def(safe, hex_c))
        color_names[color_key] = safe

    all_dids = sorted(norm_did.unique())
    models = sorted(final_df["model"].dropna().unique()) if "model" in final_df.columns else ["all"]
    optimizers = sorted(final_df["optimizer"].dropna().unique()) if "optimizer" in final_df.columns else ["all"]

    # Save CSV
    csv_rows: list[dict] = []
    for did in all_dids:
        did_mask = norm_did == did
        dname = dataset_name_map.get(did, did)
        metric_raw = str(final_df.loc[did_mask, "metric"].iloc[0]) if "metric" in final_df.columns else ""
        for model in models:
            model_mask = final_df["model"].astype(str).eq(model) if "model" in final_df.columns else pd.Series(True, index=final_df.index)
            for opt in optimizers:
                opt_mask = final_df["optimizer"].astype(str).eq(opt) if "optimizer" in final_df.columns else pd.Series(True, index=final_df.index)
                for display_name, color_key, budget, perf_col, ot_col in _ALL_METHODS:
                    mask = did_mask & model_mask & opt_mask & masks[(color_key, budget)]
                    subset = final_df.loc[mask]
                    if subset.empty:
                        continue
                    perf = pd.to_numeric(subset[perf_col], errors="coerce").dropna()
                    ot = pd.to_numeric(subset[ot_col], errors="coerce").dropna()
                    if perf.empty:
                        continue
                    csv_rows.append({
                        "dataset_id": did, "dataset_name": dname, "metric": metric_raw,
                        "model": model, "optimizer": opt,
                        "method": display_name, "color_key": color_key, "budget": budget,
                        "perf_mean": float(perf.mean()),
                        "perf_std": float(perf.std()) if len(perf) > 1 else 0.0,
                        "ot_mean": float(ot.mean()) if len(ot) else np.nan,
                        "ot_std": float(ot.std()) if len(ot) > 1 else 0.0,
                    })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = output_dir / "summary_tables_detailed.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  [✓] {csv_path.name} ({len(csv_df)} rows)")

    # Build LaTeX
    lines: list[str] = []
    lines.append("% Auto-generated detailed per-dataset/model/optimizer tables")
    lines.append("% Requires: \\usepackage{booktabs,xcolor}")
    lines.extend(color_defs)
    lines.append("")

    n_tables = 0
    for did in all_dids:
        dname = dataset_name_map.get(did, did)
        metric_raw = str(csv_df.loc[csv_df["dataset_id"] == did, "metric"].iloc[0]) if did in csv_df["dataset_id"].values else ""
        metric_display, lower_is_better, flip_sign = _METRIC_DISPLAY.get(metric_raw, (metric_raw, True, False))
        arrow = r"$\uparrow$" if not lower_is_better else r"$\downarrow$"

        for model in models:
            for opt in optimizers:
                dsub = csv_df[(csv_df["dataset_id"] == did) & (csv_df["model"] == model) & (csv_df["optimizer"] == opt)]
                if dsub.empty:
                    continue

                rows: list[tuple[str, str, float, float, float, float]] = []
                for _, r in dsub.iterrows():
                    pm, ps = float(r["perf_mean"]), float(r["perf_std"])
                    if flip_sign:
                        pm = 1.0 - pm
                    rows.append((str(r["method"]), str(r["color_key"]), pm, ps, float(r["ot_mean"]), float(r["ot_std"])))

                if not rows:
                    continue

                valid_perf = [(i, r[2]) for i, r in enumerate(rows) if not np.isnan(r[2])]
                valid_ot = [(i, r[4]) for i, r in enumerate(rows) if not np.isnan(r[4])]
                if flip_sign:
                    best_perf_idx = max(valid_perf, key=lambda x: x[1])[0] if valid_perf else -1
                else:
                    best_perf_idx = min(valid_perf, key=lambda x: x[1])[0] if valid_perf else -1
                best_ot_idx = min(valid_ot, key=lambda x: x[1])[0] if valid_ot else -1

                perf_vals = [r[2] for r in rows if not np.isnan(r[2])]
                ot_vals = [r[4] for r in rows if not np.isnan(r[4])]
                perf_dec = _decimals_for_scale(float(np.median(perf_vals))) if perf_vals else 3
                ot_dec = _decimals_for_scale(float(np.median(ot_vals))) if ot_vals else 4

                title = f"{_latex_safe(dname)} ({metric_display} {arrow}) — {model}, {opt.upper()}"

                tbl: list[str] = []
                tbl.append(r"\begin{tabular}{lrr}")
                tbl.append(r"\toprule")
                tbl.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{title}}}}} \\\\")
                tbl.append(r"\midrule")
                tbl.append(r"Method & Test perform. & Overtuning \\")
                tbl.append(r"\midrule")

                cv_ended = False
                for i, (display_name, ck, pm, ps, om, os_) in enumerate(rows):
                    if not cv_ended and "(H)" in display_name:
                        tbl.append(r"\midrule")
                        cv_ended = True
                    cname = color_names.get(ck, "black")
                    perf_str = _fmt_val(pm, ps, perf_dec)
                    ot_str = _fmt_val(om, os_, ot_dec)
                    if i == best_perf_idx:
                        perf_str = f"\\underline{{{perf_str}}}"
                    if i == best_ot_idx:
                        ot_str = f"\\underline{{{ot_str}}}"
                    safe_dn = _latex_safe(display_name)
                    tbl.append(f"\\textcolor{{{cname}}}{{{safe_dn}}} & {perf_str} & {ot_str} \\\\")

                tbl.append(r"\bottomrule")
                tbl.append(r"\end{tabular}")

                lines.append(r"\begin{table}[htbp]")
                lines.append(r"\centering")
                lines.append(r"\footnotesize")
                lines.append("\n".join(tbl))
                lines.append(r"\end{table}")
                lines.append("")
                n_tables += 1

    output_path = output_dir / "summary_tables_detailed.tex"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [✓] {output_path.name} ({n_tables} tables)")
