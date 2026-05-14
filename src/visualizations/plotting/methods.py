from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MethodSpec:
    label: str
    color: str
    marker: str


BERGMAN_LABEL = "5x5CV + Bergman"
BERGMAN_RESHUFFLING_LABEL = "5x5CV + Bergman +\nreshuffling"
BASELINE_LABEL = "Baseline"
MLPLAN_LABEL = "ML-Plan"


METHOD_COLORS: dict[str, str] = {
    BASELINE_LABEL: "#000000",
    MLPLAN_LABEL: "#575757",

    "Selection set": "#000075",
    "1SE rule": "#e6194B",
    "Surrogate mean": "#f58231",
    "Post-hoc ensembling": "#FF66CC",
    "Makarova": "#734B95",
    "Reshuffling": "#007FFF",
    BERGMAN_LABEL: "#50C878",
    BERGMAN_RESHUFFLING_LABEL: "#228B22",
    "Thresholdout": "#9A6324",
}




_METHOD_COLOR_ALIASES: dict[str, str] = {
    "Baseline (holdout)": BASELINE_LABEL,
    "Baseline (CV)": BASELINE_LABEL,
    "Selection set (CV)": "Selection set",
    "Selection set (holdout)": "Selection set",
    "Surrogate mean (CV)": "Surrogate mean",
    "Surrogate mean (holdout)": "Surrogate mean",
    "Post-hoc ensembling (CV)": "Post-hoc ensembling",
    "Post-hoc ensembling (holdout)": "Post-hoc ensembling",
    "Reshuffling (CV)": "Reshuffling",
    "Reshuffling (holdout)": "Reshuffling",
    "5x5CV + Bergman + reshuffling": BERGMAN_RESHUFFLING_LABEL,
    "MLPlan": MLPLAN_LABEL,
}


def method_color(label: str, default: str = "#202020") -> str:
    canonical_label = _METHOD_COLOR_ALIASES.get(str(label), str(label))
    return METHOD_COLORS.get(canonical_label, default)


def _normalize_text(series: pd.Series) -> pd.Series:
    # Normalize text in a categorical-safe way without mutating category sets.
    # Casting through object avoids categorical fillna/setitem issues.
    as_obj = series.astype("object")
    return as_obj.where(pd.notna(as_obj), "").astype(str).str.strip().str.lower()


def _selection_zero_or_na(df: pd.DataFrame) -> pd.Series:
    if "selection_set_size" not in df.columns:
        return pd.Series(True, index=df.index)
    selection = pd.to_numeric(df["selection_set_size"], errors="coerce")
    return selection.isna() | (selection == 0.0)


def _selection_positive(df: pd.DataFrame) -> pd.Series:
    if "selection_set_size" not in df.columns:
        return pd.Series(False, index=df.index)
    selection = pd.to_numeric(df["selection_set_size"], errors="coerce")
    return selection.fillna(0.0) > 0.0


def _reshuffling_is(df: pd.DataFrame, target: bool) -> pd.Series:
    if "reshuffling" not in df.columns:
        return pd.Series(False, index=df.index)
    return _normalize_text(df["reshuffling"]) == ("true" if target else "false")


def _inner_split_is(df: pd.DataFrame, budget: str) -> pd.Series:
    if "inner_split" not in df.columns:
        return pd.Series(False, index=df.index)
    return _normalize_text(df["inner_split"]) == budget


def _source_is(df: pd.DataFrame, source: str) -> pd.Series:
    if "trajectory_source" not in df.columns:
        return pd.Series(False, index=df.index)
    return _normalize_text(df["trajectory_source"]) == source.strip().lower()


def _baseline_mitigation_mask(df: pd.DataFrame) -> pd.Series:
    mitigation = _normalize_text(df["mitigation"]) if "mitigation" in df.columns else pd.Series("", index=df.index)
    return mitigation.isin(["", "none"])


def benchmark_mask(df: pd.DataFrame, budget: str, reshuffling: bool) -> pd.Series:
    baseline_mitigation = _baseline_mitigation_mask(df)
    return baseline_mitigation & _selection_zero_or_na(df) & _inner_split_is(df, budget) & _reshuffling_is(df, reshuffling)


def bergman_mask(df: pd.DataFrame) -> pd.Series:
    mitigation = _normalize_text(df["mitigation"]) if "mitigation" in df.columns else pd.Series("", index=df.index)
    return (
        mitigation.str.contains("bergman", na=False)
        & _reshuffling_is(df, target=False)
        & _selection_zero_or_na(df)
    )


def bergman_reshuffling_mask(df: pd.DataFrame) -> pd.Series:
    mitigation = _normalize_text(df["mitigation"]) if "mitigation" in df.columns else pd.Series("", index=df.index)
    return (
        mitigation.str.contains("bergman", na=False)
        & _reshuffling_is(df, target=True)
        & _selection_zero_or_na(df)
    )


def thresholdout_mask(df: pd.DataFrame) -> pd.Series:
    mitigation = _normalize_text(df["mitigation"]) if "mitigation" in df.columns else pd.Series("", index=df.index)
    return (
        mitigation.str.contains("thresholdout|thesholdout", regex=True, na=False)
        & _reshuffling_is(df, target=False)
        & _selection_zero_or_na(df)
    )


def build_mitigation_cdf_methods(df: pd.DataFrame, budget: str) -> list[tuple[MethodSpec, pd.Series]]:
    budget_norm = budget.strip().lower()
    if budget_norm not in {"cv", "holdout"}:
        raise ValueError("budget must be 'cv' or 'holdout'")

    if budget_norm == "cv":
        return [
            (MethodSpec(label=BASELINE_LABEL, color=METHOD_COLORS[BASELINE_LABEL], marker="o"), benchmark_mask(df, budget="cv", reshuffling=False)),
            (MethodSpec(label="Reshuffling", color=METHOD_COLORS["Reshuffling"], marker="o"), benchmark_mask(df, budget="cv", reshuffling=True)),
            (MethodSpec(label=BERGMAN_LABEL, color=METHOD_COLORS[BERGMAN_LABEL], marker="s"), bergman_mask(df)),
            (MethodSpec(label=BERGMAN_RESHUFFLING_LABEL, color=METHOD_COLORS[BERGMAN_RESHUFFLING_LABEL], marker="X"), bergman_reshuffling_mask(df)),
        ]

    return [
        (MethodSpec(label=BASELINE_LABEL, color=METHOD_COLORS[BASELINE_LABEL], marker="o"), benchmark_mask(df, budget="holdout", reshuffling=False)),
        (MethodSpec(label="Thresholdout", color=METHOD_COLORS["Thresholdout"], marker="o"), thresholdout_mask(df)),
        (MethodSpec(label="Reshuffling", color=METHOD_COLORS["Reshuffling"], marker="o"), benchmark_mask(df, budget="holdout", reshuffling=True)),
    ]


def cv_delta_methods() -> list[tuple[MethodSpec, pd.Series]]:
    raise RuntimeError("Call build_cv_delta_methods(df) with a dataframe")


def build_cv_delta_methods(df: pd.DataFrame) -> list[tuple[MethodSpec, pd.Series]]:
    return [
        (MethodSpec(label="Reshuffling", color=METHOD_COLORS["Reshuffling"], marker="o"), benchmark_mask(df, budget="cv", reshuffling=True)),
        (MethodSpec(label=BERGMAN_LABEL, color=METHOD_COLORS[BERGMAN_LABEL], marker="s"), bergman_mask(df)),
        (MethodSpec(label=BERGMAN_RESHUFFLING_LABEL, color=METHOD_COLORS[BERGMAN_RESHUFFLING_LABEL], marker="X"), bergman_reshuffling_mask(df)),
    ]


def build_holdout_delta_methods(df: pd.DataFrame) -> list[tuple[MethodSpec, pd.Series]]:
    return [
        (MethodSpec(label="Reshuffling", color=METHOD_COLORS["Reshuffling"], marker="o"), benchmark_mask(df, budget="holdout", reshuffling=True)),
        (MethodSpec(label="Thresholdout", color=METHOD_COLORS["Thresholdout"], marker="s"), thresholdout_mask(df)),
    ]


def build_delta_methods(df: pd.DataFrame, budget: str) -> list[tuple[MethodSpec, pd.Series]]:
    budget_norm = budget.strip().lower()
    if budget_norm == "cv":
        return build_cv_delta_methods(df)
    if budget_norm == "holdout":
        return build_holdout_delta_methods(df)
    raise ValueError(f"budget must be 'cv' or 'holdout', got: {budget!r}")


def incumbent_benchmark_mask(df: pd.DataFrame, budget: str) -> pd.Series:
    return _source_is(df, "default") & benchmark_mask(df, budget=budget, reshuffling=False)


def selection_set_mask(df: pd.DataFrame, budget: str) -> pd.Series:
    return (
        _source_is(df, "default")
        & _baseline_mitigation_mask(df)
        & _selection_positive(df)
        & _reshuffling_is(df, target=False)
        & _inner_split_is(df, budget)
    )


def one_se_mask(df: pd.DataFrame, budget: str) -> pd.Series:
    return (
        _source_is(df, "one_se")
        & _baseline_mitigation_mask(df)
        & _selection_positive(df)
        & _reshuffling_is(df, target=False)
        & _inner_split_is(df, budget)
    )


def post_hoc_surrogate_mask(df: pd.DataFrame, budget: str) -> pd.Series:
    return (
        _source_is(df, "post_hoc_surrogate")
        & _baseline_mitigation_mask(df)
        & _reshuffling_is(df, target=False)
        & _inner_split_is(df, budget)
    )


def post_hoc_ensemble_mask(df: pd.DataFrame, budget: str) -> pd.Series:
    return (
        _source_is(df, "post_hoc_ensemble")
        & _baseline_mitigation_mask(df)
        & _reshuffling_is(df, target=False)
        & _inner_split_is(df, budget)
    )


def makarova_mask(df: pd.DataFrame, budget: str) -> pd.Series:
    return (
        _source_is(df, "makarova")
        & _baseline_mitigation_mask(df)
        & _selection_zero_or_na(df)
        & _reshuffling_is(df, target=False)
        & _inner_split_is(df, budget)
    )


def mlplan_mask(df: pd.DataFrame) -> pd.Series:
    """Match MLPlan final-result rows.

    MLPlan data lives in ``trajectories_mlplan.csv`` and gets
    ``trajectory_source='mlplan'`` when loaded.  No budget argument is needed
    since MLPlan uses its own two-phase protocol.
    """
    return _source_is(df, "mlplan")


def build_incumbent_cdf_methods(df: pd.DataFrame, budget: str) -> list[tuple[MethodSpec, pd.Series]]:
    methods = [
        (MethodSpec(label=BASELINE_LABEL, color=METHOD_COLORS[BASELINE_LABEL], marker="o"), incumbent_benchmark_mask(df, budget)),
        (MethodSpec(label="Selection set", color=METHOD_COLORS["Selection set"], marker="o"), selection_set_mask(df, budget)),
        (MethodSpec(label="Surrogate mean", color=METHOD_COLORS["Surrogate mean"], marker="D"), post_hoc_surrogate_mask(df, budget)),
        (MethodSpec(label="Post-hoc ensembling", color=METHOD_COLORS["Post-hoc ensembling"], marker="P"), post_hoc_ensemble_mask(df, budget)),
    ]
    budget_norm = budget.strip().lower()
    if budget_norm == "cv":
        methods.insert(2, (MethodSpec(label="1SE rule", color=METHOD_COLORS["1SE rule"], marker="s"), one_se_mask(df, budget)))
        methods.append((MethodSpec(label="Makarova", color=METHOD_COLORS["Makarova"], marker="^"), makarova_mask(df, budget)))
    return methods


def build_incumbent_delta_methods(df: pd.DataFrame, budget: str) -> list[tuple[MethodSpec, pd.Series]]:
    methods = [
        (MethodSpec(label="Selection set", color=METHOD_COLORS["Selection set"], marker="o"), selection_set_mask(df, budget)),
        (MethodSpec(label="Surrogate mean", color=METHOD_COLORS["Surrogate mean"], marker="D"), post_hoc_surrogate_mask(df, budget)),
        (MethodSpec(label="Post-hoc ensembling", color=METHOD_COLORS["Post-hoc ensembling"], marker="P"), post_hoc_ensemble_mask(df, budget)),
    ]
    budget_norm = budget.strip().lower()
    if budget_norm == "cv":
        methods.insert(1, (MethodSpec(label="1SE rule", color=METHOD_COLORS["1SE rule"], marker="s"), one_se_mask(df, budget)))
        methods.append((MethodSpec(label="Makarova", color=METHOD_COLORS["Makarova"], marker="^"), makarova_mask(df, budget)))
    return methods


def combined_benchmark_mask(df: pd.DataFrame, budget: str) -> pd.Series:
    return _source_is(df, "default") & benchmark_mask(df, budget=budget, reshuffling=False)


def combined_reshuffling_mask(df: pd.DataFrame, budget: str) -> pd.Series:
    return _source_is(df, "default") & benchmark_mask(df, budget=budget, reshuffling=True)


def combined_bergman_mask(df: pd.DataFrame, budget: str) -> pd.Series:
    return _source_is(df, "default") & bergman_mask(df)


def combined_bergman_reshuffling_mask(df: pd.DataFrame, budget: str) -> pd.Series:
    return _source_is(df, "default") & bergman_reshuffling_mask(df)


def combined_thresholdout_mask(df: pd.DataFrame, budget: str) -> pd.Series:
    return _source_is(df, "default") & thresholdout_mask(df)


_ALL_METHOD_COLORS = METHOD_COLORS


def build_all_methods_cdf(df: pd.DataFrame, budget: str) -> list[tuple[MethodSpec, pd.Series]]:
    budget_norm = budget.strip().lower()
    if budget_norm == "cv":
        return [
            (MethodSpec(label="Selection set", color=_ALL_METHOD_COLORS["Selection set"], marker="o"), selection_set_mask(df, budget)),
            (MethodSpec(label="1SE rule", color=_ALL_METHOD_COLORS["1SE rule"], marker="s"), one_se_mask(df, budget)),
            (MethodSpec(label="Surrogate mean", color=_ALL_METHOD_COLORS["Surrogate mean"], marker="D"), post_hoc_surrogate_mask(df, budget)),
            (MethodSpec(label="Post-hoc ensembling", color=_ALL_METHOD_COLORS["Post-hoc ensembling"], marker="P"), post_hoc_ensemble_mask(df, budget)),
            (MethodSpec(label="Makarova", color=_ALL_METHOD_COLORS["Makarova"], marker="^"), makarova_mask(df, budget)),
            (MethodSpec(label="Reshuffling", color=_ALL_METHOD_COLORS["Reshuffling"], marker="o"), combined_reshuffling_mask(df, budget)),
            (MethodSpec(label=BERGMAN_LABEL, color=_ALL_METHOD_COLORS[BERGMAN_LABEL], marker="s"), combined_bergman_mask(df, budget)),
            (MethodSpec(label=BERGMAN_RESHUFFLING_LABEL, color=_ALL_METHOD_COLORS[BERGMAN_RESHUFFLING_LABEL], marker="X"), combined_bergman_reshuffling_mask(df, budget)),
        ]
    if budget_norm == "holdout":
        return [
            (MethodSpec(label="Selection set", color=_ALL_METHOD_COLORS["Selection set"], marker="o"), selection_set_mask(df, budget)),
            (MethodSpec(label="Surrogate mean", color=_ALL_METHOD_COLORS["Surrogate mean"], marker="D"), post_hoc_surrogate_mask(df, budget)),
            (MethodSpec(label="Post-hoc ensembling", color=_ALL_METHOD_COLORS["Post-hoc ensembling"], marker="P"), post_hoc_ensemble_mask(df, budget)),
            (MethodSpec(label="Reshuffling", color=_ALL_METHOD_COLORS["Reshuffling"], marker="o"), combined_reshuffling_mask(df, budget)),
            (MethodSpec(label="Thresholdout", color=_ALL_METHOD_COLORS["Thresholdout"], marker="o"), combined_thresholdout_mask(df, budget)),
        ]
    raise ValueError(f"budget must be 'cv' or 'holdout', got: {budget!r}")


def build_all_methods_cdf_with_benchmark(df: pd.DataFrame, budget: str) -> list[tuple[MethodSpec, pd.Series]]:
    # Benchmark must be the default-source, non-reshuffled, no-selection baseline for the same budget.
    return [
        (MethodSpec(label=BASELINE_LABEL, color=METHOD_COLORS[BASELINE_LABEL], marker="o"), combined_benchmark_mask(df, budget)),
        *build_all_methods_cdf(df, budget),
    ]


def build_all_methods_delta(df: pd.DataFrame, budget: str) -> list[tuple[MethodSpec, pd.Series]]:
    budget_norm = budget.strip().lower()
    if budget_norm == "cv":
        return [
            (MethodSpec(label="Selection set", color=_ALL_METHOD_COLORS["Selection set"], marker="o"), selection_set_mask(df, budget)),
            (MethodSpec(label="1SE rule", color=_ALL_METHOD_COLORS["1SE rule"], marker="s"), one_se_mask(df, budget)),
            (MethodSpec(label="Surrogate mean", color=_ALL_METHOD_COLORS["Surrogate mean"], marker="D"), post_hoc_surrogate_mask(df, budget)),
            (MethodSpec(label="Post-hoc ensembling", color=_ALL_METHOD_COLORS["Post-hoc ensembling"], marker="P"), post_hoc_ensemble_mask(df, budget)),
            (MethodSpec(label="Makarova", color=_ALL_METHOD_COLORS["Makarova"], marker="^"), makarova_mask(df, budget)),
            (MethodSpec(label="Reshuffling", color=_ALL_METHOD_COLORS["Reshuffling"], marker="o"), combined_reshuffling_mask(df, budget)),
            (MethodSpec(label=BERGMAN_LABEL, color=_ALL_METHOD_COLORS[BERGMAN_LABEL], marker="s"), combined_bergman_mask(df, budget)),
            (MethodSpec(label=BERGMAN_RESHUFFLING_LABEL, color=_ALL_METHOD_COLORS[BERGMAN_RESHUFFLING_LABEL], marker="X"), combined_bergman_reshuffling_mask(df, budget)),
        ]
    if budget_norm == "holdout":
        return [
            (MethodSpec(label="Selection set", color=_ALL_METHOD_COLORS["Selection set"], marker="o"), selection_set_mask(df, budget)),
            (MethodSpec(label="Surrogate mean", color=_ALL_METHOD_COLORS["Surrogate mean"], marker="D"), post_hoc_surrogate_mask(df, budget)),
            (MethodSpec(label="Post-hoc ensembling", color=_ALL_METHOD_COLORS["Post-hoc ensembling"], marker="P"), post_hoc_ensemble_mask(df, budget)),
            (MethodSpec(label="Reshuffling", color=_ALL_METHOD_COLORS["Reshuffling"], marker="o"), combined_reshuffling_mask(df, budget)),
            (MethodSpec(label="Thresholdout", color=_ALL_METHOD_COLORS["Thresholdout"], marker="o"), combined_thresholdout_mask(df, budget)),
        ]
    raise ValueError(f"budget must be 'cv' or 'holdout', got: {budget!r}")
