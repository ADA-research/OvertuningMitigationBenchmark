from pathlib import Path

import pandas as pd

from src.visualizations.preprocessing.shared_utils import TRAJECTORIES_FILE


FLOAT_LIKE_COLUMNS = {
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
}


CATEGORICAL_LIKE_COLUMNS = {
    "dataset_id",
    "optimizer",
    "inner_split",
    "mitigation",
    "problem_type",
    "model",
    "trajectory_source",
}


def _optimize_trajectory_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast common plotting columns to reduce memory footprint."""
    out = df
    for col in FLOAT_LIKE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    for col in ("iteration", "repetition", "outer_fold"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int32")

    for col in CATEGORICAL_LIKE_COLUMNS:
        if col in out.columns and out[col].dtype == object:
            out[col] = out[col].astype("category")

    return out


def load_trajectory_source(
    data_dir: Path,
    source: str = "default",
    usecols: list[str] | None = None,
    optimize_memory: bool = True,
) -> pd.DataFrame:
    """Load the trajectories of one method (default/one_se/makarova/...)."""
    return load_trajectory_sources(data_dir, [source], usecols, optimize_memory)[source.strip().lower()]


def load_trajectory_sources(
    data_dir: Path,
    sources: list[str] | None = None,
    usecols: list[str] | None = None,
    optimize_memory: bool = True,
    chunk_size: int = 500_000,
) -> dict[str, pd.DataFrame]:
    """Load trajectories per method from the combined trajectories file.

    The file is grouped by method, so each method is finalized as soon as its
    rows end; only one method's raw rows are held in memory at a time.
    """
    if sources is None:
        sources = ["default", "post_hoc_surrogate", "one_se"]
    keys = [source.strip().lower() for source in sources]

    csv_path = data_dir / TRAJECTORIES_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"Trajectories file not found: {csv_path}")
    if usecols is not None:
        header = pd.read_csv(csv_path, nrows=0).columns
        usecols = ["method", *[col for col in usecols if col in header and col != "method"]]

    loaded: dict[str, pd.DataFrame] = {}
    parts: list[pd.DataFrame] = []
    seen: list[str] = []

    def finish_method() -> None:
        if seen and seen[-1] in keys:
            # Chunks spanning two methods can widen dtypes (e.g. bool -> object); restore them.
            df = pd.concat(parts, ignore_index=True).infer_objects()
            loaded[seen[-1]] = _optimize_trajectory_dtypes(df) if optimize_memory else df
        parts.clear()

    with pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, low_memory=False) as reader:
        for chunk in reader:
            for method, part in chunk.groupby("method", sort=False):
                if not seen or method != seen[-1]:
                    if method in seen:
                        raise ValueError(f"{csv_path} is not grouped by method ('{method}' reappears)")
                    finish_method()
                    seen.append(method)
                if method in keys:
                    parts.append(part.drop(columns="method"))
            if all(key in loaded for key in keys):
                break
    finish_method()

    missing = [key for key in keys if key not in loaded]
    if missing:
        raise ValueError(f"No trajectories for method(s) {missing} in {csv_path}")
    return {key: loaded[key] for key in keys}
