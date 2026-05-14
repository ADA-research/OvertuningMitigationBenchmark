from pathlib import Path

import pandas as pd


TRAJECTORY_FILES = {
    "default": "trajectories.csv",
    "surrogate": "trajectories_surrogate_selection.csv",
    "post_hoc_surrogate": "trajectories_post_hoc_surrogate.csv",
    "posthoc_surrogate": "trajectories_post_hoc_surrogate.csv",
    "post_hoc_ensemble": "trajectories_post_hoc_ensemble.csv",
    "posthoc_ensemble": "trajectories_post_hoc_ensemble.csv",
    "one_se": "trajectories_one_se_selection.csv",
    "makarova": "trajectories_makarova.csv",
    "mlplan": "trajectories_mlplan.csv",
}


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


def resolve_trajectory_file(source: str) -> str:
    """Resolve a source key or direct file name to a trajectory CSV file name."""
    key = source.strip().lower()
    return TRAJECTORY_FILES.get(key, source)


def load_trajectories(
    data_dir: Path,
    file_name: str = "trajectories.csv",
    usecols: list[str] | None = None,
    optimize_memory: bool = True,
) -> pd.DataFrame:
    """Load a trajectories CSV from the preprocessing output directory."""
    csv_path = data_dir / file_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Trajectories file not found: {csv_path}")
    if usecols is None:
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        try:
            df = pd.read_csv(csv_path, low_memory=False, usecols=usecols)
        except ValueError:
            header = pd.read_csv(csv_path, nrows=0)
            present = [col for col in usecols if col in header.columns]
            if not present:
                raise
            df = pd.read_csv(csv_path, low_memory=False, usecols=present)
    if optimize_memory:
        df = _optimize_trajectory_dtypes(df)
    return df


def load_trajectory_source(
    data_dir: Path,
    source: str = "default",
    usecols: list[str] | None = None,
    optimize_memory: bool = True,
) -> pd.DataFrame:
    """Load trajectories by source key (default/surrogate/one_se) or file name."""
    return load_trajectories(
        data_dir=data_dir,
        file_name=resolve_trajectory_file(source),
        usecols=usecols,
        optimize_memory=optimize_memory,
    )


def load_trajectory_sources(
    data_dir: Path,
    sources: list[str] | None = None,
    usecols: list[str] | None = None,
    optimize_memory: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load multiple trajectory sources and return a mapping from source key to dataframe."""
    if sources is None:
        sources = ["default", "post_hoc_surrogate", "one_se"]

    loaded: dict[str, pd.DataFrame] = {}
    for source in sources:
        file_name = resolve_trajectory_file(source)
        key = source.strip().lower()
        loaded[key] = load_trajectories(
            data_dir=data_dir,
            file_name=file_name,
            usecols=usecols,
            optimize_memory=optimize_memory,
        )
    return loaded
