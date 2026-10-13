import numpy as np
import pandas as pd
import pytest

from src.visualizations.plotting.filters import final_rows_per_run, run_key_columns


def _reference_final_rows_per_run(df: pd.DataFrame, iteration_col: str = "iteration") -> pd.DataFrame:
    """Previous implementation (full-frame copy); the new one must return the same frame."""
    if df.empty or iteration_col not in df.columns:
        return pd.DataFrame()
    run_cols = run_key_columns(df)
    work = df.copy()
    work["__iter_num"] = pd.to_numeric(work[iteration_col], errors="coerce")
    valid = work.loc[work["__iter_num"].notna()]
    if valid.empty:
        return pd.DataFrame(columns=df.columns)
    if run_cols:
        idx = valid.groupby(run_cols, dropna=False, observed=True)["__iter_num"].idxmax()
    else:
        idx = pd.Index([valid["__iter_num"].idxmax()])
    return work.loc[idx].drop(columns=["__iter_num"]).reset_index(drop=True)


def _trajectories(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 400
    df = pd.DataFrame({
        "trajectory_source": rng.choice(["default", "one_se", "mlplan"], n),
        "experiment_id": rng.choice([f"exp{i}" for i in range(12)], n),
        "dataset_id": rng.choice([1, 2, 3], n),
        "optimizer": pd.Categorical(rng.choice(["smac", "hebo"], n), categories=["smac", "hebo", "unused"]),
        "inner_split": rng.choice(["cv", "holdout", None], n),
        "reshuffling": rng.choice([True, False], n),
        "selection_set_size": rng.choice([np.nan, 0.2], n),
        # Ties and missing/non-numeric iterations.
        "iteration": rng.choice(["0", "1", "2", "2", "x", None], n),
        "retrain_test_loss": rng.random(n).astype("float32"),
    })
    df.index = rng.permutation(np.arange(1000, 1000 + n))
    return df


@pytest.mark.parametrize("seed", range(5))
def test_final_rows_per_run_matches_reference(seed):
    df = _trajectories(seed)
    pd.testing.assert_frame_equal(final_rows_per_run(df), _reference_final_rows_per_run(df))


def test_final_rows_per_run_edge_cases():
    df = _trajectories(0)
    for case in (df.iloc[:0], df.assign(iteration=None), df[["iteration", "retrain_test_loss"]]):
        pd.testing.assert_frame_equal(final_rows_per_run(case), _reference_final_rows_per_run(case))
    assert final_rows_per_run(df.drop(columns="iteration")).empty
