"""Combined preprocessing outputs: gzip writing (preprocessing) and reading (plotting loader)."""

import os

import numpy as np
import pandas as pd
import pytest

from src.visualizations.plotting.io import _optimize_trajectory_dtypes, load_trajectory_sources
from src.visualizations.preprocessing.combine_normalization import (
    apply_normalization,
    build_normalization_ranges,
)
from src.visualizations.preprocessing.preprocessing import (
    TRAJECTORY_FILES_BY_METHOD,
    _dataset_unique_run_counts_by_model,
    _open_csv_gz,
    append_csv_file,
    preprocess_multiple_experiments,
)
from src.visualizations.preprocessing.pipeline import discover_experiment_dirs
from src.visualizations.preprocessing.shared_utils import RESULTS_FILE, TRAJECTORIES_FILE

GZIP_MAGIC = b"\x1f\x8b"
METHODS = list(TRAJECTORY_FILES_BY_METHOD)


def _write_experiment(exp_dir, dataset_id, offset):
    exp_dir.mkdir()
    group = {"dataset_id": dataset_id, "repetition": 0, "outer_fold": 0, "metric": "loss", "model": "LGBM"}
    pd.DataFrame({
        "iteration": [0, 0, 1, 1, 2, 2],
        "fold": [0, 1] * 3,
        "avg_val_score": offset + np.array([0.5, 0.5, 0.3, 0.3, 0.4, 0.4]),
        "retrained_test_score": offset + np.array([0.6, 0.6, 0.2, 0.2, 0.5, 0.5]),
        "cv_ensembled_test_score": offset + np.array([0.7, 0.7, 0.3, 0.3, 0.6, 0.6]),
        **group,
        "experiment_id": f"{exp_dir.name}::run",
    }).to_csv(exp_dir / "preprocessed_results.csv", index=False)

    for i, (method, file_name) in enumerate(TRAJECTORY_FILES_BY_METHOD.items()):
        pd.DataFrame({
            "iteration": [0, 1, 2],
            "experiment_id": f"{exp_dir.name}::run",
            **group,
            # MLPlan has no reshuffling setting, as in the real data.
            "reshuffling": np.nan if method == "mlplan" else [False, False, True],
            "val_performance": offset + np.array([0.5, 0.3, 0.3]),
            "retrain_test_loss": offset + np.array([0.6, 0.2, 0.2]) + 0.01 * i,
            "ensembled_test_loss": offset + np.array([0.7, 0.3, 0.3]),
            "retrain_overtuning": [0.0, 0.0, 0.1],
            "ensemble_overtuning": [0.0, 0.0, 0.2],
        }).to_csv(exp_dir / file_name, index=False)


    def test_discover_experiment_dirs_from_results_root(tmp_path):
        results_dir = tmp_path / "private_results"
        (results_dir / "LGBM_363700_20261012").mkdir(parents=True)
        (results_dir / "SMAC_363621_20261012").mkdir()
        (results_dir / "notes.txt").write_text("not an experiment directory")

        discovered = discover_experiment_dirs(results_dir)

        assert [path.name for path in discovered] == [
            "LGBM_363700_20261012",
            "SMAC_363621_20261012",
        ]


    def test_discover_experiment_dirs_requires_existing_root(tmp_path):
        with pytest.raises(FileNotFoundError, match="Full results directory not found"):
            discover_experiment_dirs(tmp_path / "missing")


@pytest.fixture
def combined(tmp_path):
    exp_dirs = [tmp_path / "exp_a", tmp_path / "exp_b"]
    _write_experiment(exp_dirs[0], dataset_id=1, offset=0.0)
    _write_experiment(exp_dirs[1], dataset_id=2, offset=1.0)
    # Raw post-hoc ensemble scores below the regular minimum extend the anchor range.
    pd.DataFrame({
        "dataset_id": [1], "repetition": [0], "outer_fold": [0], "metric": ["loss"], "model": ["LGBM"],
        "avg_val_score": [0.4], "retrained_test_score": [0.4], "cv_ensembled_test_score": [0.1],
    }).to_csv(exp_dirs[0] / "results_post_hoc_ensemble.csv", index=False)

    out_dir = tmp_path / "out"
    preprocess_multiple_experiments([str(d) for d in exp_dirs], output_path=str(out_dir), verbose=False)
    return exp_dirs, out_dir


def test_only_two_gzip_files_are_written(combined):
    _, out_dir = combined
    assert sorted(os.listdir(out_dir)) == sorted([RESULTS_FILE, TRAJECTORIES_FILE])
    for name in (RESULTS_FILE, TRAJECTORIES_FILE):
        assert (out_dir / name).read_bytes()[:2] == GZIP_MAGIC


def test_results_round_trip(combined):
    exp_dirs, out_dir = combined
    expected = pd.concat([pd.read_csv(d / "preprocessed_results.csv") for d in exp_dirs], ignore_index=True)
    actual = pd.read_csv(out_dir / RESULTS_FILE)
    pd.testing.assert_frame_equal(actual[expected.columns], expected)
    # Score aliases are added exactly as for the former plain-CSV output.
    pd.testing.assert_series_equal(actual["ensembled_test_loss"], expected["cv_ensembled_test_score"], check_names=False)


def test_results_write_logs_experiment_progress(combined, capsys):
    exp_dirs, out_dir = combined
    assert "[results]" not in capsys.readouterr().out
    preprocess_multiple_experiments(
        [str(directory) for directory in exp_dirs],
        output_path=str(out_dir),
        verbose=True,
    )
    output = capsys.readouterr().out
    assert "[results] Writing experiment" not in output
    assert "[results] Completed 1/2 (50.0%)" in output
    assert "[results] Completed 2/2 (100.0%)" in output
    assert "elapsed=" in output


def test_trajectory_write_logs_method_experiment_progress(combined, capsys):
    exp_dirs, out_dir = combined
    capsys.readouterr()
    preprocess_multiple_experiments(
        [str(directory) for directory in exp_dirs],
        output_path=str(out_dir),
        verbose=True,
    )
    output = capsys.readouterr().out
    assert "[trajectories] Completed 1/12 (8.3%); method=default, experiment=exp_a" in output
    assert "[trajectories] Completed 12/12 (100.0%); method=post_hoc_surrogate, experiment=exp_b" in output
    assert "rows=3; elapsed=" in output


def test_trajectories_are_normalized_per_method(combined):
    exp_dirs, out_dir = combined
    anchors = [pd.read_csv(d / "preprocessed_results.csv") for d in exp_dirs]
    anchors.append(pd.read_csv(exp_dirs[0] / "results_post_hoc_ensemble.csv"))
    ranges = build_normalization_ranges(pd.concat(anchors, ignore_index=True))

    actual = pd.read_csv(out_dir / TRAJECTORIES_FILE)
    assert list(actual["method"].unique()) == METHODS
    for method, file_name in TRAJECTORY_FILES_BY_METHOD.items():
        expected = pd.concat([pd.read_csv(d / file_name) for d in exp_dirs], ignore_index=True)
        expected["val_loss"] = expected["val_performance"]
        expected = apply_normalization(expected, ranges)
        method_rows = actual[actual["method"] == method].drop(columns="method").reset_index(drop=True).infer_objects()
        pd.testing.assert_frame_equal(method_rows[expected.columns], expected)

    # Anchor range for exp_a ensemble scores is [0.1 (post-hoc), 0.7]: (0.3 - 0.1) / 0.6.
    row = actual[(actual["method"] == "default") & (actual["dataset_id"] == 1) & (actual["iteration"] == 1)]
    assert row["normalized_ensembled_test_loss"].item() == pytest.approx(1 / 3)


def test_loader_matches_reading_separate_method_files(combined, tmp_path):
    _, out_dir = combined
    usecols = ["experiment_id", "dataset_id", "model", "iteration", "reshuffling",
               "retrain_test_loss", "normalized_ensembled_test_loss", "normalized_retrain_overtuning"]
    # A tiny chunk size makes chunks span two methods.
    loaded = load_trajectory_sources(out_dir, sources=METHODS, usecols=usecols, chunk_size=4)

    combined_df = pd.read_csv(out_dir / TRAJECTORIES_FILE)
    for method in METHODS:
        # Former behaviour: one plain CSV per method, read in full.
        method_csv = tmp_path / f"{method}.csv"
        combined_df[combined_df["method"] == method].drop(columns="method").to_csv(method_csv, index=False)
        expected = _optimize_trajectory_dtypes(pd.read_csv(method_csv, usecols=usecols, low_memory=False))
        pd.testing.assert_frame_equal(loaded[method], expected[loaded[method].columns])
    assert loaded["makarova"]["reshuffling"].dtype == bool


def test_csv_combine_bounds_chunks_and_releases_cache(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    output = tmp_path / "combined.csv.gz"
    expected = pd.DataFrame({"iteration": np.arange(25_001)})
    expected.to_csv(source, index=False)
    chunk_sizes = []
    cache_releases = []
    syncs = []
    monkeypatch.setattr(os, "posix_fadvise", lambda *args: cache_releases.append(args))
    monkeypatch.setattr(os, "fdatasync", lambda *args: syncs.append(args))

    def transform(chunk):
        chunk_sizes.append(len(chunk))
        return chunk

    with _open_csv_gz(output) as handle:
        handle.cache_flushed_size = -64 * 1024 * 1024
        append_csv_file(source, handle, False, transform=transform)

    assert chunk_sizes == [10_000, 10_000, 5_001]
    assert len(cache_releases) == 5
    assert len(syncs) == 2
    pd.testing.assert_frame_equal(pd.read_csv(output), expected)
    assert not output.with_name(output.name + ".tmp").exists()


def test_run_count_scan_releases_cache(combined, monkeypatch):
    _, out_dir = combined
    cache_releases = []
    monkeypatch.setattr(os, "posix_fadvise", lambda *args: cache_releases.append(args))
    counts, run_column = _dataset_unique_run_counts_by_model(
        out_dir / TRAJECTORIES_FILE, chunk_size=4,
    )
    assert counts == {"LGBM": {"1": 1, "2": 1}}
    assert run_column == "experiment_id"
    assert len(cache_releases) == 9
