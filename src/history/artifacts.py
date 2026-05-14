from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


ARTIFACT_ARCHIVE_NAME = "artifacts.npz"

_LEGACY_ARTIFACT_SPECS = {
    "test_labels": ("test_labels.npz", "preds"),
    "test_preds": ("test_preds.npz", "preds"),
    "val_preds": ("val.npz", "preds"),
    "val_labels": ("val.npz", "labels"),
    "selection_preds": ("sel.npz", "preds"),
    "selection_labels": ("sel.npz", "labels"),
}


def build_artifact_key(iteration: int, fold: int, name: str) -> str:
    """Return the deterministic key used inside the aggregated archive."""
    return f"iter_{iteration}/fold_{fold}/{name}"


def save_run_artifacts(output_path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Persist the aggregated prediction archive for one HPO run."""
    np.savez_compressed(output_path, **arrays)


def build_run_artifact_archive(history_runs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Collect all per-fold arrays into one compressed archive payload."""
    if not history_runs:
        return {}, {}

    # Object to store all predictions and labels
    arrays: dict[str, np.ndarray] = {}
    arrays["test_labels"] = history_runs[0].folds[0].labels.test

    # Object to store all configurations
    configs = {
        f"iter_{run.iteration}": {str(k): str(v) for k, v in dict(run.config).items()} for run in history_runs
    }

    for run in history_runs:
        # Store test predictions of retrained model
        arrays[f"iter_{run.iteration}/retrain_test_preds"] = run.retrain.preds.test

        for i_fold, fold in enumerate(run.folds):
            arrays[build_artifact_key(run.iteration, i_fold, "test_preds")] = fold.preds.test
            arrays[build_artifact_key(run.iteration, i_fold, "val_preds")] = fold.preds.val
            arrays[build_artifact_key(run.iteration, i_fold, "val_labels")] = fold.labels.val

            if fold.scores.selection is not None:
                arrays[build_artifact_key(run.iteration, i_fold, "selection_preds")] = fold.preds.selection
                arrays[build_artifact_key(run.iteration, i_fold, "selection_labels")] = fold.labels.selection

    return arrays, configs


def load_run_artifact(
    path: str | Path,
    iteration: int | None,
    fold: int | None,
    name: str,
) -> np.ndarray:
    """Load an artifact from the aggregated archive and fall back to the legacy layout."""
    base_path = Path(path)
    archive_path = base_path / "artifacts.npz" if base_path.is_dir() else base_path
    run_root = base_path if base_path.is_dir() else base_path.parent

    archive_key = "test_labels" if name == "test_labels" else build_artifact_key(iteration, fold, name)
    if archive_path.exists():
        with np.load(archive_path) as archive:
            if archive_key in archive:
                return archive[archive_key]

    legacy_file_name, legacy_key = _LEGACY_ARTIFACT_SPECS[name]
    if name == "test_labels":
        legacy_path = run_root / legacy_file_name
    else:
        if iteration is None or fold is None:
            raise KeyError(f"Artifact '{name}' requires iteration and fold")
        legacy_path = run_root / str(iteration) / str(fold) / legacy_file_name

    if not legacy_path.exists():
        raise FileNotFoundError(f"Artifact '{name}' not found in '{archive_path}' or '{legacy_path}'")

    with np.load(legacy_path) as legacy_archive:
        if legacy_key not in legacy_archive:
            raise KeyError(f"Artifact '{name}' missing key '{legacy_key}' in '{legacy_path}'")
        return legacy_archive[legacy_key]