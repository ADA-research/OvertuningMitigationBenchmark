from pathlib import Path

import numpy as np

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


DATASET_INSTANCE_COUNTS: dict[str, int] = {
    "363698": 907,
    "363613": 32769,
    "363614": 898,
    "363625": 1030,
    "363616": 76000,
    "363618": 45211,
    "363619": 10000,
    "363620": 3751,
    "363621": 748,
    "363623": 5000,
    "363624": 9822,
    "363675": 1338,
    "363626": 1000,
    "363627": 30000,
    "363628": 129880,
    "363629": 768,
    "363630": 71518,
    "363612": 1503,
    "363632": 10999,
    "363671": 1500,
    "363615": 1538,
    "363673": 150000,
    "363674": 2400,
    "363697": 5742,
    "363676": 10459,
    "363685": 1014,
    "363708": 6497,
    "363679": 19158,
    "363681": 12684,
    "363682": 1723,
    "363683": 50000,
    "363684": 2240,
    "363707": 1353,
    "363686": 13776,
    "363689": 7491,
    "363691": 12330,
    "363678": 20640,
    "363694": 5910,
    "363696": 1054,
    "363705": 21263,
    "363672": 45451,
    "363711": 1699,
    "363700": 2584,
    "363702": 3190,
    "363677": 3845,
    "363693": 45730,
    "363706": 6819,
    "363704": 4424,
    "363631": 53940,
    "363699": 78053,
    "363712": 10885,
}

SMALL_DATASET_THRESHOLD = 2500


def get_experiment_dataset_id(experiment_dir_name: str) -> str | None:
    parts = experiment_dir_name.split("_")
    if len(parts) < 2:
        return None
    return parts[1]


def is_small_dataset(dataset_id: str) -> bool:
    n_instances = DATASET_INSTANCE_COUNTS.get(str(dataset_id))
    return n_instances is not None and n_instances < SMALL_DATASET_THRESHOLD


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values, axis=1, keepdims=True)
    return exp_values / denom


def coerce_predictions_for_metric(
        preds: np.ndarray,
        problem_type: str,
        labels: np.ndarray,
) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float64)

    if problem_type == "regression":
        return preds

    if problem_type == "binary":
        if preds.ndim == 1:
            if (preds < 0).any() or (preds > 1).any():
                preds = 1.0 / (1.0 + np.exp(-preds))
            return np.column_stack([1.0 - preds, preds])

        if preds.ndim == 2 and preds.shape[1] == 1:
            pos = preds[:, 0]
            if (pos < 0).any() or (pos > 1).any():
                pos = 1.0 / (1.0 + np.exp(-pos))
            return np.column_stack([1.0 - pos, pos])

        if preds.ndim == 2:
            binary_preds = preds[:, :2]
            row_sums = binary_preds.sum(axis=1, keepdims=True)
            if not np.allclose(row_sums, 1.0, atol=1e-20):
                if (binary_preds >= 0).all() and (row_sums > 0).all():
                    binary_preds = binary_preds / row_sums
                else:
                    binary_preds = _softmax_rows(binary_preds)
            return binary_preds

        raise ValueError(f"Unsupported binary prediction shape: {preds.shape}")

    n_classes = int(np.max(labels)) + 1

    if preds.ndim == 1:
        if np.allclose(preds, np.round(preds), atol=1e-20):
            cls = np.clip(preds.astype(int), 0, n_classes - 1)
            one_hot = np.zeros((len(cls), n_classes), dtype=np.float64)
            one_hot[np.arange(len(cls)), cls] = 1.0
            return one_hot
        raise ValueError(f"Unsupported 1D multiclass predictions for shape {preds.shape}")

    if preds.ndim != 2:
        raise ValueError(f"Unsupported multiclass prediction shape: {preds.shape}")

    if preds.shape[1] > n_classes:
        preds = preds[:, :n_classes]
    elif preds.shape[1] < n_classes:
        raise ValueError(
            f"Multiclass predictions have fewer columns ({preds.shape[1]}) than classes ({n_classes})"
        )

    row_sums = preds.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=1e-20):
        if (preds >= 0).all() and (row_sums > 0).all():
            preds = preds / row_sums
        else:
            preds = _softmax_rows(preds)

    return preds


def resolve_artifact_paths(experiment_dir: Path) -> list[Path]:
    candidates = [
        experiment_dir / 'artifacts.npz',
        experiment_dir / 'phase1' / 'artifacts.npz',
        experiment_dir / 'phase2' / 'artifacts.npz',
    ]
    paths = [path for path in candidates if path.exists()]
    if not paths:
        raise FileNotFoundError(f"No artifacts.npz found in {experiment_dir}")
    return paths
