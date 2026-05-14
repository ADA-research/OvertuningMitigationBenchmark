"""
Shared pytest fixtures for tests_benchmark.
"""
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.task.task_config import (
    DefaultTaskConfig,
    InnerEvaluationConfig,
    OuterEvaluationConfig,
    SearchSpaceConfig,
    TaskConfig,
)


# ---------------------------------------------------------------------------
# Lightweight task factory helpers
# ---------------------------------------------------------------------------

def make_binary_task(iterations=3, bo_initial_random=2, optimizer="smac", debug=True, result_path=None):
    """Return a minimal binary classification TaskConfig."""
    task = DefaultTaskConfig()
    task.iterations = iterations
    task.bo_initial_random_iterations = bo_initial_random
    task.optimizer = optimizer
    task.debug = debug
    if result_path is not None:
        task.result_path = result_path
    return task


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_task():
    """Minimal binary task using DefaultTaskConfig (LGBM, adult, offline)."""
    return make_binary_task()


@pytest.fixture
def synthetic_binary_data():
    """Small synthetic binary classification dataset (200 rows, 5 features)."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.integers(0, 2, n), name="target")
    return X, y


@pytest.fixture
def synthetic_multiclass_data():
    """Small synthetic 3-class dataset."""
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.integers(0, 3, n), name="target")
    return X, y


@pytest.fixture
def synthetic_regression_data():
    """Small synthetic regression dataset."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.standard_normal(n), name="target")
    return X, y


@pytest.fixture(scope="session")
def adult_dataset():
    """Load adult dataset once per session."""
    from src.datasets.offline_dataloader import OfflineDataLoader
    loader = OfflineDataLoader()
    return loader.load(1590, problem_type="binary")


# ---------------------------------------------------------------------------
# SMAC output cleanup
# ---------------------------------------------------------------------------

@pytest.fixture
def unique_result_path():
    """Return a unique result_path string and clean up smac3_output/<path> after the test."""
    run_id = f"test_{uuid.uuid4().hex[:12]}"
    yield run_id
    smac_dir = Path("smac3_output") / run_id
    if smac_dir.exists():
        shutil.rmtree(smac_dir)


@pytest.fixture(autouse=True, scope="session")
def cleanup_test_smac_dirs():
    """After the full test session, remove any smac3_output/test_* directories."""
    yield
    smac_root = Path("smac3_output")
    if smac_root.exists():
        for d in smac_root.iterdir():
            if d.is_dir() and d.name.startswith("test_"):
                shutil.rmtree(d)
