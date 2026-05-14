# Reproducibility

This project ensures full reproducibility across the entire optimization pipeline—from data splits to hyperparameter optimization and model training.

## Global Seed

All randomness is controlled via a single `random_state` value set in `TaskConfig`:

```python
task_config.random_state = 78
```

This single seed propagates down to every component that performs random operations.

## What's Made Reproducible

### 1. **Outer Train/Test Splits**
- `TaskDataSplitter` receives `random_state` and deterministically generates all outer CV folds
- Same config + same seed → identical train/test indices
- All 10 repeats × 3 folds produce distinct, disjoint splits

### 2. **Inner Resampling**
- `CVResampler` and `HoldoutResampler` receive the seed and generate deterministic fold assignments
- When `reshuffle=False`: same validation split across all HPO iterations
- When `reshuffle=True`: different splits per iteration, but deterministically reproducible from the seed

### 3. **Hyperparameter Optimization**
- `SearchSpace` seeds `ConfigSpace.ConfigurationSpace(seed=random_state)`
- All optimizers (`RandomSearch`, `SMAC`, `HEBO`) receive the seed and produce bit-identical results on repeated runs
- Preprocessor components (scalers, encoders, imputers) are individually seeded

### 4. **ML Algorithms**
- Scikit-learn models receive `random_state` during instantiation
- PyTorch models: `torch.manual_seed(random_state)` is set globally
- HEBO optimizer receives aggressive multi-framework seeding: `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, and `scramble_seed` parameter

### 5. **Synthetic Data Generation**
- `toy_dataloader.py` uses the same seed for generating reproducible toy datasets


## Verification

See [tests/test_reproducibility.py](../tests/test_reproducibility.py) for comprehensive reproducibility tests:
- Outer split determinism
- Inner resampler determinism
- HPO determinism across all optimizers
- End-to-end benchmark reproducibility

See the [README](../README.md) for instructions on how to run the test suite.

Running any task with the same `random_state=78` twice should produce bit-identical results.