# Normalization Guide

Normalization maps losses to a shared $[0, 1]$ scale per group:

- $0$ = best observed loss
- $1$ = worst observed loss

It is applied after all experiment outputs are combined.

## Source of the ranges

Ranges are built per group from all observed raw scores, not just incumbent trajectories.

```
[dataset_id, repetition, outer_fold, metric, model]
```

The main anchor is `combined/preprocessed_results.csv`. If present, the ranges are extended with supplementary post hoc result files, including `preprocessed_results_post_hoc_ensemble.csv`. So post hoc ensemble results also contribute to the min/max ranges.

## Formula

$$
\\text{normalized} = \frac{\\text{value} - \\text{group min}}{\\text{group max} - \\text{group min}}
$$

If the denominator is at most $1e-14$, the normalized value is `NaN`.

Overtuning uses the same denominator as the corresponding test-loss column:

$$
\\text{normalized retrain overtuning} = \frac{\\text{retrain overtuning}}{\\text{ret max} - \\text{ret min}}
$$

$$
\\text{normalized ensemble overtuning} = \frac{\\text{ensemble overtuning}}{\\text{ens max} - \\text{ens min}}
$$

## Example

For dataset `363612`, repetition `0`, fold `0`, metric `roc_auc`, model `LGBM`:

| Iteration | Raw Score | Normalized |
|-----------|-----------|------------|
| Config 0 | 0.80 (min) | 0.0 |
| Config 1 | 0.85 | 0.5 |
| Config 2 | 0.90 (max) | 1.0 |

```
range = 0.90 - 0.80 = 0.10

normalized_Config0 = (0.80 - 0.80) / 0.10 = 0.0
normalized_Config1 = (0.85 - 0.80) / 0.10 = 0.5
normalized_Config2 = (0.90 - 0.80) / 0.10 = 1.0
```

## Output columns

- `normalized_val_loss`
- `normalized_retrain_test_loss`
- `normalized_ensembled_test_loss`
- `normalized_retrain_overtuning`
- `normalized_ensemble_overtuning`

