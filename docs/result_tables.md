# Released Result Tables

This document describes the two combined files produced by
`src/visualizations/preprocessing/pipeline.py` and distributed for analysis:

- `preprocessed_results.csv.gz` (fold-level/configuration results)
- `trajectories.csv.gz` (long-form trajectories, with a `method` column)

Both are UTF-8 CSV files compressed with gzip. They can be read directly with
Pandas using `pd.read_csv("...csv.gz")`; Pandas detects gzip from the filename.
The CSV format does not carry a formal type schema. Blank fields usually mean
that a value is not applicable or was unavailable. Treat identifier columns as
labels, even when their values look numeric.

The released summary files are sufficient for the published plotting pipeline
and most downstream analyses. The trajectory file is the plotting input. The
fold-level file preserves configuration-level results and is also the source
of the empirical ranges used to normalize trajectory scores. Neither summary
contains prediction arrays or replaces the raw artifacts needed to recompute
scores or rerun post-hoc procedures.

## Table 1: `preprocessed_results.csv.gz`

### Row unit and interpretation

One row represents one optimization iteration and one recorded fold within a
single HPO run. An HPO run is identified by `experiment_id`; several rows can
share an iteration when that iteration has multiple folds. `fold` is the
within-run fold index, while `outer_fold` and `repetition` identify the outer
evaluation split. The table contains the evaluated configurations, not only
the incumbent configurations retained in the trajectory table.

For a given iteration, some iteration-level values (notably
`cv_ensembled_test_score`) are repeated on each fold row. Do not count repeated
iteration-level values once per row when computing iteration aggregates.
`val_loss`, `retrain_test_loss`, and `ensembled_test_loss` are compatibility
aliases for their corresponding score columns, not additional measurements.

### Columns

| Column                      | Meaning                                                                                                                                                                                |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `iteration`               | Zero-based optimizer iteration/configuration index.                                                                                                                                    |
| `fold`                    | Fold index within the run's per-iteration evaluation; prediction artifacts use this fold index.                                                                                        |
| `train`                   | Training-split score recorded in the optimization history for this fold.                                                                                                               |
| `val`                     | Validation-split score recorded in the history for this fold.                                                                                                                          |
| `selection`               | Selection-split score, when a separate selection set is configured.                                                                                                                    |
| `test`                    | Test-split score recorded for this fold/configuration.                                                                                                                                 |
| `train_time`              | Training time recorded for the configuration/fold; seconds.                                                                                                                            |
| `val_inference_time`      | Validation prediction/inference time; seconds.                                                                                                                                         |
| `test_inference_time`     | Test prediction/inference time; seconds.                                                                                                                                               |
| `avg_val_score`           | Validation score aggregated across the run's folds for this iteration; used for ordinary incumbent selection. Lower is better.                                                         |
| `retrained_test_score`    | Test score for the individually retrained configuration, sourced from the history's `avg_test_score`; lower is better.                                                               |
| `avg_selection_score`     | Selection-set score aggregated for the iteration, when a selection set is configured; lower is better.                                                                                 |
| `avg_train_score`         | Training score aggregated for the iteration.                                                                                                                                           |
| `early_stopped_makarova`  | Boolean marker used by the Makarova stopping rule; missing when unavailable.                                                                                                           |
| `surrogate_mean`          | Surrogate-predicted score/mean recorded for the configuration; used by surrogate-based selection.                                                                                      |
| `surrogate_std`           | Surrogate predictive standard deviation, when available.                                                                                                                               |
| `surrogate_acq`           | Surrogate acquisition value recorded by the optimizer, when available.                                                                                                                 |
| `optimizer_suggest_time`  | Time spent producing the optimizer suggestion; seconds.                                                                                                                                |
| `mlplan_score`            | ML-Plan score used to select a final ML-Plan configuration; missing for other methods/configurations.                                                                                  |
| `cv_ensembled_test_score` | Test score after averaging fold-level test predictions for the same iteration and scoring the averaged predictions. Lower is better. It is repeated across that iteration's fold rows. |
| `dataset_id`              | Dataset identifier; use as a categorical key, not a numeric measurement.                                                                                                               |
| `repetition`              | Outer evaluation repetition index.                                                                                                                                                     |
| `outer_fold`              | Outer evaluation fold index.                                                                                                                                                           |
| `optimizer`               | HPO optimizer name, such as SMAC or HEBO.                                                                                                                                              |
| `metric`                  | Metric used for evaluation, such as `roc_auc` or a loss metric.                                                                                                                       |
| `problem_type`            | Task type, e.g. `binary`, `multiclass`, or `regression`.                                                                                                                          |
| `model`                   | Model/search-space estimator associated with the run.                                                                                                                                  |
| `mitigation`              | Mitigation strategy, such as `thresholdout`, `mlplan`, or a racing strategy; blank for the unmitigated baseline.                                                                    |
| `reshuffling`             | Whether the evaluation uses reshuffling; may be missing for procedures where it does not apply.                                                                                        |
| `inner_split`             | Inner evaluation split, commonly `cv` or `holdout`; missing for some mitigation procedures.                                                                                         |
| `selection_set_size`      | Configured selection-set size; zero or missing indicates no separate selection set.                                                                                                    |
| `random_state`            | Random seed recorded in the task configuration.                                                                                                                                        |
| `outer_n_folds`           | Number of outer folds configured for the evaluation.                                                                                                                                   |
| `outer_n_repeats`         | Number of outer repetitions configured for the evaluation.                                                                                                                             |
| `experiment_source`       | Name of the top-level experiment directory containing the run directory, relative to the results tree.                                                                                 |
| `raw_experiment_id`       | Original run-directory name from the result tree.                                                                                                                                      |
| `experiment_id`           | Namespaced run identifier (`experiment_source::raw_experiment_id`) intended to be unique across result roots.                                                                        |
| `val_loss`                | Compatibility alias of `avg_val_score`.                                                                                                                                               |
| `retrain_test_loss`       | Compatibility alias of `retrained_test_score`.                                                                                                                                        |
| `ensembled_test_loss`     | Compatibility alias of `cv_ensembled_test_score`.                                                                                                                                     |

### Score scale

Scores used for selection and trajectory calculations are represented on a
lower-is-better loss scale. In particular, the pipeline adds 1.0 to ROC-AUC
score fields stored in negated form before creating these summaries. This
conversion applies to the score columns used by preprocessing, not arbitrary
numeric columns. For multi-fold iterations,
`cv_ensembled_test_score` is calculated from averaged test predictions rather
than by averaging the fold scores. The task type controls prediction handling;
multiclass ensemble scoring uses negative log loss.

## Table 2: `trajectories.csv.gz`

### Row unit and interpretation

One row represents one HPO run, one iteration on its trajectory, and one
incumbent-selection method. A given run/iteration can therefore have several
rows, distinguished by `method`. The table is grouped by method in the file,
but consumers should use the `method` column rather than depend on physical row
order.

Methods currently written by the combiner are:

| `method`             | Meaning                                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `default`            | Standard incumbent selection, using validation scores unless a configured selection set supplies the selection scores.                             |
| `one_se`             | One-standard-error selection rule.                                                                                                                 |
| `makarova`           | Incumbent selection with the Makarova early-stopping rule.                                                                                         |
| `mlplan`             | ML-Plan's selected final configuration. ML-Plan has a single row (iteration 0), so its trajectory-only regret/overtuning quantities are undefined. |
| `post_hoc_ensemble`  | Post-hoc Caruana ensemble trajectory.                                                                                                              |
| `post_hoc_surrogate` | Post-hoc surrogate-based incumbent trajectory.                                                                                                     |

Rows for post-hoc methods are present only when their per-experiment
preprocessing outputs exist. The combined pipeline creates ML-Plan outputs when
needed; it does not itself run the post-hoc ensemble or surrogate procedures.
The separate `surrogate_selection` trajectory file is not a method in this
combined table.

### Columns

| Column                             | Meaning                                                                                                                                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `method`                         | Trajectory source label from the method list above.                                                                                                                                              |
| `iteration`                      | Optimizer iteration represented by the incumbent at that point. ML-Plan's single selected result is stored at iteration 0.                                                                       |
| `experiment_id`                  | Namespaced HPO run identifier.                                                                                                                                                                   |
| `experiment_source`              | Name of the top-level experiment directory containing the run directory, relative to the results tree.                                                                                           |
| `raw_experiment_id`              | Original run-directory name from the result tree.                                                                                                                                                |
| `dataset_id`                     | Dataset identifier.                                                                                                                                                                              |
| `model`                          | Model/search-space estimator associated with the run.                                                                                                                                            |
| `optimizer`                      | HPO optimizer name.                                                                                                                                                                              |
| `metric`                         | Evaluation metric name.                                                                                                                                                                          |
| `problem_type`                   | Task type, e.g. binary, multiclass, or regression.                                                                                                                                               |
| `mitigation`                     | Mitigation procedure associated with the run; blank for the unmitigated baseline.                                                                                                                |
| `reshuffling`                    | Whether evaluation uses reshuffling; may be missing when not applicable.                                                                                                                         |
| `inner_split`                    | Inner evaluation protocol, commonly `cv` or `holdout`; may be missing when not applicable.                                                                                                    |
| `selection_set_size`             | Configured selection-set size; zero or missing means no separate selection set.                                                                                                                  |
| `repetition`                     | Outer evaluation repetition index.                                                                                                                                                               |
| `outer_fold`                     | Outer evaluation fold index.                                                                                                                                                                     |
| `random_state`                   | Random seed recorded in the task configuration.                                                                                                                                                  |
| `outer_n_folds`                  | Number of outer folds configured for the evaluation.                                                                                                                                             |
| `outer_n_repeats`                | Number of outer repetitions configured for the evaluation.                                                                                                                                       |
| `val_performance`                | Validation loss of the method's selected incumbent at this iteration; lower is better.                                                                                                           |
| `retrain_test_loss`              | Test loss of that incumbent when individually retrained/evaluated; lower is better.                                                                                                              |
| `ensembled_test_loss`            | Test loss of that incumbent using the cross-validation ensemble of fold predictions; lower is better.                                                                                            |
| `retrain_meta_overfitting`       | `retrain_test_loss - val_performance`; test-minus-validation gap for the individual retrained model.                                                                                           |
| `ensemble_meta_overfitting`      | `ensembled_test_loss - val_performance`; test-minus-validation gap for the CV ensemble.                                                                                                        |
| `retrain_regret`                 | Incumbent retrained test loss minus the best retrained test loss among all configurations observed up to this iteration.                                                                         |
| `ensemble_regret`                | Incumbent ensemble test loss minus the best ensemble test loss among all configurations observed up to this iteration.                                                                           |
| `retrain_overtuning`             | Nonnegative increase in retrained test loss from the best test-performing incumbent seen so far to the current incumbent.                                                                        |
| `ensemble_overtuning`            | Nonnegative increase in ensemble test loss from the best test-performing incumbent seen so far to the current incumbent.                                                                         |
| `relative_retrain_overtuning`    | Retrained overtuning divided by the initial retrained test loss's gain over the best incumbent seen so far. If that gain is zero, the value is 0 when overtuning is zero and infinity otherwise. |
| `relative_ensemble_overtuning`   | Ensemble counterpart of `relative_retrain_overtuning`.                                                                                                                                          |
| `val_loss`                       | Compatibility alias of `val_performance`.                                                                                                                                                       |
| `normalized_val_loss`            | Validation loss mapped to the empirical range for its normalization group: `(val_performance - val_min) / (val_max - val_min)`.                                                                 |
| `normalized_retrain_test_loss`   | Retrained test loss mapped to the empirical retrained-test range: `(retrain_test_loss - ret_min) / (ret_max - ret_min)`.                                                                        |
| `normalized_ensembled_test_loss` | Ensemble test loss mapped to the empirical ensemble-test range: `(ensembled_test_loss - ens_min) / (ens_max - ens_min)`.                                                                        |
| `normalized_retrain_overtuning`  | `retrain_overtuning / (ret_max - ret_min)`. Overtuning uses the same range as retrained test loss; it is not independently min-max scaled.                                                     |
| `normalized_ensemble_overtuning` | `ensemble_overtuning / (ens_max - ens_min)`. Overtuning uses the same range as ensemble test loss; it is not independently min-max scaled.                                                     |

### Normalization and missing values

Normalization anchors are calculated separately for each
`dataset_id`, `repetition`, `outer_fold`, `metric`, and `model`. The anchors
use all available evaluated configurations in the fold-level results, across
optimizers, rather than only the selected incumbents. When available,
post-hoc ensemble result scores extend those empirical ranges. A denominator
at or below `1e-14` makes the corresponding normalized value undefined
(`NaN`). These are empirical scales, so normalized values can be outside
`[0, 1]` if a trajectory value lies outside the anchor range.

ML-Plan has no multi-iteration trajectory; its regret, overtuning, and
meta-overfitting fields are blank/NaN. Other missing values generally indicate
that a quantity does not apply or was not produced.
