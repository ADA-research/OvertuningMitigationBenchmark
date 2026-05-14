# Plotting Pipeline (User Guide)

This project generates all benchmark figures and summary CSVs from trajectory CSV files.

## What It Does

Running the plotting command creates a timestamped output folder with:

- grouped aggregate plots (main paper + appendix style views),
- optional per-dataset plots,
- critical-difference / pairwise statistical plots,
- model summary tables (performance and overtuning over datasets).

Output location format:

- `src/visualizations/results/plotting_results_YYYYMMDD_HHMMSS/`

## Prerequisites

1. Python environment with project requirements installed.
2. Plotting input directory (default: `src/visualizations/data`) containing trajectory sources used by the pipeline:
	 - `default` (required for benchmark aggregate trajectories),
	 - plus the other standard sources expected by the script (`one_se`, `post_hoc_surrogate`, `post_hoc_ensemble`, `makarova`, `mlplan`).
3. Optional but recommended: `datasets.csv` in the data directory.
	 - Enables dataset naming and small/large dataset group splits.
	 - Without it, only `all_datasets` grouped outputs are generated.

## Basic Run

From the repository root:

```bash
python -m src.visualizations.plotting.run --prod
```

Useful path overrides:

```bash
python -m src.visualizations.plotting.run \
	--data-dir src/visualizations/data \
	--output-root src/visualizations/results \
	--prod
```

## Important Flags

- `--prod`
	- Hides debug overlays on panels (`n_datasets | n_runs`).
	- Recommended for publication/export quality plots.

- `--partial`
	- Faster, reduced run.
	- Generates only the core grouped suite for `all_datasets`.
	- Skips statistical testing plots, appendix plots, benchmark aggregate trajectories, and per-dataset generation.

- `--per-dataset`
	- Enables per-dataset outputs (slow).
	- If omitted, only grouped outputs are produced.

- `--improvement-threshold FLOAT`
	- Threshold used by CDF filtering for non-tunable runs.
	- Default: `0.001`.

## Quick Recipes

Full production run (grouped + appendix + benchmark targets):

```bash
python -m src.visualizations.plotting.run --prod
```

Fast iteration run:

```bash
python -m src.visualizations.plotting.run --partial
```

Generate only per-dataset plots in addition to the full grouped pipeline:

```bash
python -m src.visualizations.plotting.run --prod --per-dataset
```

Generate only specific artifacts:

- `--only-pairwise-matrix`
- `--only-cv-retrain-plots`
- `--only-model-score-cd`
- `--only-benchmark-aggregate-selected`
- `--only-appendix-combined-plots`
- `--only-appendix-optimizer-plots`
- `--only-appendix-cdf-plots`

Example:

```bash
python -m src.visualizations.plotting.run --only-cv-retrain-plots --prod
```

## Filtering Inputs

You can restrict plotting to subsets (repeat flags as needed):

- `--dataset-id`
- `--optimizer`
- `--problem-type`
- `--mitigation`
- `--inner-split`

Example:

```bash
python -m src.visualizations.plotting.run \
	--prod --partial
```

## Notes

- The command is idempotent with respect to source data: each run writes to a new timestamped output directory.
- If no rows remain after filtering, the run fails early with a clear error.
