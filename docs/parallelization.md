# Parallelization in OvertuningBenchmark

Parallelism in this repo has two levels:

- `--n_jobs`: number of parallel Python worker processes
- `--threads_per_job`: number of threads available inside each worker for model fitting and numeric libraries

## How it works

`Experiment.run(n_jobs=...)` uses joblib with the `loky` backend, so parallelism is process-based at the task level. Each worker runs one task at a time.

In the benchmark launchers, `--threads_per_job` is used to configure thread limits for BLAS/OpenMP libraries and PyTorch, and it is also passed into the task configuration as `model_threads`. Model wrappers then use that value when constructing the inner models.

So the practical execution model is:

- outer parallelism = `n_jobs`
- inner model/library parallelism = `threads_per_job`

## CPU budget rule

The launcher enforces:

```
n_jobs * threads_per_job <= SLURM_CPUS_PER_TASK
```

This is the main safeguard against oversubscription.

## How to choose the arguments

- Increase `--n_jobs` to run more tasks concurrently.
- Increase `--threads_per_job` to give each task more CPU for model fitting.
- Usually you should trade one against the other, not increase both without increasing the CPU allocation.

## Main source files

- `src/experiments/experiment.py`
- `src/experiments/benchmark_experiments/start_experiment.py`
- `src/experiments/benchmark_experiments/start_tho_experiment.py`
