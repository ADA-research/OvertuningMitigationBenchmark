# OvertuningBenchmark

Benchmark framework for evaluating hyperparameter optimization methods and overtuning mitigation strategies.

## Installation

Tested and working with **Python 3.11.3 on WSL** (Windows Subsystem for Linux).

### 1. Clone the repository

```bash
git clone URL (omitted for anonymity)
cd OvertuningBenchmark
```

### 2. Install system dependencies

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    swig \
    git-lfs \
    libeigen3-dev \
    libopenblas-dev \
    libffi-dev
```

### 3. Set up Python virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "autogluon.tabular[fastai]" --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt
pip install "gpytorch>=1.4.0"
pip install "pymoo>=0.6.0"
pip install "HEBO==0.3.5" --no-deps
pip install IPython
```

## Repository Structure

```
OvertuningBenchmark/
├── main.py
├── README.md
├── requirements.txt
├── docs/
│   ├── datasets.md
│   ├── normalization.md
│   ├── parallelization.md
│   └── preprocessing.md
├── src/
│   ├── datasets/
│   ├── evaluators/
│   ├── experiments/
│   ├── history/
│   ├── metrics/
│   ├── mitigations/
│   ├── models/
│   ├── optimizers/
│   ├── resamplers/
│   ├── search_space/
│   ├── target/
│   ├── utils/
│   └── visualizations/
└── tests/
```


### Data assumptions

- Minimal local profile: `adult.csv`, `ilpd.csv`, and `363700.csv` in `src/datasets/datasets/`
- Full profile: all benchmark dataset CSVs pulled via Git LFS

The current test suite supports the minimal local profile and skips/tests-gates full-dataset-only checks.

### 0. Activate environment

```bash
source venv/bin/activate
```

### 1. Run tests

Run all tests:

```bash
python -m pytest -q
```

### 2. Run benchmark experiments

Example: run LGBM on dataset `363700` (binary).

```bash
python -m src.experiments.benchmark_experiments.start_experiment \
    --dataset_id 363700 \
    --model_string LGBM \
    --problem_type binary \
    --output_dir . \
    --n_jobs 1 \
    --threads_per_job 1
```

This creates a timestamped experiment folder under `results`, for example:

```text
results/LGBM_363700_YYYYMMDD_HHMMSS/
```
Parallelism guidance: see [docs/parallelization.md](docs/parallelization.md).

Since the experiments take long, we have added a sample of results for one dataset/model combination to the repository. Using those results, you should be able to run the preprocessing and visualization pipeline. 


### 3. Run per-experiment post-hoc preprocessing

You can run both post-hoc methods for each experiment directory you want to include. The results for this are already included in the repo, so you do not need to rerun these to obtain some visualizations. If you actually run a full experiment using the command above, you can apply post hoc ensembling or post hoc surrogate incumbent selection using the commands below. Note that by default, results get stored in the /results folder, while the visualziation works on the results_prd folder. This is to preserve integrety of the /results_prd folder, so if you run this yourself, make sure to move the full experiment to the results_prd folder.

Single experiment:

```bash
python -m src.experiments.benchmark_experiments.post_hoc_ensemble_experiment \
    --results_dir results/LGBM_363700 \
    --ensemble_size 40 \
    --n_jobs 1

python -m src.experiments.benchmark_experiments.post_hoc_surrogate_experiment \
    --results_dir results/LGBM_363700 \
    --n_jobs 1
```

These create per-experiment files such as:

- `trajectories_post_hoc_ensemble.csv`
- `results_post_hoc_ensemble.csv`
- `trajectories_post_hoc_surrogate.csv`
- `results_post_hoc_surrogate.csv`

### 4. Combine preprocessing outputs for plotting

The preprocessing pipeline scans all folders directly inside `results_prd/` and writes combined CSVs to `src/visualizations/data`.

```bash
python -m src.visualizations.preprocessing.pipeline
```

Notes:
Please do not use --recalculate as it collects raw results, which are not present because of their large size. Instead, just run the command above to create combined files. 
- `--recalculate` recomputes per-experiment preprocessing from raw artifacts.
- Without `--recalculate`, existing preprocessed per-experiment files are reused.
- Post-hoc files are aggregated automatically when present.


### 5. Generate plots

Generate core plots:

```bash
python -m src.visualizations.plotting.run \
    --data-dir src/visualizations/data \
    --output-root src/visualizations/results
    --partial
```
Please use the --partial flag to omit certain graphs that aggregate over multiple datasets or models, since in the partial results we only have one combination.

Outputs are written to a timestamped folder:

```text
src/visualizations/results/plotting_results_YYYYMMDD_HHMMSS/
```

### 5. Notes on full data
This script only showcases the functionality to generate plots from raw results. Since our complete raw results exceed 500GB, and the preprocessed results exceed 10GB, we are working on a way to compress and share this in a cost-efficient way, but we are not able to do that fully and anonymously at this moment, which is why we provide a sample of results. 