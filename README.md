# OvertuningBenchmark

Benchmark framework for evaluating hyperparameter optimization methods and overtuning mitigation strategies.

## Installation

Tested and working with **Python 3.11.3 on WSL** (Windows Subsystem for Linux).

### 1. Clone the repository

```bash
git clone https://github.com/ADA-research/OvertuningMitigationBenchmark.git
cd OvertuningMitigationBenchmark
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
OvertuningMitigationBenchmark/
├── LICENSE
├── README.md
├── requirements.txt
├── docs/
│   ├── datasets.md
│   ├── normalization.md
│   ├── parallelization.md
│   ├── plotting.md
│   ├── reproducability.md
│   └── result_tables.md
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
│       ├── data/
│       ├── plotting/
│       └── preprocessing/
└── tests/
```

### Data access

Benchmark tasks are retrieved from OpenML. The machine running an experiment needs internet access, and `--dataset_id` must be an OpenML task ID. The benchmark infers the problem type from the dataset ID; the benchmark CLI does not accept a `--problem_type` option.

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

Example: run the full benchmark for LGBM on OpenML task `363700` (binary).

```bash
python -m src.experiments.benchmark_experiments.start_experiment \
    --dataset_id 363700 \
    --model_string LGBM \
    --output_dir . \
    --n_jobs 1 \
    --threads_per_job 1
```

This creates a timestamped experiment folder under `results`, for example:

```text
results/LGBM_363700_YYYYMMDD_HHMMSS/
```

The benchmark writes results locally and attempts to upload task results to Google Cloud Storage. Without the service-account file `google-cloud-phd-storage-key.json` in the repository root, uploads are disabled with a warning and the local run continues.

The post-hoc ensemble and post-hoc surrogate analyses are separate from the benchmark. After each benchmark finishes, run both commands for its experiment directory before combining results:

```bash
EXPERIMENT_DIR="results/LGBM_363700_YYYYMMDD_HHMMSS"

python -m src.experiments.benchmark_experiments.post_hoc_ensemble_experiment \
    --results_dir "$EXPERIMENT_DIR" \
    --ensemble_size 40 \
    --n_jobs 1

python -m src.experiments.benchmark_experiments.post_hoc_surrogate_experiment \
    --results_dir "$EXPERIMENT_DIR" \
    --n_jobs 1
```

Parallelism guidance: see [docs/parallelization.md](docs/parallelization.md).

To preprocess full results, point `--results-dir` at the parent directory containing your experiment result directories.

```bash
python -m src.visualizations.preprocessing.pipeline \
    --results-dir results \
    --output-dir src/visualizations/data
```

The output directory receives `preprocessed_results.csv.gz` and `trajectories.csv.gz`.

### 3. Released Result Tables and Plotting Pipeline

The visualization release consists of two gzip-compressed CSV files:

- `preprocessed_results.csv.gz`: fold-level/configuration results, used as a compact analysis table and as the normalization-anchor source.
- `trajectories.csv.gz`: iteration-by-iteration incumbent results for each included method; this is the input used by the plotting pipeline.

These results can be obtained by running the benchmark. Alternatively, we have released them for our full benchmark and they can be downloaded here:

Download trajectories.csv.gz

Download preprocessed_results.csv.gz

The detailed column definitions and row meanings are in [docs/result_tables.md](docs/result_tables.md). We do not publish the full results and artifacts, including labels and model predictions. Please contact us if you would like to use them. With these two files present in `src/visualizations/data`, you can run the plotting pipeline with:

```bash
python -m src.visualizations.plotting.run \
    --data-dir src/visualizations/data \
    --output-root src/visualizations/results \
    --prod
```

Plot files are written to a timestamped folder under `src/visualizations/results/`

## Citing us

If you use our benchmark's code or data in a scientific publication, please cite:

```bibtex
@inproceedings{schroder2026large,
    title={A Large-Scale Study of Overtuning Mitigation Strategies in Hyperparameter Optimization},
    author={Schr{\"o}der, Sietse and Zellmann, Zubin and Feurer, Matthias and Baratchi, Mitra and van Rijn, Jan N and Bischl, Bernd},
    booktitle={AutoML 2026 Methods Track}
}
```
