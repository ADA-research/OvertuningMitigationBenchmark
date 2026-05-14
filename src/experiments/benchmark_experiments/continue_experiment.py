import os

DEFAULT_N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))


def _configure_thread_limits(threads_per_job: int):
    threads_per_job = max(1, int(threads_per_job))
    threads_as_str = str(threads_per_job)

    os.environ["OMP_NUM_THREADS"] = threads_as_str
    os.environ["OPENBLAS_NUM_THREADS"] = threads_as_str
    os.environ["MKL_NUM_THREADS"] = threads_as_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads_as_str
    os.environ["NUMEXPR_NUM_THREADS"] = threads_as_str
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch

    torch.set_num_threads(threads_per_job)
    torch.set_num_interop_threads(threads_per_job)

import warnings
import argparse

from src.utils.google_cloud_storage import GCSManager
from src.experiments.experiment import Experiment

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    """Goal is to continue a benchmark experiment, by finding existing results in GCS and only running the missing tasks."""

    parser = argparse.ArgumentParser(description="Continue an experiment from GCS")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment to continue")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID of benchmark to continue")
    parser.add_argument("--model_string", type=str, required=True, help="Model string of benchmark to continue")
    parser.add_argument("--large", action="store_true", help="Large or small dataset")
    parser.add_argument("--problem_type", type=str, required=True, help="Problem type of benchmark to continue")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory where the results folder is located")
    parser.add_argument("--n_jobs", type=int, default=DEFAULT_N_JOBS, help="Number of parallel Python jobs")
    parser.add_argument("--threads_per_job", type=int, default=1, help="Threads/CPUs allocated to each Python job")

    args=parser.parse_args()

    if args.n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")

    if args.threads_per_job < 1:
        raise ValueError("threads_per_job must be >= 1")

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    requested_cpus = args.n_jobs * args.threads_per_job
    if requested_cpus > slurm_cpus:
        raise ValueError(
            f"Requested n_jobs * threads_per_job = {requested_cpus}, "
            f"but only {slurm_cpus} CPUs are available via SLURM_CPUS_PER_TASK."
        )

    _configure_thread_limits(args.threads_per_job)

    from src.experiments.benchmark_experiments import benchmark_lite

    original_experiment = benchmark_lite.benchmark_one_dataset_lite(
        args.dataset_id,
        model_string=args.model_string,
        large_dataset=args.large,
        problem_type=args.problem_type,
        output_dir=args.output_dir,
        model_threads=args.threads_per_job,
    )

    manager = GCSManager()
    existing_results = manager.list_experiment_results(args.experiment_name)

    # Filter tasks based on if history.csv is present in the correct results folder
    list_of_uncompleted_tasks = [
        task for task in original_experiment.tasks
        if not f"{args.experiment_name}/{task.result_path}/history.csv" in existing_results
    ]

    continue_experiment = Experiment(
        experiment_name=args.experiment_name,
        list_of_tasks=list_of_uncompleted_tasks,
        continue_experiment=True,
        output_dir=args.output_dir,
    )

    print(f"Continuing experiment '{args.experiment_name}' with {len(list_of_uncompleted_tasks)} uncompleted tasks out of {len(original_experiment.tasks)} total tasks.")

    continue_experiment.run(n_jobs=args.n_jobs)