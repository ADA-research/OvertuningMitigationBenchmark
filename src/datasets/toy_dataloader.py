import pandas as pd
import numpy as np

from sklearn.datasets import make_classification, make_regression

from src.datasets.base_dataloader import BaseDataLoader


# Create an extremely basic binary dataset
def create_classification_dataset(seed, num_train_samples=2000, num_classes=2):
    X, y = make_classification(
        n_samples=50000,  # Number of samples
        n_features=25,  # Number of features
        n_informative=10,  # Number of informative features
        n_classes=num_classes,  # Binary classification
        random_state=seed,  # For reproducibility
        flip_y=0.1,
    )
    # Convert X to a pandas DataFrame with column names for compatibility
    column_names = [f"f{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=column_names)
    X.values.ravel()[
        np.random.default_rng(seed).choice(X.size, int(len(X) * len(X.columns) * 0.2), replace=False)] = np.nan

    # Convert y to a pandas Series
    y = pd.Series(y, name="target")

    return X, y, None


# Create a regression dataset
def create_regression_dataset(seed, num_train_samples=2000):
    X, y = make_regression(
        n_samples=50000,  # Number of samples
        n_features=25,  # Number of features
        n_informative=10,  # Number of informative features
        noise=10.0,  # Standard deviation of gaussian noise
        random_state=seed,  # For reproducibility
    )
    # Convert X to a pandas DataFrame with column names for compatibility
    column_names = [f"f{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=column_names)
    X.values.ravel()[
        np.random.default_rng(seed).choice(X.size, int(len(X) * len(X.columns) * 0.2), replace=False)] = np.nan

    # Convert y to a pandas Series
    y = pd.Series(y, name="target")

    return X, y, None


class ToyDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()

    def load(self, dataset_type: str, problem_type=None, num_train_samples: int = 2000):
        if dataset_type == "binary":
            return create_classification_dataset(
                seed=1,
                num_train_samples=num_train_samples,
                num_classes=2
            )

        elif dataset_type == "multiclass":
            return create_classification_dataset(
                seed=1,
                num_train_samples=num_train_samples,
                num_classes=10
            )

        else:
            return create_regression_dataset(
                seed=1,
                num_train_samples=num_train_samples
            )
