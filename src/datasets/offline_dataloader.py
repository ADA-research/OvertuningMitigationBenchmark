import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.datasets.base_dataloader import BaseDataLoader


class OfflineDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()


    def load(self, dataset_id: int, problem_type=None):
        if dataset_id == 1590:
            df = pd.read_csv(f'src/datasets/datasets/adult.csv')
        elif isinstance(dataset_id, int):
            df = pd.read_csv(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/{dataset_id}.csv')
        else:
            df = pd.read_csv(f'src/datasets/datasets/{dataset_id}.csv')

        y = df["target"]

        # Transform object columns to categorical columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("category")

        X = df.drop("target", axis=1)

        if problem_type in ["binary", "multiclass"] or (problem_type is None and len(y.unique()) < 8):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            y = pd.Series(y, name=y.name if hasattr(y, 'name') else 'target')

            label_map = {i: label for i, label in enumerate(label_encoder.classes_)}

            # print(f"Dataset {dataset_id} with size {X.shape[0], X.shape[1]} loaded.")
            print(dataset_id, X.shape, len(y.unique()))
            return X, y, label_map

        print(dataset_id, X.shape)
        # raise ValueError("Regression")
        return X, y, None


if __name__ == "__main__":
    task_ids = [int(x.split('.')[0]) for x in os.listdir(f'src/datasets/datasets/') if not x.startswith('adult') and not x.endswith('.zip')]
    loader = OfflineDataLoader()
    for task_id in task_ids:
        loader.load(task_id)

