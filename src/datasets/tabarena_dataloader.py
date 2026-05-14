import openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.datasets.base_dataloader import BaseDataLoader


class TabarenaDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()
        self.tasks = [
            openml.tasks.get_task(task) for task in openml.study.get_suite("tabarena-v0.1").tasks
        ]

    def load(self, dataset_id: int, problem_type=None):
        task = openml.tasks.get_task(dataset_id)
        dataset = task.get_dataset()

        X, y, _, _ = dataset.get_data(
            target=task.target_name, dataset_format="dataframe"
        )

        # Transform object columns to categorical columns
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category")

        # Ensure classification targets are numeric
        if task.task_type == "Supervised Classification":

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            y = pd.Series(y, name=y.name if hasattr(y, 'name') else 'target')

            label_map = {i: label for i, label in enumerate(label_encoder.classes_)}

            print(f"Dataset {dataset_id} with size {X.shape[0], X.shape[1]} loaded.")
            return X, y, label_map

        return X, y, None

    def get_all_binary(self):
        return [task.task_id for task in self.tasks if
                task.task_type == "Supervised Classification" and len(task.class_labels) == 2]

    def get_all_multiclass(self):
        return [task.task_id for task in self.tasks if
                task.task_type == "Supervised Classification" and len(task.class_labels) > 2]

    def get_all_classification(self):
        return self.get_all_binary() + self.get_all_multiclass()

    def get_all_regression(self):
        return [task.task_id for task in self.tasks if task.task_type == "Supervised Regression"]

    def get_all_tasks(self):
        return [task.task_id for task in self.tasks]



if __name__ == "__main__":
    loader = TabarenaDataLoader()
    print(loader.get_all_multiclass())