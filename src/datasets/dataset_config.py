import pandas as pd

# Dataset information mapping for TabArena datasets
# Key: dataset_id, Value: (n_samples, n_features, problem_type)
# Key corresponds to OpenML task ID 
DATASET_INFO = {
    363698: (907, 6, "regression"),
    363613: (32769, 9, "binary"),
    363614: (898, 38, "multiclass"),
    363625: (1030, 8, "regression"),
    363616: (76000, 170, "binary"),
    363618: (45211, 13, "binary"),
    363619: (10000, 10, "binary"),
    363620: (3751, 1776, "binary"),
    363621: (748, 4, "binary"),
    363623: (5000, 19, "binary"),
    363624: (9822, 85, "binary"),
    363675: (1338, 6, "regression"),
    363626: (1000, 20, "binary"),
    363627: (30000, 23, "binary"),
    363628: (129880, 21, "binary"),
    363629: (768, 8, "binary"),
    363630: (71518, 47, "binary"),
    363612: (1503, 5, "regression"),
    363632: (10999, 10, "binary"),
    363671: (1500, 6, "binary"),
    363615: (1538, 7, "regression"),
    363673: (150000, 10, "binary"),
    363674: (2400, 30, "binary"),
    363697: (5742, 1024, "regression"),
    363676: (10459, 23, "binary"),
    363685: (1014, 6, "multiclass"),
    363708: (6497, 12, "regression"),
    363679: (19158, 12, "binary"),
    363681: (12684, 24, "binary"),
    363682: (1723, 13, "binary"),
    363683: (50000, 212, "binary"),
    363684: (2240, 25, "binary"),
    363707: (1353, 9, "multiclass"),
    363686: (13776, 15, "regression"),
    363689: (7491, 86, "binary"),
    363691: (12330, 17, "binary"),
    363678: (20640, 8, "regression"),
    363694: (5910, 64, "binary"),
    363696: (1054, 41, "binary"),
    363705: (21263, 81, "regression"),
    363672: (45451, 9, "regression"),
    363711: (1699, 111, "multiclass"),
    363700: (2584, 15, "binary"),
    363702: (3190, 60, "multiclass"),
    363677: (3845, 1617, "multiclass"),
    363693: (45730, 9, "regression"),
    363706: (6819, 94, "binary"),
    363704: (4424, 36, "multiclass"),
    363631: (53940, 9, "regression"),
    363699: (78053, 11, "multiclass"),
    363712: (10885, 21, "binary"),
}


class DatasetConfig:
    def __init__(self, dataset_id: int):
        """
        Initialize the dataset configuration from TabArena.

        Args:
            dataset_id (int): The ID of the dataset.

        """
        # Convert to int to match keys
        if isinstance(dataset_id, str):
            dataset_id = int(dataset_id)

        dataset_info = DATASET_INFO.get(dataset_id)

        self.n_samples = dataset_info[0]
        self.problem_type = dataset_info[2]
        self.large = self.n_samples > 2500

    def is_large(self) -> bool:
        return self.large
    

    def get_problem_type(self) -> str:
        return self.problem_type
    