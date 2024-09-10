import os
from typing import List
import numpy as np
import torch
from data.prepare import DataPreProcess


class MULTI_DATASET:
    def __init__(self, dataset_name) -> None:
        self.dataset_name = dataset_name
        self.dataset_dir = f"datasets/{self.dataset_name}/lmdb"
        self.processors: List[DataPreProcess] = []

    def get_names(self, keys=["train", "val", "test"]):
        for key in keys:
            with open(os.path.join(self.dataset_dir, f"{key}.txt")) as f:
                names = f.read().splitlines()
            yield key, names

    def get_ids(self, keys=["train", "val", "test"]):
        for key in keys:
            with open(os.path.join(self.dataset_dir, f"{key}.txt")) as f:
                names = f.read().splitlines()
            ids = set([name.split("/")[0] for name in names])
            yield key, sorted(list(ids))

    def read_keys_file(self, file_name):
        with open(os.path.join(self.dataset_dir, f"{file_name}.txt")) as f:
            keys = f.read().splitlines()
        return keys

    def read_ids_file(self, file_name):
        names = self.read_keys_file(file_name)
        ids = set([name.split("/")[0] for name in names])
        return sorted(list(ids))

    def write_keys_file(self, file_name, keys):
        with open(os.path.join(self.dataset_dir, f"{file_name}.txt"), "w") as f:
            for key in keys:
                f.write(key + "\n")
        print(f"Saved keys to {file_name}.txt, {len(keys)} keys")

    def print_train_stats(self):
        stats_file = os.path.join(self.dataset_dir, "stats_train.npz")
        with np.load(stats_file) as data:
            coef_dict = {k: torch.tensor(v) for k, v in data.items()}
        for key, value in coef_dict.items():
            print(f"Key: {key}, Shape: {value.shape}, Type: {value.dtype}")
