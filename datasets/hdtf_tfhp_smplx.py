import math
import os
import random
from typing import List

from natsort import natsorted
from data.prepare import DataPreProcess, calculate_smplx_dict_stats
from datasets import MULTI_DATASET
from data.utils import get_video_duration_cv2_fast


class HDTF_TFHP_SMPLX(MULTI_DATASET):
    def __init__(self):
        super().__init__("HDTF_TFHP_SMPLX")
        self.processors = [
            DataPreProcess(
                video_dir="../HDTF/data1",
                audio_dir="../HDTF/audio1",
                output_dir="../HDTF/output1",
                dataset_name=self.dataset_name,
            ),
            DataPreProcess(
                video_dir="../TFHP/data",
                audio_dir="../TFHP/audio",
                output_dir="../TFHP/output",
                dataset_name=self.dataset_name,
            ),
        ]

    def write_all_keys(self):
        res = []
        for p in self.processors:
            for name in p.video_names:
                if not p.check_done(name):
                    continue
                video_path = p.get_video_path(name)
                duration = round(get_video_duration_cv2_fast(video_path), 2)
                if duration <= 8:
                    continue
                print(f"{video_path}, {duration}")
                res.append(name)
        self.write_keys_file("keys", res)

    def slice_data(self):
        for p in self.processors:
            p.slice_data()

    def split(self, n=64):
        ids = self.read_ids_file("keys")
        print(len(ids))
        random.seed(0)
        random.shuffle(ids)
        res = {
            "train": set(ids[: -n * 2]),
            "val": set(ids[-n * 2 : -n]),
            "test": set(ids[-n:]),
        }
        to_write = {"train": [], "val": [], "test": []}
        for video_name in self.read_keys_file("keys"):
            id = video_name.split("/")[0]
            for k, v in res.items():
                if id in v:
                    to_write[k].append(video_name)
                    break

        for k, v in to_write.items():
            self.write_keys_file(k, sorted(v))

    def calc_stats(self):
        names_train = set(self.read_keys_file("train"))
        smplx_dict_paths = []
        for p in self.processors:
            for name in p.video_names:
                if name not in names_train or not p.check_done(name):
                    continue
                smplx_dict_paths.append(p.get_smplx_dict_path(name))
        save_path = os.path.join(self.dataset_dir, "stats_train.npz")
        calculate_smplx_dict_stats(smplx_dict_paths, save_path)
