from data import LmdbWriter
import os
import torch
import numpy as np
from data.utils import smplx_dict_to_coef, walk_dir_sorted


class DataPreProcess:
    def __init__(self, video_dir, audio_dir, output_dir, dataset_name, map_size_GB=10) -> None:
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.dataset_dir = f"datasets/{dataset_name}/lmdb"
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.map_size_GB = map_size_GB
        self._writer = None
        self._video_names = None

    @property
    def video_names(self):
        if self._video_names is None:
            self._video_names = [sub_path[:-4] for sub_path in walk_dir_sorted(self.video_dir, suffix=".mp4")]
        return self._video_names

    @property
    def video_paths(self):
        return [os.path.join(self.video_dir, name) + ".mp4" for name in self.video_names]

    @property
    def audio_paths(self):
        return [os.path.join(self.audio_dir, name) + ".flac" for name in self.video_names]

    @property
    def smplx_dict_paths(self):
        return [os.path.join(self.output_dir, name, "body_track/smplx_track.pth") for name in self.video_names if self.check_done(name)]

    def get_smplx_dict_path(self, name):
        return os.path.join(self.output_dir, name, "body_track/smplx_track.pth")

    @property
    def writer(self):
        if self._writer is None:
            self._writer = LmdbWriter(self.dataset_dir, map_size_GB=self.map_size_GB)
        return self._writer

    def get_video_path(self, name):
        return os.path.join(self.video_dir, name) + ".mp4"

    def check_done(self, name):
        return os.path.exists(os.path.join(self.output_dir, name, "task.done"))

    def slice_data(self):
        self.writer.write_metadata()
        for name in self.video_names:
            if not self.check_done(name):
                continue
            audio_path = os.path.join(self.audio_dir, name + ".flac")
            self.writer.write_pair_file(audio_path, os.path.join(self.output_dir, name, "body_track/smplx_track.pth"), name)

    def write_keys(self):
        keys_path = os.path.join(self.dataset_dir, "keys.txt")
        keys = self.read_keys(keys_path)
        with open(keys_path, "a") as f:
            for name in self.video_names:
                if self.check_done(name) and name not in keys:
                    print(name)
                    f.write(name + "\n")

    def read_keys(self, keys_path):
        with open(keys_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def extract_audio(self):
        for name in self.video_names:
            video_path = os.path.join(self.video_dir, name) + ".mp4"
            audio_path = os.path.join(self.audio_dir, name) + ".flac"
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            os.system(f"ffmpeg -loglevel error -i {video_path} -vn -ar 16000 -ac 1 -sample_fmt s16 -c:a flac {audio_path}")
            print(f"{video_path} -> {audio_path}")

    def calculate_train_stats(self, file="train.txt"):
        smplx_dict_paths = []
        names = self.read_keys(os.path.join(self.dataset_dir, file))
        for name in names:
            if self.check_done(name):
                smplx_dict_paths.append(os.path.join(self.output_dir, name, "body_track/smplx_track.pth"))
        save_path = os.path.join(self.dataset_dir, "stats_train.npz")
        calculate_smplx_dict_stats(smplx_dict_paths, save_path)


def calculate_smplx_dict_stats(smplx_dict_paths, save_path):
    assert save_path.endswith(".npz"), "Save path must end with .npz"
    sums = {"exp": torch.zeros((50)).cuda(), "shape": torch.zeros((100)).cuda(), "pose": torch.zeros((6)).cuda()}
    squared_sums = {"exp": torch.zeros((50)).cuda(), "shape": torch.zeros((100)).cuda(), "pose": torch.zeros((6)).cuda()}
    counts = {"exp": 0, "shape": 0, "pose": 0}

    maxs = {
        "exp": torch.full((50,), -float("inf")).cuda(),
        "shape": torch.full((100,), -float("inf")).cuda(),
        "pose": torch.full((6,), -float("inf")).cuda(),
    }
    mins = {
        "exp": torch.full((50,), float("inf")).cuda(),
        "shape": torch.full((100,), float("inf")).cuda(),
        "pose": torch.full((6,), float("inf")).cuda(),
    }

    for smplx_dict_path in smplx_dict_paths:
        smplx_dict = torch.load(smplx_dict_path)
        coef = smplx_dict_to_coef(smplx_dict)

        for key in sums.keys():
            val = coef[key].cuda()
            sums[key] += val.sum(dim=0)
            squared_sums[key] += (val**2).sum(dim=0)
            counts[key] += val.shape[0]

            maxs[key] = torch.max(maxs[key], val.max(dim=0)[0])
            mins[key] = torch.min(mins[key], val.min(dim=0)[0])
        print(smplx_dict_path)

    means = {key: sums[key] / counts[key] for key in sums}
    variances = {key: (squared_sums[key] / counts[key]) - (means[key] ** 2) for key in sums}
    stds = {key: torch.sqrt(variances[key]) for key in sums}

    for key in sums:
        print(f"{key} \n mean: {means[key]}, \n std: {stds[key]}, \n max: {maxs[key]}, \n min: {mins[key]}")

    np.savez(
        save_path,
        shape_mean=means["shape"].cpu().numpy(),
        shape_std=stds["shape"].cpu().numpy(),
        shape_max=maxs["shape"].cpu().numpy(),
        shape_min=mins["shape"].cpu().numpy(),
        exp_mean=means["exp"].cpu().numpy(),
        exp_std=stds["exp"].cpu().numpy(),
        exp_max=maxs["exp"].cpu().numpy(),
        exp_min=mins["exp"].cpu().numpy(),
        pose_mean=means["pose"].cpu().numpy(),
        pose_std=stds["pose"].cpu().numpy(),
        pose_max=maxs["pose"].cpu().numpy(),
        pose_min=mins["pose"].cpu().numpy(),
    )
    print(f"Stats saved to {save_path}")
