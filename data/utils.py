import subprocess
import cv2
import torch
import os
import numpy as np
from natsort import natsorted

# def dict_to_tensor(smplx_dict, device=torch.device("cpu")):


def smplx_dict_to_coef(smplx_dict, device=torch.device("cpu")):
    n_frames = smplx_dict["expr"].shape[0]
    exp_param = torch.tensor(smplx_dict["expr"], device=device)  # (n,50)
    shape_param = torch.tensor(smplx_dict["shape"], device=device).expand(n_frames, -1)  # (1,100)->(n,100)
    neck_pose = torch.tensor(smplx_dict["body_pose"][:, 33:36], device=device)  # (n,63)->(n,3)
    jaw_pose = torch.tensor(smplx_dict["jaw_pose"], device=device)  # (n,3)
    head_pose = torch.cat([neck_pose, jaw_pose], dim=1)  # (n,6)
    coef = {"shape": shape_param, "exp": exp_param, "pose": head_pose}
    return coef


def smplx_dict_to_tensor(smplx_dict, dtype=torch.float32, device=torch.device("cpu")):
    return {key: torch.tensor(value, dtype=dtype, device=device) for key, value in smplx_dict.items()}


def coef_to_smplx_dict(coef_dict):
    n_frames = coef_dict["exp"].shape[0]
    body_pose = torch.zeros((n_frames, 63))
    body_pose[:, 33:36] = coef_dict["pose"][:, :3]
    smplx_dict = {
        "expr": coef_dict["exp"],
        "shape": coef_dict["shape"],
        "body_pose": body_pose,
        "jaw_pose": coef_dict["pose"][:, 3:],
    }
    return smplx_dict


def save_smplx_dict_to_style_coef(smplx_dict_path, key="style1"):
    smplx_dict = torch.load(smplx_dict_path)
    coef = smplx_dict_to_coef(smplx_dict)
    coef_numpy = {key: value.cpu().numpy() for key, value in coef.items()}
    style_dir = "datasets/style"
    os.makedirs(style_dir, exist_ok=True)
    save_path = f"{style_dir}/{key}.npz"
    np.savez(save_path, **coef_numpy)
    print(f"Save coef to {save_path}")


def get_video_duration(video_path):
    """获取视频时长（秒）"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)

def get_video_duration_cv2_fast(filename):
    video = cv2.VideoCapture(filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    assert fps > 0, f"fps: {fps}"
    return frame_count / fps

def walk_dir_sorted(input_dir, suffix=""):
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = natsorted(dirs)
        for file in natsorted(files):
            if not str(file).endswith(suffix):
                continue
            sub_path = os.path.normpath(os.path.join(os.path.relpath(root, input_dir), file))
            yield sub_path


def walk_dir_mp4(input_dir):
    for name in walk_dir_sorted(input_dir, suffix=".mp4"):
        yield name
