from data import LmdbWriter, DataPreProcess
from data.utils import *
import torch
import subprocess
import librosa
import lmdb
import pickle
from pathlib import Path

HDTF_TFHP = "datasets/HDTF_TFHP/lmdb"
TFHP_SMPLX = "datasets/TFHP_SMPLX/lmdb"
CUSTOM = "datasets/custom/lmdb"

def save_audio(lmdb_dir, out_audio="temp_audio_file1.flac"):
    lmdb_dir = Path(lmdb_dir)
    lmdb_env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False)

    with lmdb_env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                data = pickle.loads(value)
                if key.startswith(b"TH_00006"):
                    audio_data = data["audio"]
                    with open(out_audio, "wb") as f:
                        f.write(audio_data)
                    subprocess.run(["ffprobe", out_audio])
                    return  # 保存音频后立即退出
            except Exception as e:
                print(f"Could not decode value for key {key.decode('utf-8')}: {e}")

def inspeck_audio():
    save_audio(lmdb_dir="datasets/custom/lmdb", out_audio="custom_audio.flac")
    save_audio(lmdb_dir="datasets/HDTF_TFHP/lmdb", out_audio="hdtf_audio.flac")

def slice_test(writer: LmdbWriter, audio_file):
    # Example usage:
    audio, sr = librosa.load(audio_file, sr=16000)
    assert sr == 16000, "Audio sample rate must be 16000 Hz"
    audio = torch.tensor(audio, dtype=torch.float32)  # Example audio data: 40 seconds of audio at 16000 Hz
    coef = {
        "shape": torch.randn(1000, 100),  # Example shape data
        "exp": torch.randn(1000, 50),  # Example expression data
        "pose": torch.randn(1000, 6),  # Example pose data
    }
    writer.write_pair(audio, coef, "TH_00006")

def slice_test1():
    writer = LmdbWriter("datasets/custom/lmdb")
    # slice_test(writer, audio_file="datasets/custom/audio1.flac")
    writer.write_pair_file(
        audio_path="/home/juyonggroup/ldy/repos/vasa/SMPLXTracking/data/talk_v/audio.flac",
        smplx_coef_path="/home/juyonggroup/ldy/repos/vasa/SMPLXTracking/data/talk/body_track/smplx_track_ori.pth",
        entry_name="TALK_00001",
    )
    writer.inspect_lmdb()
    

def preprocess():
    processor = DataPreProcess(
        video_dir="../TFHP/data",
        audio_dir="../TFHP/audio",
        output_dir="../TFHP/output",
        dataset_name="TFHP_SMPLX",
    )
    # processor.calculate_stats(file='keys.txt')
    # processor.split_keys_check()
    processor.slice_data()

from data import LmdbDataset

def debug_data():
    items = LmdbDataset(TFHP_SMPLX, f"{TFHP_SMPLX}/train.txt", f"{TFHP_SMPLX}/stats_train.npz")
    for i,it in enumerate(items):
        print(i)