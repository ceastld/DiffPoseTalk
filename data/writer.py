import lmdb
import numpy as np
import torch
import pickle
import io
import torchaudio
from pathlib import Path
import librosa
from .utils import smplx_dict_to_coef
import torch.nn.functional as F

class LmdbWriter:
    def __init__(self, lmdb_dir, coef_fps=25, n_motions=100, map_size_GB=None) -> None:
        self.coef_fps = coef_fps
        self.n_motions = n_motions
        self.audio_sr = 16000
        self.audio_unit = 16000.0 / self.coef_fps  # num of samples per frame
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        map_size = 1024 * 1024 * 1024
        if map_size_GB is not None:
            map_size *= int(map_size_GB)
        self.lmdb_env = lmdb.open(lmdb_dir, map_size=map_size)

    def slice_data(self, audio:torch.Tensor, coef):
        """Slice audio and coef data into segments of length clip_len."""
        clip_len = self.n_motions
        n_frames = coef["exp"].shape[0]
        n_clips = (n_frames + clip_len - 1) // clip_len  # Calculate the number of clips including the last partial one
        n_audio = round(n_frames * self.audio_unit)
        if len(audio) < n_audio: 
            # Pad audio if it is shorter than the expected length
            audio = F.pad(audio, (0, n_audio - len(audio)), mode="constant", value=0)
        audio_clips = []
        coef_clips = []

        for i in range(n_clips):
            start = i * clip_len
            end = min((i + 1) * clip_len, n_frames)

            # Slice audio and coef data
            audio_clips.append(audio[round(start * self.audio_unit) : round(end * self.audio_unit)])

            # Create a dictionary for this segment and append to the list
            coef_clip = {k: coef[k][start:end] for k in coef.keys()}
            coef_clips.append(coef_clip)

        return audio_clips, coef_clips

    def write_pair_file(self, audio_path, smplx_coef_path, entry_name):
        audio, sr = librosa.load(audio_path, sr=self.audio_sr)
        audio = torch.tensor(audio, dtype=torch.float32)
        assert sr == self.audio_sr, "Audio sample rate must be 16000 Hz"
        smplx_dict = torch.load(smplx_coef_path)
        coef = smplx_dict_to_coef(smplx_dict)
        self.write_pair(audio, coef, entry_name)
        print(f"{entry_name}: audio,{audio_path} coef,{smplx_coef_path}")

    def write_metadata(self, seg_len=100):
        with self.lmdb_env.begin(write=True) as txn:
            metadata = {"seg_len": seg_len}
            txn.put("metadata".encode(), pickle.dumps(metadata))

    def write_pair(self, audio, coef, entry_name):
        """Slice audio and coef data, then write the sliced data to LMDB."""

        # Slice the audio and coefficient data
        audio_clips, coef_clips = self.slice_data(audio, coef)

        with self.lmdb_env.begin(write=True) as txn:
            metadata = {"n_frames": coef["exp"].shape[0]}
            txn.put(f"{entry_name}/metadata".encode(), pickle.dumps(metadata))

            # Write each clip
            for i, (audio_clip, coef_clip) in enumerate(zip(audio_clips, coef_clips)):
                # Save audio clip to a BytesIO object in FLAC format
                buffer = io.BytesIO()
                torchaudio.save(buffer, audio_clip.unsqueeze(0), self.audio_sr, format="flac", encoding="PCM_S", bits_per_sample=16)
                audio_data = buffer.getvalue()

                # Prepare the entry with sliced audio and coef data
                entry = {"audio": audio_data, "coef": {k: np.array(v) for k, v in coef_clip.items()}}
                txn.put(f"{entry_name}/{i:03d}".encode(), pickle.dumps(entry))

    def inspect_lmdb(self, prefix="", limit=100):
        with self.lmdb_env.begin(write=False) as txn:
            # Get all keys
            cursor = txn.cursor()
            count = 0
            for key, value in cursor:
                if count >= limit:
                    break
                if not key.startswith(prefix.encode()):
                    continue
                print(f"Key: {key.decode('utf-8')}")
                try:
                    data = pickle.loads(value)
                    self.print_structure(data)
                except Exception as e:
                    print(f"Could not decode value for key {key.decode('utf-8')}: {e}")
                print("-" * 50)
                count += 1

    @staticmethod
    def print_structure(data, indent=0):
        """Recursively print the structure of a dictionary."""
        if isinstance(data, dict):
            for k, v in data.items():
                print(" " * indent + f"{k}: ", end="")
                LmdbWriter.print_structure(v, indent + 2)
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            print(f"Array with shape: {data.shape}")
        elif isinstance(data, bytes):
            print(f"Bytes data with length: {len(data)}")
        else:
            print(data)
