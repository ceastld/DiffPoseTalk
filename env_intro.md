# Environments

1. download FLAME2020
```bash
bash setup/flame.sh
```
2. setup conda env
```bash
conda create -n diffposetalk python=3.8
conda activate diffposetalk

pip install torch torchvision torchaudio # current cuda12.1

pip install git+https://github.com/MPI-IS/mesh.git

pip install -r requirements.txt
pip install numpy==1.23.1 # for numpy.bool
pip install smplx
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install "git+https://github.com/NVlabs/nvdiffrast.git"
pip install ninja

```

3. infer
pretrained models
download from [models](https://drive.google.com/drive/folders/1pOwtK95u8O1qG_CiRdD8YcvuKSlFEk-b?usp=sharing)
then unzip to ./experiments
```bash
# demo
CUDA_VISIBLE_DEVICES="4" python demo.py --exp_name SA-hubert-WM --iter 100000 \
    -a demo/input/audio/talk.mp3 -c demo/input/coef/TH050.npy \
    -s demo/input/style/smile.npy -o talk-head-smile-ss1.mp4 \
    -n 3 -ss 1 -sa 1.15 -dtr 0.99 \
    --save_coef
```

1. trainging
```bash
# style encoder
python main_se.py --exp_name <STYLE_ENC_NAME> --data_root <DATA_ROOT> [--no_head_pose]
CUDA_VISIBLE_DEVICES="4,5,6,7" python main_se.py --exp_name head-L4H4-T0.1-BS32-1 --data_root datasets/TFHP_SMPLX/lmdb

# denoising network
python main_dpt.py --exp_name <DENOISING_NETWORK_NAME> --data_root <DATA_ROOT> --use_indicator --scheduler Warmup --audio_model hubert --style_enc_ckpt <PATH_TO_STYLE_ENC_CKPT> [--no_head_pose]
CUDA_VISIBLE_DEVICES="4,5,6,7" python main_dpt.py --exp_name head-SA-hubert-WM --data_root datasets/TFHP_SMPLX/lmdb --use_indicator --scheduler Warmup --audio_model hubert --style_enc_ckpt experiments/SE/head-L4H4-T0.1-BS32-240824_204353/checkpoints/iter_0010000.pt

```

# 表情系数使用


TODO:
- [ ] smplx下载

