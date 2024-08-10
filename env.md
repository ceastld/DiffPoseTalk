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

```

3. pretrained models
download from [models](https://drive.google.com/drive/folders/1pOwtK95u8O1qG_CiRdD8YcvuKSlFEk-b?usp=sharing)
then unzip to ./experiments


