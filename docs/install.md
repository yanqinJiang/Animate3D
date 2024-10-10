# Installation

Make sure you have an NVIDIA graphics card with at least 24GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed. The code has been tested on Ubuntu 20.04 with CUDA 12.2. (Important: This code has not been tested with PyTorch versions 2.0.0 or higher. For optimal compatibility and reliability, please use PyTorch versions prior to 2.0.0.)

```bash
# install animate3d
git clone https://github.com/yanqinJiang/Animate3D.git
cd Animate3D
conda create -n animate3d python=3.10
conda activate animate3d
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

# install threestudio-3dgs plugin
cd custom
git clone https://github.com/DSaurus/threestudio-3dgs.git
cd threestudio-3dgs
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
git clone https://github.com/DSaurus/simple-knn.git
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
```