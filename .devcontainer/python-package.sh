#!/bin/bash
rm -rf /home/user/.local/lib/python3.9/site-packages

# https://github.com/pman0214/docker_pytorch-jupyterlab/blob/master/files/install_torch.sh
cd $(dirname $0) || exit 1

CUDA_TARGETS="torch==1.9.0
torchvision==0.10.0"
TARGETS="torchaudio==0.9.0"

case $(arch) in
    aarch64|arm64)
        ;;
    x86_64|amd64)
        # build for cpu
        lines=$(echo "${CUDA_TARGETS}" | while read line; do
            echo $line+cpu
        done)
        CUDA_TARGETS=$lines
        ;;
    *)
        exit 1
esac
set -x

conda create -n py39 python=3.9 -y
source ~/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate py39 || exit 1
pip install ${CUDA_TARGETS} ${TARGETS} -f https://download.pytorch.org/whl/torch_stable.html || exit 1
conda install tqdm=4.64 matplotlib=3.5.3 scipy=1.9.3 pandas=1.5.1 numba=0.56.3 notebook=6.5.2 -y || exit 1