#!/bin/bash
python3 -m venv $HOME/topk
source $HOME/topk/bin/activate
pip install --upgrade pip

pip install torch torchvision

HOROVOD_GPU_ALLREDUCE=NCCL \
HOROVOD_NCCL_HOME=$NCCL_HOME \
HOROVOD_WITH_PYTORCH=1 \
HOROVOD_WITH_MPI=1 \
pip install --no-cache-dir horovod[pytorch]
pip install mpi4py
pip install -r requirements.txt

pip list
