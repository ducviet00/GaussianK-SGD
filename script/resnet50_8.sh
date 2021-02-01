#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=00:10:00
#$ -N resnet50_8
#$ -o ./logs/$JOB_ID.$JOB_NAME.log
#$ -j y
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299

source /etc/profile.d/modules.sh
conda deactivate
module load python/3.6/3.6.5 cuda/10.1/10.1.243 cudnn/7.6/7.6.5 nccl/2.5/2.5.6-1 openmpi/2.1.6 gcc/7.4.0
source ~/venv/pytorch+horovod/bin/activate
cd io_local_test/test_resnet/

NUM_NODES=${NHOSTS}
NUM_GPUS_PER_NODE=4
#NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})
EPOCH=1

LOG_DIR="./logs/G${NUM_PROCS}_E${EPOCH}"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}
cat $SGE_JOB_HOSTLIST > ${LOG_DIR}/$JOB_ID.$JOB_NAME.nodes.list

MPIOPTS="-np ${NUM_PROCS} --hostfile $SGE_JOB_HOSTLIST --oversubscribe -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include bond0" #-x NCCL_DEBUG=INFO"
mpirun ${MPIOPTS} python3 ./pytorch_imagenet_resnet50.py --train-dir /groups2/gaa50004/data/ILSVRC2012/pytorch/train --val-dir /groups2/gaa50004/data/ILSVRC2012/pytorch/val/ --log-dir ${LOG_DIR} --epochs ${EPOCH}

#mpirun -np 4 -map-by ppr:4:node -mca pml ob1 python3 ./pytorch_imagenet_resnet50.py --train-dir /groups2/gaa50004/data/ILSVRC2012/pytorch/train --val-dir /groups2/gaa50004/data/ILSVRC2012/pytorch/val/ --epochs 1
#python test_plot.py -f ./logs/accuracy_per_iter.log ./logs/accuracy_per_epoch.log