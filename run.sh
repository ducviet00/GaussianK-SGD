#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=05:00:00
#$ -N res50_64
#$ -o ./logs/$JOB_ID.$JOB_NAME.log
#$ -j y
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299

DNN="${dnn:-resnet50}"
lr="${lr:-0.1}"
batch_size="${batch_size:-64}"
nworkers="${nworkers:-4}"
density="${density:-0.01}"
compressor="${compressor:-topk}"
NUM_NODES=${NHOSTS}
dataset="cifar10"
nstepsupdate=1
NUM_GPUS_PER_NODE=4
#NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})
EPOCH=20

LOG_DIR="./logs/G${NUM_PROCS}_E${EPOCH}"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}
cat $SGE_JOB_HOSTLIST > ${LOG_DIR}/$JOB_ID.$JOB_NAME.nodes.list

MPIOPTS="-np ${NUM_PROCS} --hostfile $SGE_JOB_HOSTLIST --oversubscribe -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include bond0" #-x NCCL_DEBUG=INFO"
mpirun ${MPIOPTS} python3 dist_trainer.py --dnn $DNN --dataset $dataset --max-epochs $EPOCH --batch-size $batch_size --nworkers $NUM_PROCS --nsteps-update $nstepsupdate --nwpernode $NUM_GPUS_PER_NODE --density $density --compressor $compressor  


