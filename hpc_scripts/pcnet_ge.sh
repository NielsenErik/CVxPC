#!/bin/bash
#PBS -l select=8:ncpus=10:mem=20gb
#PBS -l walltime=6:00:00
#PBS -q short_cpuQ
#PBS -N crack
#PBS -M erik.nielsen@unitn.it
#PBS -m abe

# Go to working directory
cd ${PBS_O_WORKDIR}

# Load modules and environment
module load python-3.10.14
module load cuda-11.3   
source $PWD/.venv/bin/activate

# Extract cluster info
MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)          # first node = master
MASTER_PORT=29500                               # any free port is fine
NNODES=$(cat $PBS_NODEFILE | sort | uniq | wc -l)
NPROC_PER_NODE=10                               # match ncpus per node
NODE_RANK=$(grep -n $(hostname) $PBS_NODEFILE | head -n 1 | cut -d: -f1)
NODE_RANK=$((NODE_RANK - 1))

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "NODE_RANK: $NODE_RANK"

# Launch distributed training
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $PWD/src/main.py $PWD/src/config/hpc/pcnet_ge.json $PBS_ARRAY_INDEX
