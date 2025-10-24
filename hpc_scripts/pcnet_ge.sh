#!/bin/bash
#PBS -l select=4:ncpus=10:mem=30gb
#PBS -l walltime=24:00:00
#PBS -q common_cpuQ
#PBS -N crack
#PBS -M erik.nielsen@unitn.it
#PBS -m abe

# Go to working directory
cd ${PBS_O_WORKDIR}

# Load modules and environment
module load python-3.10.14
module load cuda-11.3   
source $PWD/.venv/bin/activate

# Launch distributed training
python3  $PWD/src/main.py $PWD/src/config/hpc/pcnet_ge.json $PBS_ARRAY_INDEX
