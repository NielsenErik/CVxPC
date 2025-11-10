#!/bin/bash
#PBS -l select=2:ncpus=10:mem=100gb
#PBS -l walltime=60:00:00
#PBS -q common_cpuQ
#PBS -N AEL_pcnet_ge
#PBS -M erik.nielsen@unitn.it
#PBS -m abe
#PBS -o logs/hpc/AEL/pcnet_ge/out.txt
#PBS -e logs/hpc/AEL/pcnet_ge/err.txt

cd ${PBS_O_WORKDIR}

module load python-3.10.14
module load cuda-11.3
source $PWD/.venv/bin/activate

echo "Running AEL/pcnet_ge on node $(hostname)"
python3 $PWD/src/main.py $PWD/src/config/hpc/CrackSeg9k/pcnet_ge.json $PBS_ARRAY_INDEX
