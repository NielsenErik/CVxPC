#!/bin/bash

#PBS -l select=8:ncpus=10:mem=20gb

#set max execution time
#PBS -l walltime=6:00:00

#imposta la coda di esecuzione
#PBS -q short_cpuQ

#execution outpust name
#PBS -N crack

#set mail notification
#PBS -M erik.nielsen@unitn.it
#PBS -m abe


cd ${PBS_O_WORKDIR}

module load python-3.10.14
module load cuda-11.3   
source $PWD/.venv/bin/activate
python3 $PWD/src/main.py $PWD/src/config/hpc/pcnet.json $PBS_ARRAY_INDEX
