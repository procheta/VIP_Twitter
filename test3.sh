#!/bin/bash -l
# Use the current working directory, which is the default setting. #SBATCH -D ./
# Use the current environment for this job, which is the default setting. #SBATCH --export=ALL

#SBATCH -p gpucs
#SBATCH --gres=gpu:1
#SBATCH -N 1
# Request the number of cores
#SBATCH -n 12
##SBATCH --qos=dedicated

python3 a.py
