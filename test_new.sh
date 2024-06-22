#!/bin/bash -l
# Use the current working directory, which is the default setting. #SBATCH -D ./
# Use the current environment for this job, which is the default setting. #SBATCH --export=ALL

#SBATCH -p gpucs
#SBATCH --gres=gpu:1
#SBATCH -N 1
# Request the number of cores
#SBATCH -n 12
##SBATCH --qos=dedicated

python3 main_reddit.py \
  --epochs 20 \
  --data_dir /users/psen/data/ \
  --batch_size 100 \
  --ckpt_path /users/psen/VariationalInformationPursuit/saved/211rbk59/ckpt/epoch19.ckpt \
  --max_queries 10 \
  --max_queries_test 10 \
  --lr 0.0001 \
  --tau_start 1.0 \
  --tau_end 0.2 \
  --sampling biased \
  --seed 0 \
  --name news_bias_latest
