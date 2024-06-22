#!/bin/bash -l
# Use the current working directory, which is the default setting. #SBATCH -D ./
# Use the current environment for this job, which is the default setting. #SBATCH --export=ALL

#SBATCH -p gpulowbig
#SBATCH --gres=gpu:1
#SBATCH -N 1
# Request the number of cores
#SBATCH -n 12

python3 main_mnist.py \
  --epochs 2 \
  --data mnist \
  --batch_size 128 \
  --max_queries 21 \
  --max_queries_test 21 \
  --lr 0.0001 \
  --tau_start 0.2 \
  --tau_end 0.2 \
  --sampling biased \
  --seed 0 \
  --ckpt_path /users/psen/VariationalInformationPursuit/saved/ma9zwmwt/ckpt/epoch1.ckpt \
  --name mnist_biased
