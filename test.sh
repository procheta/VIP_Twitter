#!/bin/bash -l
# Use the current working directory, which is the default setting. #SBATCH -D ./
# Use the current environment for this job, which is the default setting. #SBATCH --export=ALL

#SBATCH -p gpulowbig
#SBATCH --gres=gpu:1
#SBATCH -N 1
# Request the number of cores
#SBATCH -n 12


python3 main_news.py \
  --epochs 50 \
  --data_dir /users/psen/data/ \
  --ckpt_path /users/psen/VariationalInformationPursuit/saved/7y85zl5d/ckpt/epoch49.ckpt \
  --batch_size 128 \
  --max_queries 300 \
  --max_queries_test 300 \
  --lr 0.0001 \
  --tau_start 1.0 \
  --tau_end 0.2 \
  --sampling biased \
  --seed 0 \
  --name news_bias_latest
