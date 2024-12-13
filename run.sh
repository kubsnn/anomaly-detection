#!/bin/sh
#SBATCH -p hgx
#SBATCH -w hgx1
#SBATCH -c52
#SBATCH --gres=gpu:1

cd /home/inf151841/anomaly-detection
eval "$(/home/inf151841/anaconda3/bin/conda shell.bash hook)"

make run