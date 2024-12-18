#!/bin/sh
#SBATCH -p hgx
#SBATCH -w hgx1
#SBATCH -c26
#SBATCH --gres=gpu:1

cd /home/inf151841/anomaly-detection
eval "$(/home/inf151841/anaconda3/bin/conda shell.bash hook)"

make run ARGS="--no-load --epochs=80 --eval-interval=3 --subset-size=1000000 --learning-rate=0.001 --batch-size=52 --num-workers=26"
