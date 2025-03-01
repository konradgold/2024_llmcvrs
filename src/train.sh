#!/bin/bash
#SBATCH -A doellner-student
#SBATCH --partition sorcery
#SBATCH -C "GPU_MEM:40GB"
#SBATCH --gpus 1
#SBATCH --mem 32G
#SBATCH --cpus-per-task=16
#SBATCH --job-name "train_model"
#SBATCH --output train_model.txt
#SBATCH --time 24:00:00


cd 2024_llmcvrs/src
source "/hpi/fs00/home/konrad.goldenbaum/miniconda3/etc/profile.d/conda.sh"

conda activate llmcvrs

souce .env

python train_model.py