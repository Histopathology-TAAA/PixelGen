#!/bin/bash

#SBATCH --job-name=train_v2_sharp
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -A bcd

conda init bash
source ~/.bashrc
conda activate pixelgen

set -a
source .env
set +a

hf auth login --token "$HF_TOKEN"
python main_i2i.py fit --config configs_i2i/mist_ki67_v2_sharp.yaml
sleep infinity