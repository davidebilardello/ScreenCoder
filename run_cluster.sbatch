#!/bin/bash
#SBATCH --job-name=screencoder_vllm
#SBATCH --output=sbatch_logs/screencoder_%j.out
#SBATCH --error=sbatch_logs/screencoder_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=cvcs2025
#SBATCH --partition=all_usr_prod

module unload cuda
module load cuda/11.8

source activate screencoder

source .venv/bin/activate

# Variabili d'ambiente essenziali per evitare conflitti C++ / OpenMP segnalati prima
export KMP_DUPLICATE_LIB_OK=TRUE
export USE_TORCH=1
export USE_TF=0
export USE_PADDLE=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

export VLLM_LOGGING_LEVEL=WARNING

mkdir -p sbatch_logs

python main.py

