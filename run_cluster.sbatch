#!/bin/bash
#SBATCH --job-name=screencoder_vllm
#SBATCH --output=sbatch_logs/screencoder_%j.out
#SBATCH --error=sbatch_logs/screencoder_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --account=tesi_dbilardello
#SBATCH --partition=boost_usr_prod
#SBATCH --profile=none
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=285039@studenti.unimore.it

module unload cuda
module load cuda/11.8

source .venv/bin/activate

# Variabili d'ambiente essenziali per evitare conflitti C++ / OpenMP segnalati prima
export KMP_DUPLICATE_LIB_OK=TRUE
export USE_TORCH=1
export USE_TF=0
export USE_PADDLE=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_NO_USAGE_STATS=1

export MY_CACHE_DIR="/work/tesi_dbilardello/.cache_ml"
mkdir -p $MY_CACHE_DIR

export HF_HOME="$MY_CACHE_DIR/huggingface"
export TORCH_HOME="$MY_CACHE_DIR/torch"
export TORCHINDUCTOR_CACHE_DIR="$MY_CACHE_DIR/torchinductor"
export TRITON_CACHE_DIR="$MY_CACHE_DIR/triton"
export XDG_CACHE_HOME="$MY_CACHE_DIR"
export PIP_CACHE_DIR="$MY_CACHE_DIR/pip"
export MPLCONFIGDIR="$MY_CACHE_DIR/matplotlib"
export PLAYWRIGHT_BROWSERS_PATH="$MY_CACHE_DIR/playwright"
export XDG_CONFIG_HOME="$MY_CACHE_DIR/config"

mkdir -p sbatch_logs

python main.py