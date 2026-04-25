#!/bin/bash
#SBATCH --job-name=screencoder_eval
#SBATCH --output=sbatch_logs/screencoder_eval_%j.out
#SBATCH --error=sbatch_logs/screencoder_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --account=tesi_dbilardello
#SBATCH --partition=boost_usr_prod
#SBATCH --profile=none
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=285039@studenti.unimore.it

module unload cuda
module load cuda/11.8

source .venv/bin/activate

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

# Use node-local /tmp for Triton cache to avoid NFS file-locking races
# between tensor-parallel workers (OSError: Device or resource busy)
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

export HF_HOME="$MY_CACHE_DIR/huggingface"
export TORCH_HOME="$MY_CACHE_DIR/torch"
export TORCHINDUCTOR_CACHE_DIR="$MY_CACHE_DIR/torchinductor"
export XDG_CACHE_HOME="$MY_CACHE_DIR"
export PIP_CACHE_DIR="$MY_CACHE_DIR/pip"
export MPLCONFIGDIR="$MY_CACHE_DIR/matplotlib"
export PLAYWRIGHT_BROWSERS_PATH="$MY_CACHE_DIR/playwright"
export XDG_CONFIG_HOME="$MY_CACHE_DIR/config"

mkdir -p sbatch_logs

# ---- Configurable paths ----
REPO_ID="${REPO_ID:-leigest519/ScreenBench}"
RUN_DIR="${RUN_DIR:-data/runs/run_${SLURM_JOB_ID}}"
LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

echo "Repo:    $REPO_ID"
echo "Run dir: $RUN_DIR"

# 1) Generate HTML for every screenshot in the dataset
python dataset_runner.py --repo-id "$REPO_ID" --output "$RUN_DIR" $LIMIT_FLAG

# 2) Compute Design2Code-style metrics on the produced run
python evaluation.py --run "$RUN_DIR"
