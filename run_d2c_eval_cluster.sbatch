#!/bin/bash
#SBATCH --job-name=screencoder_d2c_eval
#SBATCH --output=sbatch_logs/screencoder_d2c_eval_%j.out
#SBATCH --error=sbatch_logs/screencoder_d2c_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=tesi_dbilardello
#SBATCH --partition=boost_usr_prod
#SBATCH --profile=none
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=285039@studenti.unimore.it
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"


module unload cuda
module load cuda/11.8

source .venv/bin/activate

export KMP_DUPLICATE_LIB_OK=TRUE
export USE_TORCH=1
export USE_TF=0
export USE_PADDLE=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Disable PaddlePaddle oneDNN + new PIR executor
export FLAGS_use_mkldnn=0
export FLAGS_enable_pir_in_executor=0
export FLAGS_enable_new_ir_in_executor=0

export MY_CACHE_DIR="/work/tesi_dbilardello/.cache_ml"
mkdir -p $MY_CACHE_DIR
export HF_HOME="$MY_CACHE_DIR/huggingface"
export TORCH_HOME="$MY_CACHE_DIR/torch"
export XDG_CACHE_HOME="$MY_CACHE_DIR"
export PIP_CACHE_DIR="$MY_CACHE_DIR/pip"
export MPLCONFIGDIR="$MY_CACHE_DIR/matplotlib"
export PLAYWRIGHT_BROWSERS_PATH="/tmp/playwright_${SLURM_JOB_ID}"
export XDG_CONFIG_HOME="$MY_CACHE_DIR/config"

mkdir -p sbatch_logs

# --- Retrieve the latest run if not provided ---
if [ -z "$1" ]; then
    LATEST_RUN=$(ls -td data/runs/run_* | head -n 1)
    RUN_DIR="${RUN_DIR:-$LATEST_RUN}"
else
    RUN_DIR="$1"
fi
shift || true
EXTRA_ARGS="$@"

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: RUN_DIR '$RUN_DIR' does not exist."
    echo "Usage: sbatch run_d2c_eval_cluster.sbatch [<path-to-run-dir>] [--use-input-screenshot]"
    exit 1
fi

echo "Installing Playwright chromium browser to cache..."
python -m playwright install chromium

echo "Running D2C official evaluation on run dir: $RUN_DIR $EXTRA_ARGS"
python run_d2c_eval.py "$RUN_DIR" $EXTRA_ARGS
