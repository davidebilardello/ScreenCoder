#!/bin/bash
#SBATCH --job-name=screencoder_metrics
#SBATCH --output=sbatch_logs/screencoder_metrics_%j.out
#SBATCH --error=sbatch_logs/screencoder_metrics_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
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

export MY_CACHE_DIR="/work/tesi_dbilardello/.cache_ml"
mkdir -p $MY_CACHE_DIR
export HF_HOME="$MY_CACHE_DIR/huggingface"
export TORCH_HOME="$MY_CACHE_DIR/torch"
export XDG_CACHE_HOME="$MY_CACHE_DIR"
export PIP_CACHE_DIR="$MY_CACHE_DIR/pip"
export MPLCONFIGDIR="$MY_CACHE_DIR/matplotlib"
export PLAYWRIGHT_BROWSERS_PATH="$MY_CACHE_DIR/playwright"
export XDG_CONFIG_HOME="$MY_CACHE_DIR/config"

mkdir -p sbatch_logs

# --- Choose RUN_DIR: edit this line, or pass as $1, or export RUN_DIR ---
RUN_DIR="${1:-${RUN_DIR:-data/runs/run_31263}}"
if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: RUN_DIR '$RUN_DIR' does not exist."
    echo "Edit run_metrics_only.sbatch and set the default, or run:"
    echo "  sbatch run_metrics_only.sbatch <path-to-run-dir>"
    exit 1
fi

echo "Run dir: $RUN_DIR"
python evaluation.py --run "$RUN_DIR"
