#!/bin/bash
#SBATCH --job-name=screencoder_eval
#SBATCH --output=sbatch_logs/screencoder_eval_%j.out
#SBATCH --error=sbatch_logs/screencoder_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
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
# Compute node has no internet — force offline (model must be pre-downloaded
# on the login node with: huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$MY_CACHE_DIR/torch"
export TORCHINDUCTOR_CACHE_DIR="$MY_CACHE_DIR/torchinductor"
export XDG_CACHE_HOME="$MY_CACHE_DIR"
export PIP_CACHE_DIR="$MY_CACHE_DIR/pip"
export MPLCONFIGDIR="$MY_CACHE_DIR/matplotlib"
export PLAYWRIGHT_BROWSERS_PATH="$MY_CACHE_DIR/playwright"
export XDG_CONFIG_HOME="$MY_CACHE_DIR/config"

mkdir -p sbatch_logs

# ---- Configurable paths ----
REPO_ID="${REPO_ID:-Leigest/ScreenCoder}"
RUN_DIR="${RUN_DIR:-data/runs/run_${SLURM_JOB_ID}}"
LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_URL="http://127.0.0.1:${VLLM_PORT}/v1"
WORKERS="${WORKERS:-8}"

echo "Repo:    $REPO_ID"
echo "Run dir: $RUN_DIR"
echo "Model:   $VLLM_MODEL"
echo "Workers: $WORKERS"

# 1) Start vllm OpenAI server in background (loaded once, reused for all samples)
mkdir -p "$RUN_DIR"
VLLM_LOG="$RUN_DIR/vllm_server.log"
echo "Starting vllm server on port $VLLM_PORT (log: $VLLM_LOG)"
python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --tensor-parallel-size 4 \
    --enforce-eager \
    --limit-mm-per-prompt '{"image": 1}' \
    --trust-remote-code \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "vllm server PID: $VLLM_PID"

# Make sure we kill the server on exit (success, error, or signal)
cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vllm server (PID $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# 2) Generate HTML for every screenshot (waits internally for the server to be ready)
python dataset_runner.py \
    --repo-id "$REPO_ID" \
    --output "$RUN_DIR" \
    --workers "$WORKERS" \
    --vllm-url "$VLLM_URL" \
    --vllm-model "$VLLM_MODEL" \
    $LIMIT_FLAG

# 3) Compute Design2Code-style metrics on the produced run
python evaluation.py --run "$RUN_DIR"
