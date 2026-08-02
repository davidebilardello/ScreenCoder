#!/bin/bash
#SBATCH --job-name=screencoder_eval
#SBATCH --output=sbatch_logs/screencoder_eval_%j.out
#SBATCH --error=sbatch_logs/screencoder_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=05:00:00
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
# Default (600s) is too short: checkpoint shards load slowly from NFS storage
export VLLM_ENGINE_READY_TIMEOUT_S=2400

export MY_CACHE_DIR="/work/tesi_dbilardello/.cache_ml"
mkdir -p $MY_CACHE_DIR

# Use node-local /tmp for Triton cache to avoid NFS file-locking races
# between tensor-parallel workers (OSError: Device or resource busy)
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

export HF_HOME="$MY_CACHE_DIR/huggingface"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TORCH_HOME="$MY_CACHE_DIR/torch"
export TORCHINDUCTOR_CACHE_DIR="$MY_CACHE_DIR/torchinductor"
export XDG_CACHE_HOME="$MY_CACHE_DIR"
export PIP_CACHE_DIR="$MY_CACHE_DIR/pip"
export MPLCONFIGDIR="$MY_CACHE_DIR/matplotlib"
# Node-local (not shared NFS) so concurrent jobs from other projects can't
# prune/replace the chromium revision this venv's playwright expects.
export PLAYWRIGHT_BROWSERS_PATH="/tmp/playwright_${SLURM_JOB_ID}"
export XDG_CONFIG_HOME="$MY_CACHE_DIR/config"

mkdir -p sbatch_logs

# Must run after PLAYWRIGHT_BROWSERS_PATH is set, otherwise it installs into
# the default (wrong) location and the pipeline can't find the browser later.
mkdir -p "$PLAYWRIGHT_BROWSERS_PATH"
python -m playwright install chromium

# ---- Parse arguments ----
RETRY_RUN_ID=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --retry)
            RETRY_RUN_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# ---- Configurable paths ----
REPO_ID="${REPO_ID:-Leigest/ScreenCoder}"
INPUT_DIR="${INPUT_DIR:-}"

if [ -n "$RETRY_RUN_ID" ]; then
    RUN_DIR="data/runs/run_${RETRY_RUN_ID}"
    echo "Retrying failed samples from run: $RETRY_RUN_ID"
else
    RUN_DIR="${RUN_DIR:-data/runs/run_${SLURM_JOB_ID}}"
    echo "Starting a new run in: $RUN_DIR"
fi
LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

DATA_SRC_FLAG="--repo-id $REPO_ID"
if [ -n "$INPUT_DIR" ]; then
    DATA_SRC_FLAG="--input-dir $INPUT_DIR"
fi

VLLM_PORT="${VLLM_PORT:-8000}"
WORKERS="${WORKERS:-8}"

# ---- Server vllm esterno (run_vllm_server.sbatch)? ----
# Se data/vllm_server.info esiste e il server risponde, lo usiamo e non ne
# avviamo uno locale. Si può anche forzare un URL con VLLM_URL=... sbatch ...
EXTERNAL_VLLM=0
INFO_FILE="data/vllm_server.info"
if [ -n "${VLLM_URL:-}" ]; then
    EXTERNAL_VLLM=1
elif [ -f "$INFO_FILE" ]; then
    # shellcheck source=/dev/null
    source "$INFO_FILE"
    if curl --noproxy '*' -sf --max-time 5 "${VLLM_INFO_URL%/}/models" >/dev/null; then
        EXTERNAL_VLLM=1
        VLLM_URL="$VLLM_INFO_URL"
        VLLM_MODEL="${VLLM_MODEL:-$VLLM_INFO_MODEL}"
        echo "Server vllm esterno trovato: $VLLM_URL (job ${VLLM_INFO_JOB:-?} su ${VLLM_INFO_HOST:-?})"
    else
        echo "File info presente ma il server non risponde: avvio un server locale."
    fi
fi

if [ "$EXTERNAL_VLLM" = "1" ]; then
    # Il client OpenAI e urllib devono bypassare eventuali proxy per il nodo del server
    VLLM_HOSTNAME=$(echo "$VLLM_URL" | sed -E 's|^[a-z]+://([^:/]+).*|\1|')
    export no_proxy="${no_proxy:+$no_proxy,}${VLLM_HOSTNAME},localhost,127.0.0.1"
    export NO_PROXY="$no_proxy"
else
    VLLM_URL="http://127.0.0.1:${VLLM_PORT}/v1"
fi

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"

if [ -n "$INPUT_DIR" ]; then
    echo "Input Dir: $INPUT_DIR"
else
    echo "Repo:      $REPO_ID"
fi
echo "Run dir:   $RUN_DIR"
echo "Model:     $VLLM_MODEL"
echo "Workers:   $WORKERS"

# 1) Start vllm OpenAI server in background (loaded once, reused for all samples)
#    — solo se non stiamo usando un server esterno già attivo.
mkdir -p "$RUN_DIR"
if [ "$EXTERNAL_VLLM" = "1" ]; then
    echo "Uso il server vllm esterno: $VLLM_URL (nessun server locale avviato)"
else
    VLLM_LOG="$RUN_DIR/vllm_server.log"
    echo "Starting vllm server on port $VLLM_PORT (log: $VLLM_LOG)"
    python -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --port "$VLLM_PORT" \
        --gpu-memory-utilization 0.9 \
        --max-model-len 16384 \
        --tensor-parallel-size 2 \
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
fi

# 2) Generate HTML for every screenshot (waits internally for the server to be ready)
# Loading the 32B checkpoint from slow NFS storage can take 30-45min; keep this
# above VLLM_ENGINE_READY_TIMEOUT_S so we don't give up while vllm is still loading.
python dataset_runner.py \
    $DATA_SRC_FLAG \
    --output "$RUN_DIR" \
    --workers "$WORKERS" \
    --vllm-url "$VLLM_URL" \
    --vllm-model "$VLLM_MODEL" \
    --vllm-timeout 3600 \
    --shuffle \
    $LIMIT_FLAG


# 3) Compute Design2Code-style metrics on the produced run
#python evaluation.py --run "$RUN_DIR"
