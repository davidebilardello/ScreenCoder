#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --output=sbatch_logs/vllm_server_%j.out
#SBATCH --error=sbatch_logs/vllm_server_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --account=tesi_dbilardello
#SBATCH --partition=boost_usr_prod
#SBATCH --profile=none
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=285039@studenti.unimore.it

# Server vLLM persistente: il modello resta caricato tra un run e l'altro.
# Uso:   sbatch run_vllm_server.sbatch
# Poi:   LIMIT=5 sbatch --gres=gpu:0 run_eval_cluster.sbatch
#        (l'eval trova il server tramite data/vllm_server.info e non ne avvia uno suo)
# Stop:  scancel <job_id>   (il file info viene rimosso automaticamente)

module unload cuda
module load cuda/11.8

source .venv/bin/activate

export KMP_DUPLICATE_LIB_OK=TRUE
export NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_NO_USAGE_STATS=1
# Default (600s) is too short: checkpoint shards load slowly from NFS storage
export VLLM_ENGINE_READY_TIMEOUT_S=2400

export MY_CACHE_DIR="/work/tesi_dbilardello/.cache_ml"
export HF_HOME="$MY_CACHE_DIR/huggingface"
export TORCH_HOME="$MY_CACHE_DIR/torch"
export TORCHINDUCTOR_CACHE_DIR="$MY_CACHE_DIR/torchinductor"
export XDG_CACHE_HOME="$MY_CACHE_DIR"

# Cache Triton su /tmp locale del nodo (evita lock NFS tra i worker tensor-parallel)
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" sbatch_logs data

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
INFO_FILE="data/vllm_server.info"
VLLM_LOG="sbatch_logs/vllm_server_${SLURM_JOB_ID}.log"
HOST=$(hostname -s)

echo "Modello: $VLLM_MODEL"
echo "Nodo:    $HOST  porta: $VLLM_PORT"
echo "Log:     $VLLM_LOG"

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

cleanup() {
    rm -f "$INFO_FILE"
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vllm server (PID $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Pubblica il file info solo quando il server risponde davvero
echo "Attendo che il server sia pronto..."
until curl --noproxy '*' -sf --max-time 3 "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERRORE: il server vllm è terminato durante il caricamento. Vedi $VLLM_LOG"
        exit 1
    fi
    sleep 5
done

cat > "$INFO_FILE" <<EOF
VLLM_INFO_URL=http://${HOST}:${VLLM_PORT}/v1
VLLM_INFO_MODEL=${VLLM_MODEL}
VLLM_INFO_HOST=${HOST}
VLLM_INFO_PORT=${VLLM_PORT}
VLLM_INFO_JOB=${SLURM_JOB_ID}
EOF
echo "Server pronto: http://${HOST}:${VLLM_PORT}/v1  (info: $INFO_FILE)"

# Resta in vita finché il server gira (o fino a scancel/walltime)
wait "$VLLM_PID"
