#!/bin/bash -l
# Questo script intercetta i comandi di PyCharm e li invia al nodo di calcolo

# Carichiamo CUDA come facevi nel file sbatch
module unload cuda
module load cuda/11.8

# Percorso assoluto dell'interprete Python del virtual environment
VENV_PYTHON="/work/tesi_dbilardello/ScreenCoder/.venv/bin/python"

# Comando SLURM su più righe per una migliore leggibilità
srun -Q \
  --immediate=10 \
  -w ailb-login-02 \
  --partition=all_serial \
  --account=tesi_dbilardello \
  --gres=gpu:1 \
  --time=60:00 \
  "$VENV_PYTHON" "$@"

