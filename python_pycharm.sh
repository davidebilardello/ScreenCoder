#!/bin/bash
    source /etc/profile
    # Il venv è basato su /usr/bin/python3.10 di sistema ed è autosufficiente:
    # niente module load (il vecchio modulo python spack non esiste più,
    # le librerie CUDA arrivano dai wheel pip di torch/vllm).

    exec /work/tesi_dbilardello/ScreenCoder/.venv/bin/python "$@"
