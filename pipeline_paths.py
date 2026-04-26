"""Centralized path resolution for the ScreenCoder pipeline.

When running multiple samples in parallel, each worker sets these
environment variables so the (otherwise hardcoded) paths in each script
resolve to a per-sample isolated directory.
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_STEM = "test1"


def _resolved(env_key: str, default_rel: str) -> Path:
    val = os.environ.get(env_key)
    if val:
        return Path(val).resolve()
    return (_REPO_ROOT / default_rel).resolve()


def input_dir() -> Path:
    return _resolved("SCREENCODER_INPUT_DIR", "data/input")


def tmp_dir() -> Path:
    return _resolved("SCREENCODER_TMP_DIR", "data/tmp")


def output_dir() -> Path:
    return _resolved("SCREENCODER_OUTPUT_DIR", "data/output")


def use_remote_vllm() -> bool:
    return os.environ.get("SCREENCODER_USE_REMOTE_VLLM", "").lower() in ("1", "true", "yes")
