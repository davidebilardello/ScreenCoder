import subprocess
import sys
import os
import time
import atexit

def setup_environment():
    """Imposta le stesse variabili d'ambiente del file sbatch."""
    print("Setting up environment variables...")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["USE_TORCH"] = "1"
    os.environ["USE_TF"] = "0"
    os.environ["USE_PADDLE"] = "0"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_pir_in_executor"] = "0"
    os.environ["FLAGS_enable_new_ir_in_executor"] = "0"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    
    # Usa una cartella '.cache_ml' locale al progetto per non sporcare il sistema
    # (Su cluster sbatch usavi /work/tesi_dbilardello/.cache_ml)
    cache_dir = os.path.abspath(".cache_ml")
    os.makedirs(cache_dir, exist_ok=True)
    
    os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface")
    os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_dir, "torchinductor")
    os.environ["MPLCONFIGDIR"] = os.path.join(cache_dir, "matplotlib")
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = os.path.join(cache_dir, "playwright")
    
    # Crea in anticipo le cartelle per evitare problemi
    for path in [os.environ["HF_HOME"], os.environ["TORCH_HOME"], os.environ["MPLCONFIGDIR"]]:
        os.makedirs(path, exist_ok=True)

    # Scarica i browser di Playwright automaticamente (utile per il debug via PyCharm)
    print("Verifica/Installazione dei browser di Playwright in corso...")
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
            env=os.environ.copy()
        )
    except Exception as e:
        print(f"Attenzione: Impossibile installare i browser di Playwright: {e}")

def main():
    setup_environment()
    print("\nStarting the Screencoder test workflow...")
    
    # Check if an input argument was provided
    input_path = "data/input/test3.png"  # Di default usa test2.png se clicchi "Play" da PyCharm
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    
    # Cartella in cui verranno salvati i risultati del test
    run_dir = os.path.join("data", "runs", "test_local_run")
    os.makedirs(run_dir, exist_ok=True)
    
    print("\n--- 0. Starting vLLM Server ---")
    print("vLLM Server startup is DISABLED in main.py.")
    print("Make sure you have manually started the vLLM server on port 8000 in another terminal!")
    
    print("\n--- 1. Running Dataset Runner (Test Mode) ---")
    cmd_runner = [
        sys.executable, "dataset_runner.py",
        "--repo-id", "Leigest/ScreenCoder", 
        "--output", run_dir,
        "--workers", "1",
        "--vllm-url", "http://127.0.0.1:8000/v1",
        "--vllm-model", "Qwen/Qwen2.5-VL-32B-Instruct"
    ]
    
    # Handle single file vs directory
    temp_dir = None
    if os.path.isfile(input_path):
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp(prefix="screencoder_input_")
        shutil.copy(input_path, temp_dir)
        cmd_runner.extend(["--input-dir", temp_dir, "--limit", "1"])
    else:
        cmd_runner.extend(["--input-dir", input_path, "--limit", "1"])
    
    print(f"Executing: {' '.join(cmd_runner)}")
    try:
        subprocess.run(cmd_runner, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] dataset_runner.py failed with exit code {e.returncode}")
        sys.exit(1)
    finally:
        # Cleanup temporary directory if created
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

    print("\n--- 2. Running Evaluation ---")
    cmd_eval = [
        sys.executable, "evaluation.py",
        "--run", run_dir
    ]
    print(f"Executing: {' '.join(cmd_eval)}")
    try:
        subprocess.run(cmd_eval, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] evaluation.py failed with exit code {e.returncode}")
        sys.exit(1)

    print(f"\nTest completed successfully! Check the output in: {run_dir}")

if __name__ == "__main__":
    main()
