"""Run the ScreenCoder pipeline over the ScreenBench HuggingFace dataset.

Dataset source:
    HuggingFace repo (default: leigest519/ScreenBench) providing image.zip and HTML.zip.
    Pairs are matched by (top-level-folder, file-stem).

Output layout produced:
    output_dir/
        {idx}_{stem}/
            input.png
            reference.html       (ground-truth from HTML.zip)
            generated.html       (final pipeline output)
            rendered.png         (screenshot of generated.html)
            tmp/                 (intermediate artifacts: bboxes, UIED, mapping, ...)
        ...
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib import error as urlerror
from urllib import request as urlrequest
from zipfile import ZipFile

import cv2
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from image_box_detection import render_html_to_png


def _normalize_to_png(src: Path, dst: Path):
    """Save `src` as a valid PNG at `dst`, regardless of original format.
    Tries PIL, cv2, imageio, and SVG rasterization in turn."""
    try:
        with Image.open(src) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img.save(dst, format="PNG")
            return
    except (UnidentifiedImageError, OSError):
        pass

    arr = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if arr is not None and cv2.imwrite(str(dst), arr):
        return

    head = src.read_bytes()[:4096]
    head_lc = head.lstrip().lower()
    if head_lc.startswith(b"<?xml") or head_lc.startswith(b"<svg"):
        try:
            import cairosvg
            cairosvg.svg2png(bytestring=src.read_bytes(), write_to=str(dst))
            return
        except ImportError:
            raise ValueError(
                f"Image {src} is SVG; install cairosvg to rasterize it."
            )

    try:
        import imageio.v3 as iio
        arr = iio.imread(str(src))
        Image.fromarray(arr).save(dst, format="PNG")
        return
    except Exception:
        pass

    magic = head[:16]
    raise ValueError(
        f"Could not decode image: {src} (magic bytes: {magic!r})"
    )

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_STEM = "test1"  # the hardcoded stem used across the pipeline scripts

PIPELINE_SCRIPTS = [
    "block_parsor.py",
    "html_generator.py",
    "image_box_detection.py",
    "UIED/run_single.py",
    "mapping.py",
    "image_replacer.py",
]


def _run_script(script_rel: str, env: dict):
    script_path = REPO_ROOT / script_rel
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail_out = "\n".join(proc.stdout.splitlines()[-40:])
        tail_err = "\n".join(proc.stderr.splitlines()[-40:])
        raise RuntimeError(
            f"{script_rel} failed (exit {proc.returncode})\n"
            f"--- stdout (tail) ---\n{tail_out}\n"
            f"--- stderr (tail) ---\n{tail_err}"
        )


def _wait_for_vllm(base_url: str, timeout: float = 300.0) -> bool:
    """Poll the vllm /v1/models endpoint until it returns 200 or timeout."""
    deadline = time.time() + timeout
    url = base_url.rstrip("/") + "/models"
    while time.time() < deadline:
        try:
            with urlrequest.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except (urlerror.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2.0)
    return False


def run_pipeline_for_image(image_path: Path, sample_out: Path, work_root: Path):
    """Run the full pipeline on `image_path` and collect outputs into `sample_out`.
    `work_root` is an isolated per-sample directory used for intermediate I/O."""
    sample_out.mkdir(parents=True, exist_ok=True)
    sample_tmp_dst = sample_out / "tmp"

    work_input = work_root / "data" / "input"
    work_tmp = work_root / "data" / "tmp"
    work_output = work_root / "data" / "output"
    for d in (work_input, work_tmp, work_output):
        d.mkdir(parents=True, exist_ok=True)

    target_input = work_input / f"{PIPELINE_STEM}.png"
    _normalize_to_png(image_path, target_input)

    env = os.environ.copy()
    env["SCREENCODER_INPUT_DIR"] = str(work_input)
    env["SCREENCODER_TMP_DIR"] = str(work_tmp)
    env["SCREENCODER_OUTPUT_DIR"] = str(work_output)

    for script in PIPELINE_SCRIPTS:
        _run_script(script, env)

    generated_html = work_output / f"{PIPELINE_STEM}_layout_final.html"
    if not generated_html.exists():
        raise FileNotFoundError(f"Pipeline did not produce expected file: {generated_html}")

    shutil.copy2(target_input, sample_out / "input.png")
    shutil.copy2(generated_html, sample_out / "generated.html")

    cropped_dir = work_output / "cropped_images"
    if cropped_dir.exists():
        dst_cropped = sample_out / "cropped_images"
        if dst_cropped.exists():
            shutil.rmtree(dst_cropped)
        shutil.copytree(cropped_dir, dst_cropped)

    if sample_tmp_dst.exists():
        shutil.rmtree(sample_tmp_dst)
    shutil.copytree(work_tmp, sample_tmp_dst)

    rendered_png = sample_out / "rendered.png"
    try:
        render_html_to_png(sample_out / "generated.html", rendered_png)
    except Exception as e:
        print(f"[warn] rendering failed for {sample_out.name}: {e}")


def _to_key(name: str):
    p = PurePosixPath(name)
    idx = p.parts[0]
    stem = PurePosixPath(p.name).stem
    return idx, stem


def _is_real_entry(n: str) -> bool:
    if n.endswith('/'):
        return False
    parts = PurePosixPath(n).parts
    if any(p == '__MACOSX' for p in parts):
        return False
    if PurePosixPath(n).name.startswith('._'):
        return False
    return True


def _process_one_sample(name: str, img_bytes: bytes, html_bytes: bytes,
                        sample_out: Path, work_root: Path) -> tuple[str, str]:
    sample_out.mkdir(parents=True, exist_ok=True)
    (sample_out / "reference.html").write_bytes(html_bytes)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tf.write(img_bytes)
        tmp_img_path = Path(tf.name)

    try:
        run_pipeline_for_image(tmp_img_path, sample_out, work_root)
        return (name, "ok")
    except Exception as e:
        print(f"[error] {name}: {e}")
        traceback.print_exc()
        (sample_out / "error.log").write_text(f"{e}\n\n{traceback.format_exc()}")
        return (name, "error")
    finally:
        try:
            tmp_img_path.unlink()
        except OSError:
            pass
        if work_root.exists():
            shutil.rmtree(work_root, ignore_errors=True)


def run_dataset(repo_id: str, output_dir: Path, limit: int | None = None,
                skip_existing: bool = True, workers: int = 1,
                vllm_url: str | None = None,
                vllm_model: str | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_remote = bool(vllm_url) or os.environ.get("SCREENCODER_USE_REMOTE_VLLM") == "1"
    if use_remote:
        if vllm_url:
            os.environ["SCREENCODER_VLLM_URL"] = vllm_url
        if vllm_model:
            os.environ["SCREENCODER_VLLM_MODEL"] = vllm_model
        os.environ["SCREENCODER_USE_REMOTE_VLLM"] = "1"

        url = os.environ.get("SCREENCODER_VLLM_URL", "http://localhost:8000/v1")
        print(f"Waiting for vllm server at {url} ...")
        if not _wait_for_vllm(url):
            raise RuntimeError(f"vllm server at {url} did not become ready within timeout")
        print(f"vllm server is up at {url}")
    elif workers > 1:
        print("[warn] workers>1 without remote vllm: each worker would load its own "
              "in-process vllm. Consider --vllm-url to use a shared server.")

    img_zip = hf_hub_download(repo_id=repo_id, filename="image.zip", repo_type="dataset")
    html_zip = hf_hub_download(repo_id=repo_id, filename="HTML.zip", repo_type="dataset")

    workdirs_root = output_dir / "_workdirs"
    workdirs_root.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str]] = []

    with ZipFile(img_zip) as iz, ZipFile(html_zip) as hz:
        img_names = [n for n in iz.namelist() if _is_real_entry(n)]
        html_names = [n for n in hz.namelist() if _is_real_entry(n)]

        print(f"image.zip entries: {len(img_names)} (e.g. {img_names[:3]})")
        print(f"HTML.zip entries:  {len(html_names)} (e.g. {html_names[:3]})")

        html_index = {_to_key(n): n for n in html_names}
        pairs = []
        for n in img_names:
            key = _to_key(n)
            if key in html_index:
                pairs.append((key, n, html_index[key]))

        if not pairs:
            print("[warn] no (folder, stem) matches; falling back to stem-only pairing")
            html_by_stem = {PurePosixPath(n).stem: n for n in html_names}
            for n in img_names:
                stem = PurePosixPath(n).stem
                if stem in html_by_stem:
                    pairs.append(((stem, stem), n, html_by_stem[stem]))

        pairs.sort(key=lambda x: (x[0][0], x[0][1]))

        if limit is not None:
            pairs = pairs[:limit]

        print(f"paired examples: {len(pairs)}")

        # Pre-filter skipped, pre-load bytes (zip not thread-safe across reads)
        zip_lock = threading.Lock()
        todo = []
        for (idx, stem), img_member, html_member in pairs:
            name = f"{idx}_{stem}"
            sample_out = output_dir / name
            if skip_existing and (sample_out / "generated.html").exists():
                results.append((name, "skipped"))
                continue
            todo.append((name, img_member, html_member, sample_out))

        def _read_pair(img_member: str, html_member: str) -> tuple[bytes, bytes]:
            with zip_lock:
                return iz.read(img_member), hz.read(html_member)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = {}
            for i, (name, img_member, html_member, sample_out) in enumerate(todo):
                img_bytes, html_bytes = _read_pair(img_member, html_member)
                work_root = workdirs_root / f"w{i}_{name}"
                fut = ex.submit(_process_one_sample, name, img_bytes, html_bytes,
                                sample_out, work_root)
                futures[fut] = name

            for fut in tqdm(as_completed(futures), total=len(futures), desc="dataset"):
                results.append(fut.result())

    if workdirs_root.exists():
        shutil.rmtree(workdirs_root, ignore_errors=True)

    ok = sum(1 for _, s in results if s == "ok")
    err = sum(1 for _, s in results if s == "error")
    skp = sum(1 for _, s in results if s == "skipped")
    print(f"\nDataset run complete: {ok} ok, {err} errors, {skp} skipped (total {len(results)})")
    return results


def main():
    ap = argparse.ArgumentParser(description="Run ScreenCoder pipeline over the ScreenBench HF dataset.")
    ap.add_argument("--repo-id", type=str, default="Leigest/ScreenCoder", help="HuggingFace dataset repo id")
    ap.add_argument("--output", type=Path, required=True, help="Output directory for per-sample results")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N samples")
    ap.add_argument("--no-skip-existing", action="store_true", help="Re-run samples even if generated.html already exists")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SCREENCODER_WORKERS", "1")),
                    help="Number of concurrent samples (each fires HTTP calls to the shared vllm server)")
    ap.add_argument("--vllm-url", type=str, default=os.environ.get("SCREENCODER_VLLM_URL"),
                    help="Base URL of a running vllm OpenAI-compatible server (e.g. http://localhost:8000/v1). "
                         "If set, scripts use the remote client (no in-process model load).")
    ap.add_argument("--vllm-model", type=str, default=os.environ.get("SCREENCODER_VLLM_MODEL"),
                    help="Model name to send to the vllm server (must match what the server is serving).")
    args = ap.parse_args()

    run_dataset(args.repo_id, args.output, limit=args.limit,
                skip_existing=not args.no_skip_existing,
                workers=args.workers, vllm_url=args.vllm_url, vllm_model=args.vllm_model)


if __name__ == "__main__":
    main()
