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
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path, PurePosixPath
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
DATA_INPUT = REPO_ROOT / "data" / "input"
DATA_TMP = REPO_ROOT / "data" / "tmp"
DATA_OUTPUT = REPO_ROOT / "data" / "output"
PIPELINE_STEM = "test1"  # the hardcoded stem used across the pipeline scripts

PIPELINE_SCRIPTS = [
    "block_parsor.py",
    "html_generator.py",
    "image_box_detection.py",
    "UIED/run_single.py",
    "mapping.py",
    "image_replacer.py",
]


def _run_script(script_rel: str):
    script_path = REPO_ROOT / script_rel
    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        cwd=str(REPO_ROOT),
    )


def _clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_pipeline_for_image(image_path: Path, sample_out: Path):
    """Run the full pipeline on `image_path` and collect outputs into `sample_out`."""
    sample_out.mkdir(parents=True, exist_ok=True)
    sample_tmp_dst = sample_out / "tmp"

    # Prepare input slot: the pipeline reads from data/input/test1.png
    DATA_INPUT.mkdir(parents=True, exist_ok=True)
    target_input = DATA_INPUT / f"{PIPELINE_STEM}.png"
    _normalize_to_png(image_path, target_input)

    # Clean intermediate / output dirs for this run
    _clear_dir(DATA_TMP)
    _clear_dir(DATA_OUTPUT)

    # Run the pipeline
    for script in PIPELINE_SCRIPTS:
        _run_script(script)

    # Collect artifacts
    generated_html = DATA_OUTPUT / f"{PIPELINE_STEM}_layout_final.html"
    if not generated_html.exists():
        raise FileNotFoundError(f"Pipeline did not produce expected file: {generated_html}")

    shutil.copy2(target_input, sample_out / "input.png")
    shutil.copy2(generated_html, sample_out / "generated.html")

    # Copy cropped_images dir alongside generated.html (needed for rendering)
    cropped_dir = DATA_OUTPUT / "cropped_images"
    if cropped_dir.exists():
        dst_cropped = sample_out / "cropped_images"
        if dst_cropped.exists():
            shutil.rmtree(dst_cropped)
        shutil.copytree(cropped_dir, dst_cropped)

    # Archive intermediates
    if sample_tmp_dst.exists():
        shutil.rmtree(sample_tmp_dst)
    shutil.copytree(DATA_TMP, sample_tmp_dst)

    # Render generated HTML -> PNG
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


def run_dataset(repo_id: str, output_dir: Path, limit: int | None = None, skip_existing: bool = True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_zip = hf_hub_download(repo_id=repo_id, filename="image.zip", repo_type="dataset")
    html_zip = hf_hub_download(repo_id=repo_id, filename="HTML.zip", repo_type="dataset")

    # Backup the existing data/input/test1.png (if any) so we can restore it later
    backup_input = None
    original_test1 = DATA_INPUT / f"{PIPELINE_STEM}.png"
    if original_test1.exists():
        backup_input = original_test1.read_bytes()

    results = []
    try:
        def _is_real_entry(n: str) -> bool:
            if n.endswith('/'):
                return False
            parts = PurePosixPath(n).parts
            if any(p == '__MACOSX' for p in parts):
                return False
            if PurePosixPath(n).name.startswith('._'):
                return False
            return True

        with ZipFile(img_zip) as iz, ZipFile(html_zip) as hz:
            img_names = [n for n in iz.namelist() if _is_real_entry(n)]
            html_names = [n for n in hz.namelist() if _is_real_entry(n)]
            html_index = {_to_key(n): n for n in html_names}

            pairs = []
            for n in img_names:
                key = _to_key(n)
                if key in html_index:
                    pairs.append((key, n, html_index[key]))
            pairs.sort(key=lambda x: (x[0][0], x[0][1]))

            if limit is not None:
                pairs = pairs[:limit]

            print(f"paired examples: {len(pairs)}")

            for (idx, stem), img_member, html_member in tqdm(pairs, desc="dataset"):
                name = f"{idx}_{stem}"
                sample_out = output_dir / name

                if skip_existing and (sample_out / "generated.html").exists():
                    results.append((name, "skipped"))
                    continue

                sample_out.mkdir(parents=True, exist_ok=True)

                # Write reference HTML
                (sample_out / "reference.html").write_bytes(hz.read(html_member))

                # Extract image to a temp file for the pipeline
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    tf.write(iz.read(img_member))
                    tmp_img_path = Path(tf.name)

                try:
                    run_pipeline_for_image(tmp_img_path, sample_out)
                    results.append((name, "ok"))
                except Exception as e:
                    print(f"[error] {name}: {e}")
                    traceback.print_exc()
                    (sample_out / "error.log").write_text(f"{e}\n\n{traceback.format_exc()}")
                    results.append((name, "error"))
                finally:
                    try:
                        tmp_img_path.unlink()
                    except OSError:
                        pass
    finally:
        if backup_input is not None:
            original_test1.write_bytes(backup_input)

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
    args = ap.parse_args()

    run_dataset(args.repo_id, args.output, limit=args.limit, skip_existing=not args.no_skip_existing)


if __name__ == "__main__":
    main()
