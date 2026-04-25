"""Run the ScreenCoder pipeline over a Design2Code-style dataset.

Dataset layout expected:
    dataset_dir/
        sample01.png
        sample01.html   (optional, reference ground-truth)
        sample02.png
        ...

Output layout produced:
    output_dir/
        sample01/
            input.png
            reference.html       (if present in dataset)
            generated.html       (final pipeline output)
            rendered.png         (screenshot of generated.html)
            tmp/                 (intermediate artifacts: bboxes, UIED, mapping, ...)
        ...
"""
import argparse
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

from image_box_detection import render_html_to_png

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
    shutil.copy2(image_path, target_input)

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


def run_dataset(dataset_dir: Path, output_dir: Path, limit: int | None = None, skip_existing: bool = True):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(dataset_dir.glob("*.png"))
    if limit is not None:
        images = images[:limit]

    # Backup the existing data/input/test1.png (if any) so we can restore it later
    backup_input = None
    original_test1 = DATA_INPUT / f"{PIPELINE_STEM}.png"
    if original_test1.exists():
        backup_input = original_test1.read_bytes()

    results = []
    try:
        for img in tqdm(images, desc="dataset"):
            name = img.stem
            sample_out = output_dir / name

            if skip_existing and (sample_out / "generated.html").exists():
                results.append((name, "skipped"))
                continue

            # Copy reference HTML if present
            ref_html = dataset_dir / f"{name}.html"
            sample_out.mkdir(parents=True, exist_ok=True)
            if ref_html.exists():
                shutil.copy2(ref_html, sample_out / "reference.html")

            try:
                run_pipeline_for_image(img, sample_out)
                results.append((name, "ok"))
            except Exception as e:
                print(f"[error] {name}: {e}")
                traceback.print_exc()
                (sample_out / "error.log").write_text(f"{e}\n\n{traceback.format_exc()}")
                results.append((name, "error"))
    finally:
        if backup_input is not None:
            original_test1.write_bytes(backup_input)

    ok = sum(1 for _, s in results if s == "ok")
    err = sum(1 for _, s in results if s == "error")
    skp = sum(1 for _, s in results if s == "skipped")
    print(f"\nDataset run complete: {ok} ok, {err} errors, {skp} skipped (total {len(results)})")
    return results


def main():
    ap = argparse.ArgumentParser(description="Run ScreenCoder pipeline over a dataset of screenshots.")
    ap.add_argument("--dataset", type=Path, required=True, help="Directory with {name}.png [+ optional {name}.html] files")
    ap.add_argument("--output", type=Path, required=True, help="Output directory for per-sample results")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N samples")
    ap.add_argument("--no-skip-existing", action="store_true", help="Re-run samples even if generated.html already exists")
    args = ap.parse_args()

    run_dataset(args.dataset, args.output, limit=args.limit, skip_existing=not args.no_skip_existing)


if __name__ == "__main__":
    main()
