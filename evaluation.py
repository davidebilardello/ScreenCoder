"""Design2Code-style evaluation for ScreenCoder runs.

Compares each generated rendering against its reference screenshot using:
  * CLIP similarity (high-level, image-image cosine)
  * OCR-block matching (low-level): block reproduction, text, spatial, color

Expected run-dir layout (produced by dataset_runner.py):
    run_dir/
        sample01/
            input.png       (reference screenshot)
            rendered.png    (rendering of generated.html)
        ...

Outputs:
    run_dir/metrics.json
    run_dir/metrics.csv
"""
import argparse
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ---- Lazy singletons -----------------------------------------------------

_CLIP = {"model": None, "processor": None, "device": None}
_OCR = {"model": None}


def _get_clip():
    if _CLIP["model"] is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "openai/clip-vit-base-patch32"
        _CLIP["model"] = CLIPModel.from_pretrained(model_name).to(device).eval()
        _CLIP["processor"] = CLIPProcessor.from_pretrained(model_name)
        _CLIP["device"] = device
    return _CLIP["model"], _CLIP["processor"], _CLIP["device"]


def _get_ocr():
    if _OCR["model"] is None:
        from paddleocr import PaddleOCR
        _OCR["model"] = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _OCR["model"]


# ---- High-level: CLIP similarity ----------------------------------------

def clip_similarity(img_a_path: Path, img_b_path: Path) -> float:
    import torch
    from PIL import Image

    model, processor, device = _get_clip()
    images = [Image.open(img_a_path).convert("RGB"), Image.open(img_b_path).convert("RGB")]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    sim = (feats[0] @ feats[1]).item()
    # Cosine in [-1,1] -> map to [0,1]
    return float((sim + 1.0) / 2.0)


# ---- Low-level: OCR block matching --------------------------------------

def ocr_blocks(image_path: Path):
    """Return a list of {text, bbox=(x1,y1,x2,y2), color=(r,g,b)} for `image_path`."""
    ocr = _get_ocr()
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    H, W = img.shape[:2]
    raw = ocr.ocr(str(image_path), cls=True)
    blocks = []
    if not raw:
        return blocks
    # PaddleOCR returns [[(poly, (text, conf)), ...]] (one entry per page)
    page = raw[0] if raw and isinstance(raw[0], list) else raw
    if page is None:
        return blocks
    for entry in page:
        try:
            poly, (text, conf) = entry
        except Exception:
            continue
        if not text or not text.strip():
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
        x2, y2 = min(W, int(max(xs))), min(H, int(max(ys)))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # cv2 is BGR -> convert to RGB tuple
        b, g, r = crop.reshape(-1, 3).mean(axis=0)
        blocks.append({
            "text": text.strip(),
            "bbox": (x1, y1, x2, y2),
            "color": (float(r), float(g), float(b)),
            "img_size": (W, H),
        })
    return blocks


def _text_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _center_norm(bbox, size):
    x1, y1, x2, y2 = bbox
    W, H = size
    return np.array([(x1 + x2) / 2.0 / max(W, 1), (y1 + y2) / 2.0 / max(H, 1)])


def block_metrics(ref_blocks, pred_blocks, text_threshold: float = 0.5) -> dict:
    """Match blocks by text similarity (Hungarian) and compute the 4 low-level metrics."""
    if not ref_blocks or not pred_blocks:
        return {"block": 0.0, "text": 0.0, "spatial": 0.0, "color": 0.0,
                "n_ref": len(ref_blocks), "n_pred": len(pred_blocks), "n_matched": 0}

    n_r, n_p = len(ref_blocks), len(pred_blocks)
    cost = np.ones((n_r, n_p), dtype=np.float64)
    sim_mat = np.zeros_like(cost)
    for i, rb in enumerate(ref_blocks):
        for j, pb in enumerate(pred_blocks):
            s = _text_sim(rb["text"], pb["text"])
            sim_mat[i, j] = s
            cost[i, j] = 1.0 - s

    row_ind, col_ind = linear_sum_assignment(cost)

    text_scores, spatial_scores, color_scores = [], [], []
    matched = 0
    for i, j in zip(row_ind, col_ind):
        if sim_mat[i, j] < text_threshold:
            continue
        matched += 1
        rb, pb = ref_blocks[i], pred_blocks[j]
        text_scores.append(sim_mat[i, j])

        c_r = _center_norm(rb["bbox"], rb["img_size"])
        c_p = _center_norm(pb["bbox"], pb["img_size"])
        # Centers in [0,1]^2 -> distance in [0, sqrt(2)]
        spatial_scores.append(1.0 - float(np.linalg.norm(c_r - c_p)) / np.sqrt(2.0))

        c1 = np.array(rb["color"]) / 255.0
        c2 = np.array(pb["color"]) / 255.0
        # Per-channel L1 distance in [0,1] averaged
        color_scores.append(1.0 - float(np.mean(np.abs(c1 - c2))))

    block_score = matched / max(n_r, n_p)
    return {
        "block": float(block_score),
        "text": float(np.mean(text_scores)) if text_scores else 0.0,
        "spatial": float(np.mean(spatial_scores)) if spatial_scores else 0.0,
        "color": float(np.mean(color_scores)) if color_scores else 0.0,
        "n_ref": n_r,
        "n_pred": n_p,
        "n_matched": matched,
    }


# ---- Per-sample / dataset orchestration ---------------------------------

METRIC_KEYS = ["clip", "block", "text", "spatial", "color"]


def evaluate_example(reference_png: Path, rendered_png: Path) -> dict:
    metrics = {"clip": clip_similarity(reference_png, rendered_png)}
    ref_blocks = ocr_blocks(reference_png)
    pred_blocks = ocr_blocks(rendered_png)
    metrics.update(block_metrics(ref_blocks, pred_blocks))
    return metrics


def evaluate_dataset(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    samples = sorted(p for p in run_dir.iterdir() if p.is_dir())

    per_example = {}
    for sample in tqdm(samples, desc="evaluate"):
        ref = sample / "input.png"
        pred = sample / "rendered.png"
        if not ref.exists() or not pred.exists():
            print(f"[skip] {sample.name}: missing input.png or rendered.png")
            continue
        try:
            per_example[sample.name] = evaluate_example(ref, pred)
        except Exception as e:
            print(f"[error] {sample.name}: {e}")
            per_example[sample.name] = {"error": str(e)}

    # Aggregate (mean over numeric metrics, ignoring errored examples)
    aggregate = {}
    for k in METRIC_KEYS:
        values = [m[k] for m in per_example.values() if isinstance(m, dict) and k in m]
        aggregate[k] = float(np.mean(values)) if values else 0.0

    result = {"per_example": per_example, "aggregate": aggregate}

    out_json = run_dir / "metrics.json"
    out_json.write_text(json.dumps(result, indent=2))

    out_csv = run_dir / "metrics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample"] + METRIC_KEYS + ["n_ref", "n_pred", "n_matched"])
        for name, m in per_example.items():
            if not isinstance(m, dict) or "error" in m:
                w.writerow([name] + [""] * (len(METRIC_KEYS) + 3))
                continue
            w.writerow([name] + [m.get(k, "") for k in METRIC_KEYS]
                       + [m.get("n_ref", ""), m.get("n_pred", ""), m.get("n_matched", "")])
        w.writerow(["AGGREGATE"] + [aggregate[k] for k in METRIC_KEYS] + ["", "", ""])

    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")
    print("Aggregate:", json.dumps(aggregate, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description="Evaluate a ScreenCoder dataset run with Design2Code-style metrics.")
    ap.add_argument("--run", type=Path, required=True, help="Run directory produced by dataset_runner.py")
    args = ap.parse_args()
    evaluate_dataset(args.run)


if __name__ == "__main__":
    main()
