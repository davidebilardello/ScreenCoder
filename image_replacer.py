import argparse
import json
from pathlib import Path
from bs4 import BeautifulSoup
import cv2
import re

from pipeline_paths import input_dir, tmp_dir, output_dir, PIPELINE_STEM


def main(args):
    # --- Phase 1: Crop and Save All Images First ---

    # 1. Load data
    if not args.mapping.exists():
        print(f"Warning: Mapping file {args.mapping} not found. Skipping image replacement.")
        args.output_html.parent.mkdir(exist_ok=True, parents=True)
        if args.gray_html.exists():
            args.output_html.write_text(
                args.gray_html.read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8",
            )
        with open(tmp_dir() / "degraded.flag", "a", encoding="utf-8") as f:
            f.write("image_replacer: mapping file missing, gray HTML used as final output\n")
        return

    mapping_data = json.loads(args.mapping.read_text())
    uied_data = json.loads(args.uied.read_text())
    original_image = cv2.imread(str(args.original_image))

    if original_image is None:
        raise ValueError(f"Could not load the original image from {args.original_image}")

    # Get image shapes to calculate a simple, global scaling factor
    H_proc, W_proc, _ = uied_data['img_shape']
    H_orig, W_orig, _ = original_image.shape
    scale_x = W_orig / W_proc
    scale_y = H_orig / H_proc
    print(f"Using global scaling for cropping: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")

    uied_boxes = {
        comp['id']: (comp['column_min'], comp['row_min'], comp['width'], comp['height'])
        for comp in uied_data['compos']
    }

    # 2. Create a directory for cropped images
    crop_dir = args.output_html.parent / "cropped_images"
    crop_dir.mkdir(exist_ok=True)
    print(f"Saving cropped images to: {crop_dir.resolve()}")

    # 3. Iterate through mappings and save cropped images to files
    for region_id, region_data in mapping_data.items():
        for placeholder_id, uied_id in region_data['mapping'].items():
            if uied_id not in uied_boxes:
                print(f"Warning: UIED ID {uied_id} from mapping not found. Skipping placeholder {placeholder_id}.")
                continue

            uied_bbox = uied_boxes[uied_id]

            x_proc, y_proc, w_proc, h_proc = uied_bbox
            x_tf = x_proc * scale_x
            y_tf = y_proc * scale_y
            w_tf = w_proc * scale_x
            h_tf = h_proc * scale_y

            x1, y1 = int(x_tf), int(y_tf)
            x2, y2 = int(x_tf + w_tf), int(y_tf + h_tf)

            h_img, w_img, _ = original_image.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            cropped_img = original_image[y1:y2, x1:x2]

            if cropped_img.size == 0:
                print(f"Warning: Cropped image for {placeholder_id} is empty. Skipping.")
                continue

            output_path = crop_dir / f"{placeholder_id}.png"
            cv2.imwrite(str(output_path), cropped_img)

    # --- Phase 2: Use BeautifulSoup to Replace Placeholders by Order ---

    print("\nStarting offline HTML processing with BeautifulSoup...")
    html_content = args.gray_html.read_text()
    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. Find all placeholder elements by their class, in document order.
    all_potential_placeholders = soup.select(
        ".bg-gray-200, .bg-gray-300, .bg-gray-400, .bg-gray-500, .bg-gray-600, img")
    placeholder_elements = []

    for el in all_potential_placeholders:
        if el.name and el.name.lower() == 'svg':
            continue
        if el.name and el.name.lower() != 'img' and el.get_text(strip=True):
            continue
        if el.name and el.name.lower() == 'img':
            src = el.get('src', '')
            # Skip real images (already have http/https/data URLs)
            if src.startswith('http://') or src.startswith('https://') or src.startswith('data:'):
                continue
        placeholder_elements.append(el)

    # 2. Get the placeholder IDs from the mapping file in the correct, sorted order.
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    ordered_placeholder_ids = []
    # Sort region IDs numerically to process them in order
    for region_id in sorted(mapping_data.keys(), key=int):
        region_mapping = mapping_data[region_id]['mapping']
        # Sort the placeholder IDs within each region naturally (e.g., ph1, ph2, ph10)
        sorted_ph_ids = sorted(region_mapping.keys(), key=natural_sort_key)
        ordered_placeholder_ids.extend(sorted_ph_ids)

    # 3. Check for count mismatches
    if len(placeholder_elements) != len(ordered_placeholder_ids):
        print(
            f"Warning: Mismatch in counts! Found {len(placeholder_elements)} gray boxes in HTML, but {len(ordered_placeholder_ids)} mappings.")
    else:
        print(f"Found {len(placeholder_elements)} gray boxes to replace.")

    # 4. Iterate through both lists, create a proper <img> tag, and replace the placeholder.
    for i, ph_element in enumerate(placeholder_elements):
        if i >= len(ordered_placeholder_ids):
            print(f"Warning: More gray boxes in HTML than mappings. Stopping at box {i + 1}.")
            break

        ph_id = ordered_placeholder_ids[i]
        relative_img_path = f"{crop_dir.name}/{ph_id}.png"

        # --- Create a new <img> tag and replace the placeholder ---

        # a. Get all classes from the original placeholder to preserve styling.
        original_classes = ph_element.get('class', [])
        if isinstance(original_classes, str):
            original_classes = original_classes.split()
        elif original_classes is None:
            original_classes = []

        gray_classes = {'bg-gray-200', 'bg-gray-300', 'bg-gray-400', 'bg-gray-500', 'bg-gray-600'}
        original_classes = [c for c in original_classes if c not in gray_classes]

        # b. Create the new <img> tag
        img_tag = soup.new_tag("img", src=relative_img_path)
        img_tag['class'] = original_classes

        # c. Replace the placeholder with the new image tag.
        ph_element.replace_with(img_tag)

    # Save the modified HTML
    args.output_html.write_text(str(soup))

    print(f"\nSuccessfully replaced {min(len(placeholder_elements), len(ordered_placeholder_ids))} placeholders.")
    print(f"Final HTML generated at {args.output_html.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replace placeholder divs in an HTML file with cropped images based on UIED mappings.")
    parser.add_argument("--mapping", type=Path, required=False, help="Path to the mapping JSON file from mapping.py.")
    parser.add_argument("--uied", type=Path, required=False, help="Path to the UIED JSON file.")
    parser.add_argument("--original-image", type=Path, required=False, help="Path to the original screenshot image.")
    parser.add_argument("--gray-html", type=Path, required=False,
                        help="Path to the input HTML file with gray placeholders.")
    parser.add_argument("--output-html", type=Path, required=False, help="Path to save the final, modified HTML file.")

    parser.set_defaults(
        mapping=tmp_dir() / f"mapping_full_{PIPELINE_STEM}.json",
        uied=tmp_dir() / "ip" / f"{PIPELINE_STEM}.json",
        original_image=input_dir() / f"{PIPELINE_STEM}.png",
        gray_html=tmp_dir() / f"{PIPELINE_STEM}_layout.html",
        output_html=output_dir() / f"{PIPELINE_STEM}_layout_final.html",
    )

    args = parser.parse_args()
    main(args)
