#!/usr/bin/env python3
"""
build_competition_corpus.py

Downloads and prepares IAM + Bentham + IAMRIMES line-level datasets from Hugging Face,
resizes images to a fixed height, and writes train/val/test CSVs and PNGs.

Usage example:
  python build_competition_corpus.py --out_dir handwriting_competition \
    --iam_max 10000 --bentham_max 8000 --iamrimes_max 12000 --height 64 --seed 42
"""

import os
import csv
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# Helper: try to extract text field from dataset item
def extract_text_from_item(item):
    for key in ("text", "transcription", "transcript", "label", "gt", "transcriptions"):
        if key in item and item[key] is not None:
            return item[key]
    # some datasets use 'line' or 'annotation' naming; fallback to any str field
    for k, v in item.items():
        if isinstance(v, str) and len(v) > 0:
            return v
    return None

def save_image_from_pil(pil_img, out_path, height):
    # pil_img is grayscale or rgb, resize preserving aspect ratio to `height`
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")
    w, h = pil_img.size
    new_h = height
    new_w = max(1, int(w * (new_h / h)))
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    pil_img.save(out_path, format="PNG")

def iterate_dataset_and_collect(ds_id, split, max_lines, out_dir_images, height, sample_by_writer=False):
    """
    Loads HF dataset split and returns list of (image_path, text) saved to out_dir_images.
    """
    print(f"> Loading {ds_id} split={split} ... (may download files)")
    try:
        ds = load_dataset(ds_id, split=split)
    except Exception as e:
        print(f"Failed to load {ds_id} ({split}): {e}")
        return []

    total = len(ds)
    print(f"  dataset reports {total} rows")

    # If sampling by writer is requested but dataset has no writer id, fallback to random
    use_writer = sample_by_writer and ("writer" in ds.column_names or "writer_id" in ds.column_names or "author" in ds.column_names)
    print("  sample_by_writer:", sample_by_writer, " -> using writer metadata:", use_writer)

    indices = list(range(total))
    random.shuffle(indices)

    saved = []
    count = 0
    for idx in tqdm(indices, desc=f"{ds_id} items"):
        if max_lines and count >= max_lines:
            break
        item = ds[int(idx)]
        # obtain PIL image
        if "image" not in item:
            # skip if no image column
            continue
        img = item["image"]
        # text extraction
        txt = extract_text_from_item(item)
        if txt is None:
            continue
        # convert to PIL and save to disk
        try:
            pil = img.convert("L")
        except Exception:
            try:
                pil = Image.open(img).convert("L")
            except Exception as e:
                # skip problematic item
                continue
        # save
        fname = f"{len(saved)+1:08d}.png"
        out_path = os.path.join(out_dir_images, fname)
        save_image_from_pil(pil, out_path, height)
        saved.append((out_path, txt))
        count += 1
    print(f"  saved {count} images from {ds_id}")
    return saved

def main(args):
    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out_images = os.path.join(args.out_dir, "images")
    os.makedirs(out_images, exist_ok=True)

    collected = []

    # Datasets to fetch (HF id, split, max_lines)
    ds_list = [
        ("Teklia/IAM-line", args.iam_split, args.iam_max),
        ("staghado/Bentham", args.bentham_split, args.bentham_max),
        ("Artemis-IA/IAMRIMES", args.iamrimes_split, args.iamrimes_max),
    ]

    for ds_id, split, max_lines in ds_list:
        if max_lines is None or max_lines <= 0:
            print(f"Skipping {ds_id} (max_lines <= 0)")
            continue
        try:
            part = iterate_dataset_and_collect(ds_id, split, max_lines, out_images, args.height, sample_by_writer=args.sample_by_writer)
            collected.extend(part)
        except Exception as e:
            print(f"Error processing {ds_id}: {e}")

    if len(collected) == 0:
        print("No images collected. Exiting.")
        return

    # Shuffle and split
    if args.shuffle:
        random.shuffle(collected)

    n_total = len(collected)
    n_val = int(n_total * args.val_ratio)
    n_test = int(n_total * args.test_ratio)
    n_train = n_total - n_val - n_test
    train = collected[:n_train]
    val = collected[n_train : n_train + n_val]
    test = collected[n_train + n_val :]

    print(f"Total images: {n_total}  train={len(train)} val={len(val)} test={len(test)}")

    # write csvs
    def write_csv(rows, path):
        with open(path, "w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "text"])
            for img_path, txt in rows:
                w.writerow([img_path, txt])

    write_csv(train, os.path.join(args.out_dir, "train.csv"))
    write_csv(val, os.path.join(args.out_dir, "val.csv"))
    write_csv(test, os.path.join(args.out_dir, "test.csv"))

    print("Saved CSVs at:", args.out_dir)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="handwriting_competition", help="output directory")
    parser.add_argument("--height", type=int, default=64, help="target image height in px")
    parser.add_argument("--iam_max", type=int, default=10000, help="max lines to take from IAM (0 to skip)")
    parser.add_argument("--bentham_max", type=int, default=8000, help="max lines to take from Bentham (0 to skip)")
    parser.add_argument("--iamrimes_max", type=int, default=12000, help="max lines to take from IAMRIMES (0 to skip)")
    parser.add_argument("--iam_split", type=str, default="train", help="split name for IAM-line")
    parser.add_argument("--bentham_split", type=str, default="train", help="split name for Bentham")
    parser.add_argument("--iamrimes_split", type=str, default="train", help="split name for IAMRIMES")
    parser.add_argument("--shuffle", action="store_true", help="shuffle combined data")
    parser.add_argument("--val_ratio", type=float, default=0.08, help="validation fraction (default 8%)")
    parser.add_argument("--test_ratio", type=float, default=0.07, help="test fraction (default 7%)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--sample_by_writer", action="store_true", help="try to sample by writer if metadata available")
    args = parser.parse_args()
    main(args)
