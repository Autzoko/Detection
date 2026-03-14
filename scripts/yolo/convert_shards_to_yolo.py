"""
Convert WebDataset-style shard tar files to YOLO detection format.

Shard format: .tar files containing paired {id}.png + {id}.json files
JSON format: {"has_lesion": 0/1, "bboxes": [[x_min, y_min, x_max, y_max], ...],
              "image_width": W, "image_height": H, "dataset": "...", "sample_id": "...",
              "slice_id": "...", "slice_idx": N}

YOLO format:
  dataset/
  ├── images/train/  (PNG files)
  ├── images/val/
  ├── labels/train/  (TXT files: class x_center y_center width height, normalized)
  └── labels/val/

Usage:
    python convert_shards_to_yolo.py \
        --shard_dir /Volumes/Lang/Research/Data/3D\ Ultrasound/Shards \
        --output_dir /Volumes/Lang/Research/Data/3D\ Ultrasound/yolo_dataset \
        --datasets BIrads Class2 Class3 Class4 Abus \
        --val_fraction 0.15 \
        --neg_ratio 0.3 \
        --seed 42
"""

import argparse
import json
import os
import random
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert shard tar files to YOLO format")
    parser.add_argument("--shard_dir", type=str, required=True,
                        help="Root directory containing dataset subdirs with shard tars")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for YOLO dataset")
    parser.add_argument("--datasets", nargs="+",
                        default=["BIrads", "Class2", "Class3", "Class4", "Abus"],
                        help="Dataset subdirectories to include")
    parser.add_argument("--val_fraction", type=float, default=0.15,
                        help="Fraction of volumes to use for validation")
    parser.add_argument("--neg_ratio", type=float, default=0.3,
                        help="Ratio of negative slices to keep (relative to positive count). "
                             "Set to 0 to exclude all negatives, -1 to keep all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_neg_per_volume", type=int, default=50,
                        help="Max negative slices per volume to prevent domination by large volumes")
    parser.add_argument("--include_duying_neg", action="store_true",
                        help="Include Duying (unlabeled) as additional negative samples")
    parser.add_argument("--duying_max", type=int, default=5000,
                        help="Max Duying negative samples to include")
    return parser.parse_args()


def xyxy_to_yolo(bbox, img_w, img_h):
    """Convert [x_min, y_min, x_max, y_max] to YOLO [x_center, y_center, w, h] normalized."""
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return x_center, y_center, w, h


def process_shard(tar_path, temp_dir):
    """Extract samples from a single shard tar file.

    Returns list of (png_path, annotation_dict) tuples.
    """
    samples = []
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        json_members = {m.name: m for m in members if m.name.endswith(".json")}
        png_members = {m.name: m for m in members if m.name.endswith(".png")}

        for json_name, json_member in json_members.items():
            base = json_name.replace(".json", "")
            png_name = base + ".png"
            if png_name not in png_members:
                continue

            # Extract JSON annotation
            f = tar.extractfile(json_member)
            if f is None:
                continue
            annotation = json.loads(f.read().decode("utf-8"))
            f.close()

            # Extract PNG to temp dir
            tar.extract(png_members[png_name], path=temp_dir)
            png_path = os.path.join(temp_dir, png_name)

            samples.append((png_path, annotation))
    return samples


def main():
    args = parse_args()
    random.seed(args.seed)

    shard_dir = Path(args.shard_dir)
    output_dir = Path(args.output_dir)

    # Create output directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Phase 1: Scan all shards to collect volume IDs per dataset for train/val split
    print("=== Phase 1: Scanning shards to collect volume IDs ===")
    volume_samples = defaultdict(list)  # (dataset, sample_id) -> list of (shard_path, json_name)

    for dataset_name in args.datasets:
        dataset_dir = shard_dir / dataset_name
        if not dataset_dir.exists():
            print(f"  WARNING: {dataset_dir} does not exist, skipping")
            continue

        shard_files = sorted(dataset_dir.glob("shard-*.tar"))
        print(f"  {dataset_name}: {len(shard_files)} shards")

        for shard_path in shard_files:
            with tarfile.open(shard_path, "r") as tar:
                for member in tar.getmembers():
                    if not member.name.endswith(".json"):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    ann = json.loads(f.read().decode("utf-8"))
                    f.close()
                    key = (ann["dataset"], ann["sample_id"])
                    volume_samples[key].append({
                        "shard_path": str(shard_path),
                        "base_name": member.name.replace(".json", ""),
                        "has_lesion": ann["has_lesion"],
                    })

    # Split volumes into train/val
    all_volumes = list(volume_samples.keys())
    random.shuffle(all_volumes)

    # Group by dataset for stratified split
    dataset_volumes = defaultdict(list)
    for vol_key in all_volumes:
        dataset_volumes[vol_key[0]].append(vol_key)

    train_volumes = set()
    val_volumes = set()
    for dataset_name, vols in dataset_volumes.items():
        n_val = max(1, int(len(vols) * args.val_fraction))
        val_vols = set(vols[:n_val])
        train_vols = set(vols[n_val:])
        val_volumes.update(val_vols)
        train_volumes.update(train_vols)
        print(f"  {dataset_name}: {len(train_vols)} train / {len(val_vols)} val volumes")

    print(f"\n  Total: {len(train_volumes)} train / {len(val_volumes)} val volumes")

    # Phase 2: Determine which slices to include (handle class imbalance)
    print("\n=== Phase 2: Selecting slices (handling class imbalance) ===")

    # Count positives per split
    train_pos = sum(1 for vol in train_volumes
                    for s in volume_samples[vol] if s["has_lesion"])
    val_pos = sum(1 for vol in val_volumes
                  for s in volume_samples[vol] if s["has_lesion"])
    train_neg = sum(1 for vol in train_volumes
                    for s in volume_samples[vol] if not s["has_lesion"])
    val_neg = sum(1 for vol in val_volumes
                  for s in volume_samples[vol] if not s["has_lesion"])

    print(f"  Train: {train_pos} positive, {train_neg} negative slices")
    print(f"  Val:   {val_pos} positive, {val_neg} negative slices")

    # Determine negative sampling
    if args.neg_ratio < 0:
        max_neg_train = train_neg
        max_neg_val = val_neg
    elif args.neg_ratio == 0:
        max_neg_train = 0
        max_neg_val = 0
    else:
        max_neg_train = int(train_pos * args.neg_ratio)
        max_neg_val = int(val_pos * args.neg_ratio)

    print(f"  Keeping up to {max_neg_train} train / {max_neg_val} val negative slices")

    # Phase 3: Extract and convert
    print("\n=== Phase 3: Extracting and converting ===")

    stats = {"train": {"pos": 0, "neg": 0, "bboxes": 0},
             "val": {"pos": 0, "neg": 0, "bboxes": 0}}

    # Process per-volume to control negative sampling
    for split_name, split_volumes, max_neg in [
        ("train", train_volumes, max_neg_train),
        ("val", val_volumes, max_neg_val),
    ]:
        neg_budget = max_neg
        # Group volumes by shard for efficient extraction
        shard_to_samples = defaultdict(list)
        for vol_key in split_volumes:
            for sample_info in volume_samples[vol_key]:
                shard_to_samples[sample_info["shard_path"]].append({
                    **sample_info,
                    "vol_key": vol_key,
                })

        # Collect negatives first to subsample
        all_neg_entries = []
        for shard_path, entries in shard_to_samples.items():
            for entry in entries:
                if not entry["has_lesion"]:
                    all_neg_entries.append((shard_path, entry))

        # Subsample negatives
        random.shuffle(all_neg_entries)
        selected_neg = set()
        if neg_budget > 0:
            for shard_path, entry in all_neg_entries[:neg_budget]:
                selected_neg.add((shard_path, entry["base_name"]))

        # Now extract from each shard
        processed_shards = 0
        total_shards = len(shard_to_samples)
        for shard_path, entries in shard_to_samples.items():
            processed_shards += 1
            if processed_shards % 10 == 0:
                print(f"  [{split_name}] Processing shard {processed_shards}/{total_shards}")

            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    with tarfile.open(shard_path, "r") as tar:
                        for entry in entries:
                            base = entry["base_name"]
                            is_positive = entry["has_lesion"]

                            # Skip negatives not in selected set
                            if not is_positive and (shard_path, base) not in selected_neg:
                                continue

                            json_name = base + ".json"
                            png_name = base + ".png"

                            # Extract files
                            try:
                                tar.extract(json_name, path=temp_dir)
                                tar.extract(png_name, path=temp_dir)
                            except KeyError:
                                continue

                            # Read annotation
                            with open(os.path.join(temp_dir, json_name)) as f:
                                ann = json.load(f)

                            # Create unique filename
                            safe_id = ann["sample_id"].replace("/", "_").replace(" ", "_")
                            fname = f"{ann['dataset']}_{safe_id}_{ann['slice_id']}"

                            # Copy image
                            src_png = os.path.join(temp_dir, png_name)
                            dst_png = output_dir / "images" / split_name / f"{fname}.png"
                            shutil.copy2(src_png, dst_png)

                            # Create YOLO label file
                            dst_label = output_dir / "labels" / split_name / f"{fname}.txt"
                            img_w = ann["image_width"]
                            img_h = ann["image_height"]

                            if ann["bboxes"]:
                                lines = []
                                for bbox in ann["bboxes"]:
                                    xc, yc, w, h = xyxy_to_yolo(bbox, img_w, img_h)
                                    lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                                dst_label.write_text("\n".join(lines) + "\n")
                                stats[split_name]["pos"] += 1
                                stats[split_name]["bboxes"] += len(ann["bboxes"])
                            else:
                                # Empty label file for negative samples
                                dst_label.write_text("")
                                stats[split_name]["neg"] += 1

                            # Clean up extracted files
                            os.remove(src_png)
                            os.remove(os.path.join(temp_dir, json_name))

                except Exception as e:
                    print(f"  ERROR processing {shard_path}: {e}")

    # Print summary
    print("\n=== Conversion Summary ===")
    for split_name in ["train", "val"]:
        s = stats[split_name]
        total = s["pos"] + s["neg"]
        print(f"  {split_name}: {total} images ({s['pos']} positive, {s['neg']} negative), "
              f"{s['bboxes']} total bboxes")

    # Save dataset info
    info = {
        "datasets": args.datasets,
        "val_fraction": args.val_fraction,
        "neg_ratio": args.neg_ratio,
        "seed": args.seed,
        "stats": stats,
        "train_volumes": len(train_volumes),
        "val_volumes": len(val_volumes),
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n  Dataset info saved to {output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
