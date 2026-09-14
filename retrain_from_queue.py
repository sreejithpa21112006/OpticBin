"""
OpticBin — Retrain from Review Queue
======================================
CLI utility to incorporate Gemini-verified samples from review_queue/
back into the training dataset for the next training run.

Usage:
    python retrain_from_queue.py [--dry-run] [--min-confidence 0.7]

Options:
    --dry-run           Print what would happen without copying files
    --min-confidence    Minimum Gemini confidence to accept a pseudo-label (default: 0.70)
    --agreements-only   Only include samples where ML and Gemini agreed
    --split             Target dataset split: train / val (default: train)
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

REVIEW_QUEUE_DIR = Path("review_queue")
DATASET_DIR = Path("dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move verified samples from review_queue into the training dataset."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying any files.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
        metavar="FLOAT",
        help="Minimum Gemini confidence to accept a pseudo-label (0.0–1.0). Default: 0.70",
    )
    parser.add_argument(
        "--agreements-only",
        action="store_true",
        help="Only include samples where the ML model and Gemini Vision agreed.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Target dataset split folder. Default: train",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not REVIEW_QUEUE_DIR.exists():
        print(f"[!] Review queue directory '{REVIEW_QUEUE_DIR}' does not exist.")
        print("    Run the OpticBin app and classify some uncertain items first.")
        return

    queue_items = list(REVIEW_QUEUE_DIR.glob("*/metadata.json"))
    if not queue_items:
        print("[!] No items found in the review queue.")
        return

    print(f"\nOpticBin - Retrain from Review Queue")
    print(f"{'=' * 50}")
    print(f"Queue directory : {REVIEW_QUEUE_DIR.resolve()}")
    print(f"Dataset target  : {DATASET_DIR}/{args.split}/")
    print(f"Min confidence  : {args.min_confidence:.2f}")
    print(f"Agreements only : {args.agreements_only}")
    print(f"Dry run         : {args.dry_run}")
    print(f"{'=' * 50}\n")

    counters: dict[str, int] = defaultdict(int)
    skipped = 0
    copied = 0
    errors = 0

    for meta_path in sorted(queue_items):
        folder = meta_path.parent
        image_path = folder / "image.jpg"

        if not image_path.exists():
            print(f"  [SKIP] {folder.name} - image.jpg missing")
            skipped += 1
            continue

        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            print(f"  [ERROR] {folder.name} - failed to read metadata: {exc}")
            errors += 1
            continue

        # Determine the best label to use
        # Priority: human-verified label > Gemini label (if confident enough)
        human_label = meta.get("verified_label")
        gemini_label = meta.get("gemini_label")
        gemini_conf = float(meta.get("gemini_confidence") or 0.0)
        agreement = meta.get("agreement", True)

        if human_label:
            final_label = human_label
            source = "human-verified"
        elif gemini_label and gemini_conf >= args.min_confidence:
            if args.agreements_only and not agreement:
                print(f"  [SKIP] {folder.name} - disagreement excluded (--agreements-only)")
                skipped += 1
                continue
            final_label = gemini_label
            source = f"gemini ({gemini_conf:.2f})"
        else:
            print(f"  [SKIP] {folder.name} - confidence too low ({gemini_conf:.2f} < {args.min_confidence:.2f})")
            skipped += 1
            continue

        # Destination: dataset/<split>/<label>/<folder_name>.jpg
        dest_dir = DATASET_DIR / args.split / final_label
        dest_file = dest_dir / f"{folder.name}.jpg"

        agreement_marker = "AGREE" if agreement else "DISAGREE"
        print(f"  [{agreement_marker}] {folder.name}")
        print(f"           ML={meta.get('ml_label')}({meta.get('ml_confidence', 0):.2f}) "
              f"-> label={final_label} [{source}]")

        if not args.dry_run:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, dest_file)
                # Mark as incorporated in metadata
                meta["incorporated"] = True
                meta["incorporated_split"] = args.split
                meta["incorporated_label"] = final_label
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                copied += 1
                counters[final_label] += 1
            except Exception as exc:
                print(f"  [ERROR] Failed to copy: {exc}")
                errors += 1
        else:
            copied += 1
            counters[final_label] += 1

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Summary {'(DRY RUN - no files changed)' if args.dry_run else ''}")
    print(f"  Copied  : {copied}")
    print(f"  Skipped : {skipped}")
    print(f"  Errors  : {errors}")
    print(f"\n  Samples added per class:")
    for label in sorted(counters):
        bar = "#" * counters[label]
        print(f"    {label:<12} {counters[label]:>4}  {bar}")

    if not args.dry_run and copied > 0:
        print(f"\n[OK] {copied} samples added to {DATASET_DIR}/{args.split}/")
        print("   Run  python train.py  to retrain the model with the new data.")
    elif args.dry_run and copied > 0:
        print(f"\n   (Dry run - rerun without --dry-run to apply changes)")


if __name__ == "__main__":
    main()
