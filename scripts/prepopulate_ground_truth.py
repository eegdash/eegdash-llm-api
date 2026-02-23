#!/usr/bin/env python3
"""
Prepopulate cache with ground truth data from CSV.

This script reads the ground truth CSV file and loads valid entries into
the TaggingCache as ground truth (never invalidated by config changes).

Usage:
    cd eegdash-llm-api
    uv run python scripts/prepopulate_ground_truth.py

Options:
    --csv PATH      Path to ground truth CSV (default: auto-detect from tagger repo)
    --cache PATH    Path to cache file (default: /tmp/eegdash-llm-api/cache/tagging_cache.json)
    --dry-run       Show what would be loaded without actually saving
    --verbose       Show detailed output for each dataset
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.cache import TaggingCache


def find_ground_truth_csv() -> Path:
    """Find the ground truth CSV file."""
    # Try common locations
    candidates = [
        Path(__file__).parent.parent.parent / "eegdash-llm-tagger" / "ground-truth-data" / "dataset_summary.csv",
        Path.home() / "neuroscience_work" / "eegdash-llm-tagger" / "ground-truth-data" / "dataset_summary.csv",
        Path("/Users/kuntalkokate/neuroscience_work/eegdash-llm-tagger/ground-truth-data/dataset_summary.csv"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find ground truth CSV. Please specify --csv path.\n"
        f"Searched: {[str(p) for p in candidates]}"
    )


def parse_tags(row: dict) -> dict:
    """
    Parse tags from CSV row.

    Returns dict with pathology, modality, type arrays.
    Returns None if all tags are empty.
    """
    pathology = row.get("Type Subject", "").strip()
    modality = row.get("modality of exp", "").strip()
    exp_type = row.get("type of exp", "").strip()

    # Skip rows with no tags
    if not pathology and not modality and not exp_type:
        return None

    return {
        "pathology": [pathology] if pathology else [],
        "modality": [modality] if modality else [],
        "type": [exp_type] if exp_type else [],
        "confidence": {
            "pathology": 1.0 if pathology else 0.0,
            "modality": 1.0 if modality else 0.0,
            "type": 1.0 if exp_type else 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepopulate cache with ground truth data"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Path to ground truth CSV file"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/tmp/eegdash-llm-api/cache/tagging_cache.json"),
        help="Path to cache file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be loaded without saving"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    # Find CSV file
    csv_path = args.csv or find_ground_truth_csv()
    print(f"Reading ground truth from: {csv_path}")

    # Ensure cache directory exists
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    # Initialize cache (config_hash doesn't matter for ground truth)
    cache = TaggingCache(cache_path=args.cache, config_hash="ground_truth_loader")

    # Track statistics
    total = 0
    loaded = 0
    skipped_empty = 0
    skipped_no_id = 0

    # Read and process CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total += 1

            # Get dataset ID
            dataset_id = row.get("dataset", "").strip()
            if not dataset_id:
                skipped_no_id += 1
                continue

            # Parse tags
            tags = parse_tags(row)
            if tags is None:
                skipped_empty += 1
                if args.verbose:
                    print(f"  Skipped {dataset_id}: no tags")
                continue

            # Load into cache
            if args.verbose:
                print(f"  Loading {dataset_id}: pathology={tags['pathology']}, "
                      f"modality={tags['modality']}, type={tags['type']}")

            if not args.dry_run:
                cache.set_ground_truth(dataset_id, tags)

            loaded += 1

    # Summary
    print()
    print("=" * 50)
    print("Ground Truth Prepopulation Summary")
    print("=" * 50)
    print(f"Total rows:        {total}")
    print(f"Loaded:            {loaded}")
    print(f"Skipped (no tags): {skipped_empty}")
    print(f"Skipped (no ID):   {skipped_no_id}")
    print()

    if args.dry_run:
        print("DRY RUN - no changes saved")
    else:
        print(f"Cache file: {args.cache}")

        # Show final cache stats
        stats = cache.stats()
        print(f"Total cache entries: {stats['total_entries']}")

        gt_datasets = cache.list_ground_truth_datasets()
        print(f"Ground truth datasets: {len(gt_datasets)}")


if __name__ == "__main__":
    main()
