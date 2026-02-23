#!/usr/bin/env python3
"""
Enqueue all datasets from EEGDash MongoDB for batch tagging.

This script:
1. Fetches all datasets from the EEGDash API
2. Optionally skips datasets that have ground truth
3. Builds metadata for each dataset
4. Enqueues them for processing by the worker

Usage:
    cd eegdash-llm-api
    uv run python scripts/enqueue_all_datasets.py [--skip-ground-truth] [--dry-run] [--limit N]

Options:
    --skip-ground-truth    Skip datasets that have ground truth entries in cache
    --dry-run              Show what would be enqueued without actually enqueueing
    --limit N              Only enqueue first N datasets (for testing)
    --verbose              Show detailed output
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.services.cache import TaggingCache
from src.services.queue import TaggingQueue


# EEGDash API base URL
EEGDASH_API_URL = os.environ.get("EEGDASH_API_URL", "https://data.eegdash.org")


def build_source_url(dataset: dict) -> str | None:
    """
    Build source URL for a dataset.

    Tries multiple sources in order:
    1. external_links.source_url (if present)
    2. github_url (if present)
    3. Construct from source + dataset_id
    """
    # Try external_links first
    external_links = dataset.get("external_links", {})
    if external_links and external_links.get("source_url"):
        return external_links["source_url"]

    # Try github_url
    if dataset.get("github_url"):
        return dataset["github_url"]

    # Construct from source + dataset_id
    source = dataset.get("source", "openneuro")
    dataset_id = dataset.get("dataset_id", "")

    if not dataset_id:
        return None

    if source == "openneuro":
        return f"https://github.com/OpenNeuroDatasets/{dataset_id}"

    # Unknown source, can't construct URL
    return None


def sanitize_text(text: str | None) -> str:
    """Remove null characters and other problematic Unicode from text."""
    if not text:
        return ""
    # Remove null characters that Postgres can't handle
    return text.replace("\u0000", "").replace("\x00", "")


def build_metadata_snapshot(dataset: dict) -> dict:
    """
    Build metadata snapshot from dataset document.

    Extracts relevant fields for tagging and sanitizes text.
    """
    return {
        "title": sanitize_text(dataset.get("title", "")),
        "dataset_description": sanitize_text(dataset.get("description", "")),
        "readme": sanitize_text(dataset.get("readme", "")),
        "paper_abstract": sanitize_text(dataset.get("paper_abstract", "")),
        # Add any other relevant fields
    }


async def fetch_all_datasets(limit: int | None = None) -> list[dict]:
    """
    Fetch all datasets from the EEGDash API.

    Args:
        limit: Optional limit on number of datasets to fetch

    Returns:
        List of dataset documents
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Use pagination - EEGDash API supports limit parameter
        page_size = 1000
        params = {"limit": page_size}

        url = f"{EEGDASH_API_URL}/api/eegdash/datasets"
        response = await client.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        datasets = data.get("data", [])

        print(f"Fetched {len(datasets)} datasets from API")

        if limit:
            datasets = datasets[:limit]
            print(f"Limited to {len(datasets)} datasets")

        return datasets


async def main():
    parser = argparse.ArgumentParser(
        description="Enqueue all datasets for batch tagging"
    )
    parser.add_argument(
        "--skip-ground-truth",
        action="store_true",
        help="Skip datasets that have ground truth entries"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be enqueued without actually enqueueing"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only enqueue first N datasets (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    # Validate environment
    postgres_url = os.environ.get("POSTGRES_URL")
    if not postgres_url:
        print("ERROR: POSTGRES_URL environment variable not set")
        sys.exit(1)

    # Load cache for ground truth checking
    cache_path = Path(os.environ.get("CACHE_DIR", "/tmp/eegdash-llm-api/cache")) / "tagging_cache.json"
    cache = TaggingCache(cache_path=cache_path, config_hash="enqueue_script")
    ground_truth_datasets = set(cache.list_ground_truth_datasets()) if args.skip_ground_truth else set()

    if args.skip_ground_truth:
        print(f"Will skip {len(ground_truth_datasets)} ground truth datasets")

    # Fetch datasets
    print(f"Fetching datasets from {EEGDASH_API_URL}...")
    datasets = await fetch_all_datasets(limit=args.limit)

    # Initialize queue
    queue = None
    if not args.dry_run:
        queue = TaggingQueue(postgres_url)
        await queue.initialize()

    # Track statistics
    total = 0
    enqueued = 0
    skipped_ground_truth = 0
    skipped_no_url = 0
    duplicates = 0

    print("\nProcessing datasets...")

    for dataset in datasets:
        total += 1
        dataset_id = dataset.get("dataset_id", "")

        if not dataset_id:
            skipped_no_url += 1
            continue

        # Check ground truth
        if args.skip_ground_truth and dataset_id in ground_truth_datasets:
            skipped_ground_truth += 1
            if args.verbose:
                print(f"  Skipped {dataset_id}: ground truth exists")
            continue

        # Build source URL
        source_url = build_source_url(dataset)
        if not source_url:
            skipped_no_url += 1
            if args.verbose:
                print(f"  Skipped {dataset_id}: no source URL")
            continue

        # Build metadata snapshot
        metadata = build_metadata_snapshot(dataset)

        if args.verbose:
            print(f"  Enqueueing {dataset_id}: {source_url}")

        if not args.dry_run:
            job_id, is_new = await queue.enqueue(
                dataset_id=dataset_id,
                source_url=source_url,
                metadata_snapshot=metadata,
            )
            if is_new:
                enqueued += 1
            else:
                duplicates += 1
        else:
            enqueued += 1

    # Cleanup
    if queue:
        await queue.close()

    # Summary
    print()
    print("=" * 50)
    print("Batch Enqueue Summary")
    print("=" * 50)
    print(f"Total datasets:         {total}")
    print(f"Enqueued:               {enqueued}")
    print(f"Duplicates:             {duplicates}")
    print(f"Skipped (ground truth): {skipped_ground_truth}")
    print(f"Skipped (no URL):       {skipped_no_url}")
    print()

    if args.dry_run:
        print("DRY RUN - no jobs were actually enqueued")
    else:
        # Show queue stats
        queue = TaggingQueue(postgres_url)
        await queue.initialize()
        stats = await queue.get_stats()
        await queue.close()

        print("Current Queue Status:")
        print(f"  Pending:    {stats['pending']}")
        print(f"  Processing: {stats['processing']}")
        print(f"  Completed:  {stats['completed']}")
        print(f"  Failed:     {stats['failed']}")


if __name__ == "__main__":
    asyncio.run(main())
