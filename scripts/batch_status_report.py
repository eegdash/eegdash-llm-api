#!/usr/bin/env python3
"""
Generate status report for batch tagging operations.

Shows:
- Queue statistics (pending, processing, completed, failed)
- Success rate
- Cache statistics
- Ground truth coverage

Usage:
    cd eegdash-llm-api
    uv run python scripts/batch_status_report.py [--watch]

Options:
    --watch    Continuously update the report every 30 seconds
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.services.cache import TaggingCache
from src.services.queue import TaggingQueue


async def get_queue_stats(postgres_url: str) -> dict:
    """Get queue statistics."""
    queue = TaggingQueue(postgres_url)
    await queue.initialize()
    stats = await queue.get_stats()
    await queue.close()
    return stats


def get_cache_stats(cache_path: Path) -> dict:
    """Get cache statistics."""
    cache = TaggingCache(cache_path=cache_path, config_hash="report")
    stats = cache.stats()
    gt_datasets = cache.list_ground_truth_datasets()
    return {
        **stats,
        "ground_truth_count": len(gt_datasets),
    }


def print_report(queue_stats: dict, cache_stats: dict):
    """Print the status report."""
    print()
    print("=" * 60)
    print("  EEGDash Batch Tagging Status Report")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Queue Statistics
    print("QUEUE STATUS")
    print("-" * 40)
    print(f"  Pending:      {queue_stats['pending']:>6}")
    print(f"  Processing:   {queue_stats['processing']:>6}")
    print(f"  Completed:    {queue_stats['completed']:>6}")
    print(f"  Failed:       {queue_stats['failed']:>6}")
    print()

    total_done = queue_stats['completed'] + queue_stats['failed']
    if total_done > 0:
        success_rate = queue_stats['completed'] / total_done * 100
        print(f"  Success Rate: {success_rate:>6.1f}%")
    else:
        print(f"  Success Rate:    N/A")

    total_all = total_done + queue_stats['pending'] + queue_stats['processing']
    if total_all > 0:
        progress = total_done / total_all * 100
        print(f"  Progress:     {progress:>6.1f}%")
    print()

    # Cache Statistics
    print("CACHE STATUS")
    print("-" * 40)
    print(f"  Total Entries:    {cache_stats['total_entries']:>6}")
    print(f"  Ground Truth:     {cache_stats['ground_truth_count']:>6}")
    print(f"  Unique Datasets:  {cache_stats['unique_datasets']:>6}")
    print(f"  Config Hash:      {cache_stats['config_hash']}")
    print()

    # Progress Bar
    if total_all > 0:
        bar_width = 40
        filled = int(bar_width * total_done / total_all)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  Progress: [{bar}] {total_done}/{total_all}")
    print()


async def main():
    parser = argparse.ArgumentParser(
        description="Generate batch tagging status report"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously update the report"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Update interval in seconds (default: 30)"
    )

    args = parser.parse_args()

    # Validate environment
    postgres_url = os.environ.get("POSTGRES_URL")
    if not postgres_url:
        print("ERROR: POSTGRES_URL environment variable not set")
        sys.exit(1)

    cache_path = Path(os.environ.get("CACHE_DIR", "/tmp/eegdash-llm-api/cache")) / "tagging_cache.json"

    if args.watch:
        print("Watching batch tagging progress (Ctrl+C to stop)...")
        try:
            while True:
                # Clear screen
                print("\033[2J\033[H", end="")

                queue_stats = await get_queue_stats(postgres_url)
                cache_stats = get_cache_stats(cache_path)
                print_report(queue_stats, cache_stats)

                print(f"(Refreshing every {args.interval} seconds, Ctrl+C to stop)")

                await asyncio.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped watching.")
    else:
        queue_stats = await get_queue_stats(postgres_url)
        cache_stats = get_cache_stats(cache_path)
        print_report(queue_stats, cache_stats)


if __name__ == "__main__":
    asyncio.run(main())
