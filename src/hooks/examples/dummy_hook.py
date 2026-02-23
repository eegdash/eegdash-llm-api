#!/usr/bin/env python3
"""
Dummy hook that writes a test field to a dataset.

Use this to verify that the hook framework and DB connection are working.
The written field is clearly identifiable as test data.

Usage:
    cd eegdash-llm-api
    uv run python src/hooks/examples/dummy_hook.py ds004362 --dry-run --verbose
    uv run python src/hooks/examples/dummy_hook.py ds004362 --verbose
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.hooks.base import DatasetHook


class DummyHook(DatasetHook):
    name = "dummy"

    def process(self, dataset_id: str, dataset: dict[str, Any]) -> dict[str, Any]:
        return {
            "dummy_test": {
                "message": "Hello from DummyHook",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "dataset_title": dataset.get("title", "unknown"),
            }
        }


if __name__ == "__main__":
    DummyHook.run()
