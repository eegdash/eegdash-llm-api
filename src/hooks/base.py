"""Base class for dataset hooks.

A hook takes a dataset as input, runs custom logic (computation, LLM call,
etc.), and writes any fields back to the EEGDash MongoDB database.

Usage:
    from src.hooks.base import DatasetHook

    class MyHook(DatasetHook):
        name = "my-hook"

        def process(self, dataset_id: str, dataset: dict) -> dict:
            # Your logic here
            return {"my_field": "my_value"}

    if __name__ == "__main__":
        MyHook.run()

Then run:
    uv run python my_hook.py ds004362
    uv run python my_hook.py ds004362 --dry-run --verbose
"""

import argparse
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.services.mongodb_http_updater import MongoDBHttpUpdater

logger = logging.getLogger(__name__)


class DatasetHook(ABC):
    """Base class for dataset processing hooks.

    Subclass this and implement ``process()`` to define your own hook.
    Call ``MyHook.run()`` to execute it with full CLI and DB support.
    """

    name: str = "unnamed-hook"

    @abstractmethod
    def process(self, dataset_id: str, dataset: dict[str, Any]) -> dict[str, Any]:
        """Process a dataset and return fields to write to MongoDB.

        Args:
            dataset_id: The dataset identifier (e.g. "ds004362").
            dataset: The full dataset document from MongoDB.

        Returns:
            A dict of fields to $set on the dataset document.
            Return an empty dict to skip the write.
        """
        ...

    @classmethod
    def run(cls) -> None:
        """CLI entry point. Handles args, DB connection, and error handling."""
        load_dotenv()

        parser = argparse.ArgumentParser(
            description=f"Run the '{cls.name}' dataset hook"
        )
        parser.add_argument(
            "dataset_id",
            help="Dataset identifier (e.g. ds004362)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Fetch dataset and run process(), but do not write to DB",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable debug logging",
        )
        parser.add_argument(
            "--database",
            default=os.environ.get("MONGODB_DATABASE", "eegdash"),
            help="Database name (default: eegdash)",
        )

        args = parser.parse_args()

        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        api_url = os.environ.get("EEGDASH_API_URL", "https://data.eegdash.org")
        admin_token = os.environ.get("EEGDASH_ADMIN_TOKEN", "")

        if not admin_token:
            logger.error("EEGDASH_ADMIN_TOKEN environment variable not set")
            sys.exit(1)

        logger.info("Hook:       %s", cls.name)
        logger.info("API URL:    %s", api_url)
        logger.info("Database:   %s", args.database)
        logger.info("Dataset ID: %s", args.dataset_id)

        updater = MongoDBHttpUpdater(
            api_url=api_url,
            admin_token=admin_token,
            database=args.database,
        )
        updater.connect()

        try:
            # Fetch dataset
            dataset = updater.get_dataset(args.dataset_id)
            if dataset is None:
                logger.error("Dataset '%s' not found in database", args.dataset_id)
                sys.exit(1)

            logger.info("Dataset '%s' found", args.dataset_id)

            # Run the hook
            hook = cls()
            result = hook.process(args.dataset_id, dataset)

            if not isinstance(result, dict):
                logger.error("process() must return a dict, got %s", type(result).__name__)
                sys.exit(1)

            if not result:
                logger.info("process() returned empty dict — nothing to write")
                return

            logger.info("process() returned fields: %s", list(result.keys()))

            if args.dry_run:
                logger.info("Dry run — skipping write. Would set: %s", result)
                return

            # Write to DB
            success = updater.update_fields(args.dataset_id, result)
            if not success:
                logger.error("Failed to write fields to '%s'", args.dataset_id)
                sys.exit(1)

            logger.info("Successfully wrote fields to '%s'", args.dataset_id)

            # Read back for verification
            updated = updater.get_dataset(args.dataset_id)
            if updated and args.verbose:
                for key in result:
                    logger.debug("  %s = %s", key, updated.get(key))

        except Exception:
            logger.exception("Hook '%s' failed for dataset '%s'", cls.name, args.dataset_id)
            sys.exit(1)
        finally:
            updater.close()
