"""MongoDB updater for writing LLM tags to EEGDash database.

This module provides idempotent tag updates to MongoDB with:
- Conditional updates (only if config/metadata changed)
- Atomic $set operations (don't overwrite unrelated fields)
- Provenance tracking (config_hash, metadata_hash, model, tagged_at)

Target document structure after update:
    {
        "_id": "ds001234",
        // ... existing dataset fields untouched ...
        "tags": {
            "pathology": ["Healthy"],
            "modality": ["Visual"],
            "type": ["Perception"],
            "confidence": {"pathology": 0.9, "modality": 0.85, "type": 0.8}
        },
        "tagger_meta": {
            "config_hash": "abc123...",
            "metadata_hash": "def456...",
            "model": "openai/gpt-4-turbo",
            "tagged_at": "2024-01-19T12:00:00Z"
        }
    }
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

logger = logging.getLogger(__name__)


class MongoDBUpdater:
    """Updates MongoDB with LLM-generated tags.

    Performs idempotent updates that only modify tag-related fields,
    leaving the rest of the dataset document unchanged.

    Usage:
        updater = MongoDBUpdater("mongodb://...", "eegdash", "datasets")

        # Update tags for a dataset
        updater.update_tags(
            dataset_id="ds001234",
            tags={
                "pathology": ["Healthy"],
                "modality": ["Visual"],
                "type": ["Perception"],
                "confidence": {"pathology": 0.9}
            },
            config_hash="abc123",
            metadata_hash="def456",
            model="openai/gpt-4-turbo"
        )

        # Check if update is needed
        needs_update = updater.needs_update("ds001234", "abc123", "def456")
    """

    def __init__(
        self,
        connection_string: str,
        database: str = "eegdash",
        collection: str = "datasets",
    ):
        """Initialize MongoDB connection.

        Args:
            connection_string: MongoDB connection URL
            database: Database name
            collection: Collection name for datasets
        """
        self.connection_string = connection_string
        self.database_name = database
        self.collection_name = collection
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._collection: Optional[Collection] = None

    def connect(self) -> None:
        """Establish connection to MongoDB."""
        self._client = MongoClient(self.connection_string)
        self._db = self._client[self.database_name]
        self._collection = self._db[self.collection_name]
        logger.info(f"Connected to MongoDB: {self.database_name}.{self.collection_name}")

    def close(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._collection = None

    def needs_update(
        self,
        dataset_id: str,
        config_hash: str,
        metadata_hash: str,
    ) -> bool:
        """Check if a dataset needs tag update.

        Returns True if:
        - Dataset has no tags yet
        - config_hash differs (prompt/examples changed)
        - metadata_hash differs (dataset content changed)

        Args:
            dataset_id: Dataset identifier
            config_hash: Current config hash (prompt + few_shot_examples)
            metadata_hash: Current metadata hash

        Returns:
            True if tags should be updated
        """
        if not self._collection:
            raise RuntimeError("Not connected to MongoDB")

        doc = self._collection.find_one(
            {"dataset_id": dataset_id},
            {"tagger_meta": 1}
        )

        if not doc:
            logger.debug(f"Dataset {dataset_id} not found in MongoDB")
            return True  # No document = needs update (will be a new insert)

        tagger_meta = doc.get("tagger_meta")
        if not tagger_meta:
            logger.debug(f"Dataset {dataset_id} has no tagger_meta, needs update")
            return True

        existing_config = tagger_meta.get("config_hash")
        existing_metadata = tagger_meta.get("metadata_hash")

        if existing_config != config_hash:
            logger.debug(f"Dataset {dataset_id} config_hash changed, needs update")
            return True

        if existing_metadata != metadata_hash:
            logger.debug(f"Dataset {dataset_id} metadata_hash changed, needs update")
            return True

        logger.debug(f"Dataset {dataset_id} tags are current, no update needed")
        return False

    def update_tags(
        self,
        dataset_id: str,
        tags: dict[str, Any],
        config_hash: str,
        metadata_hash: str,
        model: str,
        reasoning: Optional[dict[str, str]] = None,
    ) -> bool:
        """Update tags for a dataset.

        Performs an atomic $set operation that only modifies tag-related fields.
        Uses upsert=False to avoid creating documents (dataset should exist).

        Args:
            dataset_id: Dataset identifier
            tags: Tag dictionary with pathology, modality, type, confidence
            config_hash: Config hash for provenance
            metadata_hash: Metadata hash for provenance
            model: Model used for tagging
            reasoning: Optional reasoning from LLM

        Returns:
            True if document was updated, False if not found
        """
        if not self._collection:
            raise RuntimeError("Not connected to MongoDB")

        now = datetime.now(timezone.utc)

        # Build the update document
        update = {
            "$set": {
                "tags": {
                    "pathology": tags.get("pathology", []),
                    "modality": tags.get("modality", []),
                    "type": tags.get("type", []),
                    "confidence": tags.get("confidence", {}),
                },
                "tagger_meta": {
                    "config_hash": config_hash,
                    "metadata_hash": metadata_hash,
                    "model": model,
                    "tagged_at": now.isoformat(),
                },
            }
        }

        # Add reasoning if provided
        if reasoning:
            update["$set"]["tags"]["reasoning"] = reasoning

        # Update only if dataset exists (don't create new documents)
        result = self._collection.update_one(
            {"dataset_id": dataset_id},
            update,
            upsert=False
        )

        if result.matched_count > 0:
            logger.info(f"Updated tags for {dataset_id}")
            return True
        else:
            logger.warning(f"Dataset {dataset_id} not found in MongoDB, cannot update tags")
            return False

    def update_tags_conditional(
        self,
        dataset_id: str,
        tags: dict[str, Any],
        config_hash: str,
        metadata_hash: str,
        model: str,
        reasoning: Optional[dict[str, str]] = None,
    ) -> bool:
        """Update tags only if config or metadata hash differs.

        This is an atomic conditional update that avoids race conditions.

        Args:
            dataset_id: Dataset identifier
            tags: Tag dictionary
            config_hash: Current config hash
            metadata_hash: Current metadata hash
            model: Model used
            reasoning: Optional reasoning

        Returns:
            True if updated, False if no update needed or not found
        """
        if not self._collection:
            raise RuntimeError("Not connected to MongoDB")

        now = datetime.now(timezone.utc)

        # Filter: update only if hashes differ or tagger_meta doesn't exist
        filter_query = {
            "dataset_id": dataset_id,
            "$or": [
                {"tagger_meta": {"$exists": False}},
                {"tagger_meta.config_hash": {"$ne": config_hash}},
                {"tagger_meta.metadata_hash": {"$ne": metadata_hash}},
            ]
        }

        update = {
            "$set": {
                "tags": {
                    "pathology": tags.get("pathology", []),
                    "modality": tags.get("modality", []),
                    "type": tags.get("type", []),
                    "confidence": tags.get("confidence", {}),
                },
                "tagger_meta": {
                    "config_hash": config_hash,
                    "metadata_hash": metadata_hash,
                    "model": model,
                    "tagged_at": now.isoformat(),
                },
            }
        }

        if reasoning:
            update["$set"]["tags"]["reasoning"] = reasoning

        result = self._collection.update_one(filter_query, update)

        if result.modified_count > 0:
            logger.info(f"Conditionally updated tags for {dataset_id}")
            return True
        elif result.matched_count > 0:
            logger.debug(f"Dataset {dataset_id} tags already current")
            return False
        else:
            logger.warning(f"Dataset {dataset_id} not found or no update needed")
            return False

    def get_tags(self, dataset_id: str) -> Optional[dict[str, Any]]:
        """Get current tags for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Tags dict or None if not found/no tags
        """
        if not self._collection:
            raise RuntimeError("Not connected to MongoDB")

        doc = self._collection.find_one(
            {"dataset_id": dataset_id},
            {"tags": 1, "tagger_meta": 1}
        )

        if not doc:
            return None

        return {
            "tags": doc.get("tags"),
            "tagger_meta": doc.get("tagger_meta"),
        }

    def get_untagged_datasets(
        self,
        config_hash: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Find datasets that need tagging.

        Returns datasets where:
        - No tagger_meta exists, OR
        - config_hash differs from current

        Args:
            config_hash: Current config hash to compare against
            limit: Maximum number of results

        Returns:
            List of dataset documents (dataset_id, source_url fields)
        """
        if not self._collection:
            raise RuntimeError("Not connected to MongoDB")

        cursor = self._collection.find(
            {
                "$or": [
                    {"tagger_meta": {"$exists": False}},
                    {"tagger_meta.config_hash": {"$ne": config_hash}},
                ]
            },
            {"dataset_id": 1, "source_url": 1, "github_url": 1}
        ).limit(limit)

        return list(cursor)

    def get_stats(self) -> dict[str, Any]:
        """Get tagging statistics.

        Returns:
            Dict with counts of tagged/untagged datasets
        """
        if not self._collection:
            raise RuntimeError("Not connected to MongoDB")

        total = self._collection.count_documents({})
        tagged = self._collection.count_documents({"tags": {"$exists": True}})
        untagged = total - tagged

        return {
            "total_datasets": total,
            "tagged": tagged,
            "untagged": untagged,
        }
