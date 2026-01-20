"""HTTP-based MongoDB updater for writing LLM tags via EEGDash API.

This module uses the EEGDash API (https://data.eegdash.org) instead of
direct MongoDB connections. This is safer and works without exposing
MongoDB ports.

Safety: Always checks if dataset exists before updating to prevent
accidental creation of new documents.

API Endpoints used:
- GET  /api/{db}/datasets/{dataset_id}  - Check if dataset exists
- POST /admin/{db}/datasets             - Update dataset (uses $set)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Default API base URL
DEFAULT_API_URL = "https://data.eegdash.org"


class MongoDBHttpUpdater:
    """Updates MongoDB with LLM-generated tags via HTTP API.

    Uses the EEGDash API instead of direct MongoDB connection.
    Performs safe updates that only modify existing datasets.

    Usage:
        updater = MongoDBHttpUpdater(
            api_url="https://data.eegdash.org",
            admin_token="your-admin-token",
            database="eegdash"
        )

        # Update tags for a dataset
        success = updater.update_tags(
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
    """

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        admin_token: str = "",
        database: str = "eegdash",
        timeout: float = 30.0,
    ):
        """Initialize HTTP updater.

        Args:
            api_url: Base URL of the EEGDash API
            admin_token: Admin bearer token for write operations
            database: Database name (default: eegdash)
            timeout: HTTP request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.admin_token = admin_token
        self.database = database
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    def connect(self) -> None:
        """Initialize HTTP client."""
        self._client = httpx.Client(
            timeout=self.timeout,
            headers={
                "User-Agent": "EEGDash-LLM-Tagger/1.0",
                "Accept": "application/json",
            }
        )
        logger.info(f"HTTP updater initialized for {self.api_url}")

    def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def _get_headers(self, with_auth: bool = False) -> dict[str, str]:
        """Get headers for requests."""
        headers = {"Content-Type": "application/json"}
        if with_auth and self.admin_token:
            headers["Authorization"] = f"Bearer {self.admin_token}"
        return headers

    def dataset_exists(self, dataset_id: str) -> bool:
        """Check if a dataset exists in MongoDB.

        Args:
            dataset_id: Dataset identifier

        Returns:
            True if dataset exists, False otherwise
        """
        if not self._client:
            raise RuntimeError("Not connected - call connect() first")

        url = f"{self.api_url}/api/{self.database}/datasets/{dataset_id}"

        try:
            response = self._client.get(url, headers=self._get_headers())

            if response.status_code == 200:
                data = response.json()
                return data.get("success", False) and data.get("data") is not None
            elif response.status_code == 404:
                return False
            else:
                logger.warning(f"Unexpected status {response.status_code} checking dataset {dataset_id}")
                return False

        except httpx.RequestError as e:
            logger.error(f"HTTP error checking dataset {dataset_id}: {e}")
            return False

    def get_dataset(self, dataset_id: str) -> Optional[dict[str, Any]]:
        """Get a dataset from MongoDB.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dataset document or None if not found
        """
        if not self._client:
            raise RuntimeError("Not connected - call connect() first")

        url = f"{self.api_url}/api/{self.database}/datasets/{dataset_id}"

        try:
            response = self._client.get(url, headers=self._get_headers())

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("data")
            return None

        except httpx.RequestError as e:
            logger.error(f"HTTP error getting dataset {dataset_id}: {e}")
            return None

    def needs_update(
        self,
        dataset_id: str,
        config_hash: str,
        metadata_hash: str,
    ) -> bool:
        """Check if a dataset needs tag update.

        Returns True if:
        - Dataset exists but has no tags yet
        - config_hash differs (prompt/examples changed)
        - metadata_hash differs (dataset content changed)

        Returns False if:
        - Dataset doesn't exist (can't update non-existent dataset)
        - Hashes match (already up to date)

        Args:
            dataset_id: Dataset identifier
            config_hash: Current config hash
            metadata_hash: Current metadata hash

        Returns:
            True if tags should be updated
        """
        dataset = self.get_dataset(dataset_id)

        if not dataset:
            logger.debug(f"Dataset {dataset_id} not found - cannot update")
            return False  # Can't update non-existent dataset

        tagger_meta = dataset.get("tagger_meta")
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

        SAFETY: First checks if dataset exists. Will NOT create new documents.

        Args:
            dataset_id: Dataset identifier
            tags: Tag dictionary with pathology, modality, type, confidence
            config_hash: Config hash for provenance
            metadata_hash: Metadata hash for provenance
            model: Model used for tagging
            reasoning: Optional reasoning from LLM

        Returns:
            True if document was updated, False if not found or error
        """
        if not self._client:
            raise RuntimeError("Not connected - call connect() first")

        # SAFETY CHECK: Verify dataset exists before updating
        if not self.dataset_exists(dataset_id):
            logger.warning(f"Dataset {dataset_id} not found in MongoDB - skipping update")
            return False

        now = datetime.now(timezone.utc)

        # Build the update payload
        # The API uses $set internally, so we only send the fields to update
        update_payload = {
            "dataset_id": dataset_id,
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

        # Add reasoning if provided
        if reasoning:
            update_payload["tags"]["reasoning"] = reasoning

        url = f"{self.api_url}/admin/{self.database}/datasets"

        try:
            response = self._client.post(
                url,
                json=update_payload,
                headers=self._get_headers(with_auth=True),
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("updated_count", 0) > 0 or "updated" in data.get("message", "").lower():
                    logger.info(f"Updated tags for {dataset_id}")
                    return True
                else:
                    logger.info(f"Tags unchanged for {dataset_id}")
                    return True  # Still successful, just no changes
            else:
                logger.error(f"Failed to update {dataset_id}: {response.status_code} - {response.text}")
                return False

        except httpx.RequestError as e:
            logger.error(f"HTTP error updating dataset {dataset_id}: {e}")
            return False

    def get_tags(self, dataset_id: str) -> Optional[dict[str, Any]]:
        """Get current tags for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Tags dict or None if not found/no tags
        """
        dataset = self.get_dataset(dataset_id)

        if not dataset:
            return None

        return {
            "tags": dataset.get("tags"),
            "tagger_meta": dataset.get("tagger_meta"),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get tagging statistics.

        Note: This is an approximation using the datasets summary endpoint.

        Returns:
            Dict with counts of tagged/untagged datasets
        """
        if not self._client:
            raise RuntimeError("Not connected - call connect() first")

        url = f"{self.api_url}/api/{self.database}/datasets/summary"

        try:
            response = self._client.get(url, headers=self._get_headers())

            if response.status_code == 200:
                data = response.json()
                totals = data.get("totals", {})
                return {
                    "total_datasets": totals.get("datasets", 0),
                    "tagged": "unknown",  # API doesn't expose this directly
                    "untagged": "unknown",
                }
            return {"total_datasets": 0, "tagged": "unknown", "untagged": "unknown"}

        except httpx.RequestError as e:
            logger.error(f"HTTP error getting stats: {e}")
            return {"total_datasets": 0, "tagged": "unknown", "untagged": "unknown"}
