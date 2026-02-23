"""
Content-addressable cache for LLM tagging results.

This module implements a write-through cache with content-addressable keys
for automatic invalidation when inputs change.

Also supports ground truth entries that are never invalidated.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Sentinel hash for ground truth entries (never invalidated)
GROUND_TRUTH_HASH = "GROUND_TRUTH"


class TaggingCache:
    """
    Write-through cache with content-addressable keys.

    Features:
    - Automatic invalidation via content hashing
    - Stale result fallback for graceful degradation
    - Persistent JSON storage (Redis-upgradeable)

    Cache Key Format:
        {dataset_id}:{metadata_hash}:{config_hash}:{model}

    Example:
        "ds004398:a1b2c3d4:e5f6g7h8:openai/gpt-4-turbo"
    """

    def __init__(self, cache_path: Path, config_hash: str):
        """
        Initialize the cache.

        Args:
            cache_path: Path to the JSON cache file
            config_hash: Hash of configuration (few-shot + prompt) for invalidation
        """
        self.cache_path = cache_path
        self.config_hash = config_hash
        self._cache = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # Corrupted cache - start fresh
                return {}
        return {}

    def _save(self) -> None:
        """Persist cache to disk atomically."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file then rename for atomicity
        temp_path = self.cache_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2, default=str, ensure_ascii=False)
            temp_path.replace(self.cache_path)
        except IOError:
            # Cache save failed - continue without persistence
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def compute_hash(data: Any) -> str:
        """
        Compute SHA-256 hash of JSON-serializable data.

        Args:
            data: Any JSON-serializable data

        Returns:
            First 16 characters of hex-encoded SHA-256 hash
        """
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def build_key(self, dataset_id: str, metadata: Dict[str, Any], model: str) -> str:
        """
        Build content-addressable cache key.

        Args:
            dataset_id: Dataset identifier
            metadata: Extracted metadata dict
            model: LLM model identifier

        Returns:
            Cache key in format: {dataset_id}:{metadata_hash}:{config_hash}:{model}
        """
        # Filter to relevant metadata fields only
        relevant_keys = {
            "title", "dataset_description", "readme",
            "participants_overview", "tasks", "events", "paper_abstract"
        }
        filtered = {k: v for k, v in metadata.items() if k in relevant_keys and v}
        metadata_hash = self.compute_hash(filtered)

        return f"{dataset_id}:{metadata_hash}:{self.config_hash}:{model}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result by exact key.

        Args:
            key: Cache key

        Returns:
            Cached result dict or None if not found
        """
        entry = self._cache.get(key)
        return entry.get("result") if entry else None

    def get_any_for_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get any cached result for dataset (stale fallback).

        Used for graceful degradation when exact key doesn't match
        but we have some previous result for the dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dict with 'result', 'timestamp', and 'stale' flag, or None
        """
        for key, entry in self._cache.items():
            if key.startswith(f"{dataset_id}:"):
                return {
                    "result": entry["result"],
                    "timestamp": entry["timestamp"],
                    "stale": True
                }
        return None

    def set(self, key: str, result: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """
        Store result in cache with metadata snapshot.

        Args:
            key: Cache key
            result: Tagging result to store
            metadata: Metadata snapshot for debugging
        """
        self._cache[key] = {
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata_snapshot": metadata
        }
        self._save()

    def delete(self, key: str) -> bool:
        """
        Delete a specific cache entry.

        Args:
            key: Cache key to delete

        Returns:
            True if entry was deleted, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            self._save()
            return True
        return False

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache = {}
        self._save()

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with total_entries, config_hash, and unique datasets
        """
        datasets = set()
        for key in self._cache.keys():
            parts = key.split(":")
            if parts:
                datasets.add(parts[0])

        return {
            "total_entries": len(self._cache),
            "config_hash": self.config_hash,
            "unique_datasets": len(datasets),
            "datasets": sorted(datasets)
        }

    def list_entries(self, dataset_id: Optional[str] = None) -> list[Dict[str, Any]]:
        """
        List cache entries, optionally filtered by dataset.

        Args:
            dataset_id: Optional filter by dataset

        Returns:
            List of cache entry summaries
        """
        entries = []
        for key, entry in self._cache.items():
            if dataset_id and not key.startswith(f"{dataset_id}:"):
                continue

            parts = key.split(":")
            entries.append({
                "key": key,
                "dataset_id": parts[0] if parts else "unknown",
                "timestamp": entry.get("timestamp"),
                "pathology": entry.get("result", {}).get("pathology", []),
                "modality": entry.get("result", {}).get("modality", []),
                "type": entry.get("result", {}).get("type", []),
                "is_ground_truth": entry.get("is_ground_truth", False),
            })

        return entries

    def set_ground_truth(self, dataset_id: str, result: Dict[str, Any]) -> None:
        """
        Store ground truth that won't be invalidated by config/metadata changes.

        Ground truth entries use sentinel hashes (GROUND_TRUTH) so they are
        never invalidated when config or metadata changes.

        Args:
            dataset_id: Dataset identifier
            result: Ground truth tags (pathology, modality, type, confidence)
        """
        key = f"{dataset_id}:{GROUND_TRUTH_HASH}:{GROUND_TRUTH_HASH}:ground_truth"
        self._cache[key] = {
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_ground_truth": True,
        }
        self._save()

    def get_ground_truth(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Check for ground truth entry for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Ground truth result dict or None if not found
        """
        key = f"{dataset_id}:{GROUND_TRUTH_HASH}:{GROUND_TRUTH_HASH}:ground_truth"
        entry = self._cache.get(key)
        if entry and entry.get("is_ground_truth"):
            return entry.get("result")
        return None

    def has_ground_truth(self, dataset_id: str) -> bool:
        """
        Check if dataset has ground truth entry.

        Args:
            dataset_id: Dataset identifier

        Returns:
            True if ground truth exists for this dataset
        """
        return self.get_ground_truth(dataset_id) is not None

    def list_ground_truth_datasets(self) -> list[str]:
        """
        List all dataset IDs that have ground truth entries.

        Returns:
            List of dataset IDs with ground truth
        """
        datasets = []
        for key, entry in self._cache.items():
            if entry.get("is_ground_truth"):
                parts = key.split(":")
                if parts:
                    datasets.append(parts[0])
        return sorted(datasets)
