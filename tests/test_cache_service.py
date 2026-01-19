"""
Unit tests for TaggingCache service.
"""
import pytest
import json
from pathlib import Path

from src.services.cache import TaggingCache


class TestTaggingCacheBasics:
    """Basic cache functionality tests."""

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a TaggingCache instance with temp directory."""
        cache_path = temp_cache_dir / "test_cache.json"
        return TaggingCache(cache_path=cache_path, config_hash="testconfig123")

    def test_compute_hash_deterministic(self, cache):
        """Test that hash is deterministic for same input."""
        data = {"title": "Test", "readme": "Content"}
        hash1 = cache.compute_hash(data)
        hash2 = cache.compute_hash(data)
        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of SHA-256

    def test_compute_hash_different_for_different_data(self, cache):
        """Test that different data produces different hashes."""
        hash1 = cache.compute_hash({"title": "Test1"})
        hash2 = cache.compute_hash({"title": "Test2"})
        assert hash1 != hash2

    def test_compute_hash_order_independent(self, cache):
        """Test that key order doesn't affect hash."""
        hash1 = cache.compute_hash({"a": 1, "b": 2})
        hash2 = cache.compute_hash({"b": 2, "a": 1})
        assert hash1 == hash2

    def test_build_key_format(self, cache):
        """Test cache key format."""
        metadata = {"title": "Visual Study", "readme": "Test content"}
        key = cache.build_key(
            dataset_id="ds001234",
            metadata=metadata,
            model="openai/gpt-4-turbo"
        )

        parts = key.split(":")
        assert len(parts) == 4
        assert parts[0] == "ds001234"
        assert parts[2] == "testconfig123"  # config_hash
        assert parts[3] == "openai/gpt-4-turbo"

    def test_build_key_filters_irrelevant_metadata(self, cache):
        """Test that irrelevant metadata fields are filtered out."""
        metadata_full = {
            "title": "Test",
            "readme": "Content",
            "irrelevant_field": "Should be ignored",
            "another_field": 12345
        }
        metadata_minimal = {"title": "Test", "readme": "Content"}

        key_full = cache.build_key("ds001", metadata_full, "model")
        key_minimal = cache.build_key("ds001", metadata_minimal, "model")

        # Keys should be identical since irrelevant fields are filtered
        assert key_full == key_minimal


class TestTaggingCacheStorage:
    """Cache storage and retrieval tests."""

    @pytest.fixture
    def cache(self, temp_cache_dir):
        cache_path = temp_cache_dir / "test_cache.json"
        return TaggingCache(cache_path=cache_path, config_hash="testconfig123")

    def test_set_and_get(self, cache):
        """Test storing and retrieving cache entries."""
        key = "ds001:hash1:testconfig123:model"
        result = {"pathology": ["Healthy"], "modality": ["Visual"]}
        metadata = {"title": "Test"}

        cache.set(key, result, metadata)
        retrieved = cache.get(key)

        assert retrieved == result

    def test_get_nonexistent_key(self, cache):
        """Test getting a nonexistent key returns None."""
        result = cache.get("nonexistent:key:here:model")
        assert result is None

    def test_get_any_for_dataset_returns_stale(self, cache):
        """Test get_any_for_dataset returns stale results."""
        # Store with specific key
        key = "ds001:hash1:testconfig123:model"
        result = {"pathology": ["Healthy"]}
        cache.set(key, result, {})

        # Request with different hash should return stale
        retrieved = cache.get_any_for_dataset("ds001")

        assert retrieved is not None
        assert retrieved["stale"] == True
        assert retrieved["result"] == result

    def test_get_any_for_dataset_no_match(self, cache):
        """Test get_any_for_dataset returns None when no match."""
        result = cache.get_any_for_dataset("ds999")
        assert result is None

    def test_delete_entry(self, cache):
        """Test deleting a cache entry."""
        key = "ds001:hash:testconfig123:model"
        cache.set(key, {"test": "data"}, {})

        deleted = cache.delete(key)
        assert deleted == True
        assert cache.get(key) is None

    def test_delete_nonexistent(self, cache):
        """Test deleting nonexistent entry returns False."""
        deleted = cache.delete("nonexistent:key:here:model")
        assert deleted == False

    def test_clear(self, cache):
        """Test clearing all entries."""
        cache.set("ds001:h:c:m", {"a": 1}, {})
        cache.set("ds002:h:c:m", {"b": 2}, {})

        cache.clear()

        assert cache.stats()["total_entries"] == 0

    def test_stats(self, cache):
        """Test cache statistics."""
        cache.set("ds001:h1:c:m", {"a": 1}, {})
        cache.set("ds001:h2:c:m", {"b": 2}, {})
        cache.set("ds002:h1:c:m", {"c": 3}, {})

        stats = cache.stats()

        assert stats["total_entries"] == 3
        assert stats["unique_datasets"] == 2
        assert set(stats["datasets"]) == {"ds001", "ds002"}

    def test_list_entries(self, cache):
        """Test listing cache entries."""
        cache.set("ds001:h:c:m", {"pathology": ["Healthy"]}, {})

        entries = cache.list_entries()

        assert len(entries) == 1
        assert entries[0]["dataset_id"] == "ds001"
        assert entries[0]["pathology"] == ["Healthy"]

    def test_list_entries_filtered(self, cache):
        """Test listing entries filtered by dataset_id."""
        cache.set("ds001:h:c:m", {"pathology": ["Healthy"]}, {})
        cache.set("ds002:h:c:m", {"pathology": ["Epilepsy"]}, {})

        entries = cache.list_entries(dataset_id="ds001")

        assert len(entries) == 1
        assert entries[0]["dataset_id"] == "ds001"


class TestTaggingCachePersistence:
    """Cache persistence tests."""

    def test_persistence_across_instances(self, temp_cache_dir):
        """Test that cache persists across instances."""
        cache_path = temp_cache_dir / "persist_test.json"

        # Create and populate first instance
        cache1 = TaggingCache(cache_path, "config")
        cache1.set("ds001:h:c:m", {"data": "test"}, {})

        # Create second instance with same path
        cache2 = TaggingCache(cache_path, "config")

        assert cache2.get("ds001:h:c:m") == {"data": "test"}

    def test_handles_corrupted_cache(self, temp_cache_dir):
        """Test graceful handling of corrupted cache file."""
        cache_path = temp_cache_dir / "corrupted.json"
        cache_path.write_text("not valid json {{{")

        # Should not raise, should start with empty cache
        cache = TaggingCache(cache_path, "config")
        assert cache.stats()["total_entries"] == 0


class TestTaggingCacheInvalidation:
    """Tests for cache invalidation behavior."""

    def test_different_metadata_produces_different_key(self, temp_cache_dir):
        """Verify metadata changes invalidate cache."""
        cache = TaggingCache(temp_cache_dir / "cache.json", "config")

        metadata1 = {"title": "Study A", "readme": "Content 1"}
        metadata2 = {"title": "Study A", "readme": "Content 2"}  # Changed readme

        key1 = cache.build_key("ds001", metadata1, "model")
        key2 = cache.build_key("ds001", metadata2, "model")

        # Keys should differ due to different metadata hash
        assert key1 != key2
        assert key1.split(":")[0] == key2.split(":")[0]  # Same dataset_id
        assert key1.split(":")[1] != key2.split(":")[1]  # Different metadata hash

    def test_different_config_produces_different_key(self, temp_cache_dir):
        """Verify config changes invalidate cache."""
        metadata = {"title": "Test", "readme": "Content"}

        cache1 = TaggingCache(temp_cache_dir / "c1.json", "config_v1")
        cache2 = TaggingCache(temp_cache_dir / "c2.json", "config_v2")

        key1 = cache1.build_key("ds001", metadata, "model")
        key2 = cache2.build_key("ds001", metadata, "model")

        assert key1 != key2
        assert key1.split(":")[2] != key2.split(":")[2]  # Different config hash

    def test_different_model_produces_different_key(self, temp_cache_dir):
        """Verify model changes invalidate cache."""
        cache = TaggingCache(temp_cache_dir / "cache.json", "config")
        metadata = {"title": "Test", "readme": "Content"}

        key1 = cache.build_key("ds001", metadata, "openai/gpt-4")
        key2 = cache.build_key("ds001", metadata, "anthropic/claude-3")

        assert key1 != key2
        assert key1.split(":")[3] != key2.split(":")[3]  # Different model

    def test_stale_fallback_across_config_changes(self, temp_cache_dir):
        """Test stale fallback works across config changes."""
        cache_path = temp_cache_dir / "cache.json"
        metadata = {"title": "Test", "readme": "Content"}

        # Create cache with old config
        cache_old = TaggingCache(cache_path, "config_v1")
        old_key = cache_old.build_key("ds001", metadata, "model")
        cache_old.set(old_key, {"pathology": ["OldResult"]}, metadata)

        # Create cache with new config
        cache_new = TaggingCache(cache_path, "config_v2")

        # Exact key lookup should fail (different config hash)
        new_key = cache_new.build_key("ds001", metadata, "model")
        assert cache_new.get(new_key) is None

        # Stale fallback should work
        stale = cache_new.get_any_for_dataset("ds001")
        assert stale is not None
        assert stale["stale"] == True
        assert stale["result"]["pathology"] == ["OldResult"]
