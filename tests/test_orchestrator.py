"""
Unit tests for TaggingOrchestrator.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.services.orchestrator import TaggingOrchestrator
from src.services.cache import TaggingCache


class TestTaggingOrchestratorCacheFlow:
    """Tests for cache-first workflow."""

    @pytest.fixture
    def cache(self, temp_cache_dir):
        return TaggingCache(temp_cache_dir / "cache.json", "testconfig")

    @pytest.fixture
    def orchestrator(self, mock_openrouter_tagger, cache, temp_cache_dir):
        return TaggingOrchestrator(
            tagger=mock_openrouter_tagger,
            cache=cache,
            allow_on_demand=True,
            serve_stale_on_error=True,
            abstract_cache_path=temp_cache_dir / "abstract.json"
        )

    def test_tag_cache_hit_skips_llm(
        self, orchestrator, cache, mock_openrouter_tagger, sample_metadata
    ):
        """Test that cache hit skips LLM call."""
        # Pre-populate cache
        key = cache.build_key("ds001", sample_metadata, mock_openrouter_tagger.model)
        cache.set(key, {"pathology": ["Cached"]}, sample_metadata)

        # Mock metadata extraction to return same metadata
        with patch.object(orchestrator, '_clone_repo'), \
             patch.object(orchestrator, '_extract_metadata', return_value=sample_metadata):
            result = orchestrator.tag("ds001", "https://github.com/test/ds001")

        assert result["from_cache"] == True
        assert result["pathology"] == ["Cached"]
        mock_openrouter_tagger.tag_with_details.assert_not_called()

    def test_tag_cache_miss_calls_llm(
        self, orchestrator, mock_openrouter_tagger, sample_metadata
    ):
        """Test that cache miss triggers LLM call."""
        with patch.object(orchestrator, '_clone_repo'), \
             patch.object(orchestrator, '_extract_metadata', return_value=sample_metadata):
            result = orchestrator.tag("ds001", "https://github.com/test/ds001")

        assert result["from_cache"] == False
        mock_openrouter_tagger.tag_with_details.assert_called_once()

    def test_tag_force_refresh_bypasses_cache(
        self, orchestrator, cache, mock_openrouter_tagger, sample_metadata
    ):
        """Test that force_refresh bypasses cache."""
        # Pre-populate cache
        key = cache.build_key("ds001", sample_metadata, mock_openrouter_tagger.model)
        cache.set(key, {"pathology": ["Cached"]}, sample_metadata)

        with patch.object(orchestrator, '_clone_repo'), \
             patch.object(orchestrator, '_extract_metadata', return_value=sample_metadata):
            result = orchestrator.tag(
                "ds001", "https://github.com/test/ds001", force_refresh=True
            )

        assert result["from_cache"] == False
        mock_openrouter_tagger.tag_with_details.assert_called_once()

    def test_tag_stores_result_in_cache(
        self, orchestrator, cache, mock_openrouter_tagger, sample_metadata
    ):
        """Test that LLM result is stored in cache."""
        with patch.object(orchestrator, '_clone_repo'), \
             patch.object(orchestrator, '_extract_metadata', return_value=sample_metadata):
            orchestrator.tag("ds001", "https://github.com/test/ds001")

        # Verify cache was populated
        assert cache.stats()["total_entries"] == 1
        assert "ds001" in cache.stats()["datasets"]


class TestTaggingOrchestratorErrorHandling:
    """Tests for error handling and stale fallback."""

    @pytest.fixture
    def cache(self, temp_cache_dir):
        return TaggingCache(temp_cache_dir / "cache.json", "testconfig")

    @pytest.fixture
    def orchestrator(self, mock_openrouter_tagger, cache, temp_cache_dir):
        return TaggingOrchestrator(
            tagger=mock_openrouter_tagger,
            cache=cache,
            allow_on_demand=True,
            serve_stale_on_error=True,
            abstract_cache_path=temp_cache_dir / "abstract.json"
        )

    def test_clone_failure_serves_stale(
        self, orchestrator, cache, mock_openrouter_tagger
    ):
        """Test stale result served when clone fails."""
        # Pre-populate cache with old result
        old_metadata = {"title": "Old", "readme": "Old content"}
        key = cache.build_key("ds001", old_metadata, mock_openrouter_tagger.model)
        cache.set(key, {"pathology": ["StaleResult"]}, old_metadata)

        with patch.object(
            orchestrator, '_clone_repo', side_effect=RuntimeError("Clone failed")
        ):
            result = orchestrator.tag("ds001", "https://github.com/test/ds001")

        assert result["stale"] == True
        assert result["pathology"] == ["StaleResult"]
        assert "error" in result

    def test_llm_failure_serves_stale(
        self, orchestrator, cache, mock_openrouter_tagger, sample_metadata
    ):
        """Test stale result served when LLM fails."""
        # Pre-populate cache
        key = cache.build_key("ds001", sample_metadata, mock_openrouter_tagger.model)
        cache.set(key, {"pathology": ["StaleResult"]}, sample_metadata)

        # Make LLM fail
        mock_openrouter_tagger.tag_with_details.side_effect = Exception("API error")

        # Use different metadata to force cache miss
        different_metadata = {"title": "Different", "readme": "Different content"}
        with patch.object(orchestrator, '_clone_repo'), \
             patch.object(orchestrator, '_extract_metadata', return_value=different_metadata):
            result = orchestrator.tag("ds001", "https://github.com/test/ds001")

        assert result["stale"] == True
        assert "error" in result

    def test_insufficient_metadata_returns_unknown(self, orchestrator):
        """Test handling of datasets with no readme/description."""
        metadata = {"title": "Test"}  # Missing readme and dataset_description

        with patch.object(orchestrator, '_clone_repo'), \
             patch.object(orchestrator, '_extract_metadata', return_value=metadata):
            result = orchestrator.tag("ds001", "https://github.com/test/ds001")

        assert result["pathology"] == ["Unknown"]
        assert "Insufficient metadata" in result.get("error", "")


class TestTaggingOrchestratorOnDemandControl:
    """Tests for on-demand tagging control."""

    def test_on_demand_disabled_serves_cached_only(
        self, mock_openrouter_tagger, temp_cache_dir, sample_metadata
    ):
        """Test that on_demand=False only serves cached results."""
        cache = TaggingCache(temp_cache_dir / "cache.json", "testconfig")
        orchestrator = TaggingOrchestrator(
            tagger=mock_openrouter_tagger,
            cache=cache,
            allow_on_demand=False  # Disable on-demand tagging
        )

        with patch.object(orchestrator, '_clone_repo'), \
             patch.object(orchestrator, '_extract_metadata', return_value=sample_metadata):
            result = orchestrator.tag("ds001", "https://github.com/test/ds001")

        # Should not call LLM
        mock_openrouter_tagger.tag_with_details.assert_not_called()
        assert "error" in result
        assert "on-demand" in result["error"].lower()

    def test_on_demand_disabled_serves_stale_if_available(
        self, mock_openrouter_tagger, temp_cache_dir, sample_metadata
    ):
        """Test that on_demand=False serves stale if available."""
        cache = TaggingCache(temp_cache_dir / "cache.json", "testconfig")

        # Pre-populate with stale result
        old_metadata = {"title": "Old", "readme": "Old content"}
        key = cache.build_key("ds001", old_metadata, mock_openrouter_tagger.model)
        cache.set(key, {"pathology": ["StaleResult"]}, old_metadata)

        orchestrator = TaggingOrchestrator(
            tagger=mock_openrouter_tagger,
            cache=cache,
            allow_on_demand=False
        )

        with patch.object(orchestrator, '_clone_repo'), \
             patch.object(orchestrator, '_extract_metadata', return_value=sample_metadata):
            result = orchestrator.tag("ds001", "https://github.com/test/ds001")

        assert result["pathology"] == ["StaleResult"]
        assert result["stale"] == True
        mock_openrouter_tagger.tag_with_details.assert_not_called()
