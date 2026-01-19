"""
Unit tests for FastAPI endpoints.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_returns_healthy_status(self, temp_cache_dir, mock_tagger_class):
        """Test health check returns healthy status."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'LLM_MODEL': 'openai/gpt-4-turbo',
            'CACHE_DIR': str(temp_cache_dir)
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class):
            from src.api.main import app
            with TestClient(app) as client:
                response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "cache_entries" in data
        assert "config_hash" in data


class TestRootEndpoint:
    """Tests for GET /"""

    def test_root_returns_api_info(self, temp_cache_dir, mock_tagger_class):
        """Test root endpoint returns API info."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir)
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class):
            from src.api.main import app
            with TestClient(app) as client:
                response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "EEGDash LLM Tagger API"
        assert "endpoints" in data


class TestTagEndpoint:
    """Tests for POST /api/v1/tag"""

    def test_tag_dataset_success(
        self, temp_cache_dir, mock_tagger_class,
        mock_git_clone, mock_metadata_extraction, mock_abstract_fetcher
    ):
        """Test successful tagging of a dataset."""
        # mock_abstract_fetcher is auto-applied as fixture, not context manager
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir)
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class), \
            mock_git_clone, mock_metadata_extraction:
            from src.api.main import app
            with TestClient(app) as client:
                response = client.post("/api/v1/tag", json={
                    "dataset_id": "ds001234",
                    "source_url": "https://github.com/OpenNeuroDatasets/ds001234"
                })

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == "ds001234"
        assert "pathology" in data
        assert "modality" in data
        assert "type" in data
        assert isinstance(data["pathology"], list)
        assert isinstance(data["confidence"], dict)

    def test_tag_dataset_returns_cached_result(
        self, temp_cache_dir, mock_tagger_class,
        mock_git_clone, mock_metadata_extraction, mock_abstract_fetcher
    ):
        """Test that second request returns cached result."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir)
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class), \
            mock_git_clone, mock_metadata_extraction:
            from src.api.main import app
            with TestClient(app) as client:
                # First request
                client.post("/api/v1/tag", json={
                    "dataset_id": "ds001234",
                    "source_url": "https://github.com/OpenNeuroDatasets/ds001234"
                })
                # Second request - should be cached
                response = client.post("/api/v1/tag", json={
                    "dataset_id": "ds001234",
                    "source_url": "https://github.com/OpenNeuroDatasets/ds001234"
                })

        assert response.json()["from_cache"] == True

    def test_tag_dataset_force_refresh(
        self, temp_cache_dir, mock_tagger_class,
        mock_git_clone, mock_metadata_extraction, mock_abstract_fetcher
    ):
        """Test force_refresh bypasses cache."""
        # mock_abstract_fetcher is auto-applied as fixture, not context manager
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir)
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class), \
            mock_git_clone, mock_metadata_extraction:
            from src.api.main import app
            with TestClient(app) as client:
                # First request
                client.post("/api/v1/tag", json={
                    "dataset_id": "ds001234",
                    "source_url": "https://github.com/OpenNeuroDatasets/ds001234"
                })
                # Force refresh
                response = client.post("/api/v1/tag", json={
                    "dataset_id": "ds001234",
                    "source_url": "https://github.com/OpenNeuroDatasets/ds001234",
                    "force_refresh": True
                })

        assert response.json()["from_cache"] == False


class TestGetCachedTagsEndpoint:
    """Tests for GET /api/v1/tags/{dataset_id}"""

    def test_get_cached_tags_not_found(self, temp_cache_dir, mock_tagger_class):
        """Test 404 when no cached tags exist."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir)
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class):
            from src.api.main import app
            with TestClient(app) as client:
                response = client.get("/api/v1/tags/ds999999")

        assert response.status_code == 404

    def test_get_cached_tags_success(
        self, temp_cache_dir, mock_tagger_class,
        mock_git_clone, mock_metadata_extraction, mock_abstract_fetcher
    ):
        """Test retrieving cached tags."""
        # mock_abstract_fetcher is auto-applied as fixture, not context manager
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir)
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class), \
            mock_git_clone, mock_metadata_extraction:
            from src.api.main import app
            with TestClient(app) as client:
                # First, tag a dataset
                client.post("/api/v1/tag", json={
                    "dataset_id": "ds001234",
                    "source_url": "https://github.com/OpenNeuroDatasets/ds001234"
                })
                # Now retrieve cached tags
                response = client.get("/api/v1/tags/ds001234")

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == "ds001234"
        assert data["from_cache"] == True


class TestCacheEndpoints:
    """Tests for cache management endpoints."""

    def test_cache_stats_empty(self, temp_cache_dir, mock_tagger_class):
        """Test cache stats when empty."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir)
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class):
            from src.api.main import app
            with TestClient(app) as client:
                response = client.get("/api/v1/cache/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_entries"] == 0

    def test_cache_entries_empty(self, temp_cache_dir, mock_tagger_class):
        """Test listing cache entries when empty."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir)
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class):
            from src.api.main import app
            with TestClient(app) as client:
                response = client.get("/api/v1/cache/entries")

        assert response.status_code == 200
        assert response.json() == []

    def test_cache_clear(
        self, temp_cache_dir, mock_tagger_class,
        mock_git_clone, mock_metadata_extraction, mock_abstract_fetcher
    ):
        """Test clearing the cache."""
        # mock_abstract_fetcher is auto-applied as fixture, not context manager
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir)
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class), \
            mock_git_clone, mock_metadata_extraction:
            from src.api.main import app
            with TestClient(app) as client:
                # Add an entry
                client.post("/api/v1/tag", json={
                    "dataset_id": "ds001234",
                    "source_url": "https://github.com/OpenNeuroDatasets/ds001234"
                })
                # Clear cache
                response = client.delete("/api/v1/cache")

                assert response.status_code == 200

                # Verify empty
                stats = client.get("/api/v1/cache/stats").json()
                assert stats["total_entries"] == 0

    def test_delete_specific_cache_entry(
        self, temp_cache_dir, mock_tagger_class,
        mock_git_clone, mock_metadata_extraction, mock_abstract_fetcher
    ):
        """Test deleting a specific cache entry."""
        # mock_abstract_fetcher is auto-applied as fixture, not context manager
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir)
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class), \
            mock_git_clone, mock_metadata_extraction:
            from src.api.main import app
            with TestClient(app) as client:
                client.post("/api/v1/tag", json={
                    "dataset_id": "ds001234",
                    "source_url": "https://github.com/OpenNeuroDatasets/ds001234"
                })

                # Get cache entries to find the key
                entries = client.get("/api/v1/cache/entries").json()
                cache_key = entries[0]["key"]

                # Delete specific entry
                response = client.delete(f"/api/v1/cache/{cache_key}")
                assert response.status_code == 200

    def test_delete_nonexistent_cache_entry(self, temp_cache_dir, mock_tagger_class):
        """Test 404 when deleting nonexistent entry."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir)
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class):
            from src.api.main import app
            with TestClient(app) as client:
                response = client.delete("/api/v1/cache/nonexistent:key:here:model")

        assert response.status_code == 404
