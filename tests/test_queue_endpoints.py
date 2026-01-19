"""
Tests for queue API endpoints.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


class TestQueueEndpointsDisabled:
    """Tests for queue endpoints when queue is disabled."""

    @pytest.fixture
    def client_no_queue(self, temp_cache_dir, mock_tagger_class):
        """Create test client without queue enabled."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir),
            # No POSTGRES_URL = queue disabled
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class):
            from src.api.main import app
            with TestClient(app) as client:
                yield client

    def test_enqueue_returns_503_when_disabled(self, client_no_queue):
        """Test enqueue returns 503 when queue not enabled."""
        response = client_no_queue.post("/api/v1/tag/enqueue", json={
            "dataset_id": "ds001234",
            "source_url": "https://github.com/test/ds001234",
            "metadata_snapshot": {"title": "Test"}
        })

        assert response.status_code == 503
        assert "Queue not enabled" in response.json()["detail"]

    def test_enqueue_batch_returns_503_when_disabled(self, client_no_queue):
        """Test batch enqueue returns 503 when queue not enabled."""
        response = client_no_queue.post("/api/v1/tag/enqueue/batch", json={
            "datasets": [{
                "dataset_id": "ds001234",
                "source_url": "https://github.com/test/ds001234",
                "metadata_snapshot": {"title": "Test"}
            }]
        })

        assert response.status_code == 503

    def test_queue_stats_returns_503_when_disabled(self, client_no_queue):
        """Test queue stats returns 503 when queue not enabled."""
        response = client_no_queue.get("/api/v1/queue/stats")

        assert response.status_code == 503

    def test_job_status_returns_503_when_disabled(self, client_no_queue):
        """Test job status returns 503 when queue not enabled."""
        response = client_no_queue.get("/api/v1/queue/status/ds001234")

        assert response.status_code == 503

    def test_health_shows_queue_disabled(self, client_no_queue):
        """Test health endpoint shows queue disabled."""
        response = client_no_queue.get("/health")

        assert response.status_code == 200
        assert response.json()["queue_enabled"] is False


class TestQueueEndpointsEnabled:
    """Tests for queue endpoints when queue is enabled."""

    @pytest.fixture
    def mock_queue(self):
        """Create a mock queue."""
        queue = AsyncMock()
        queue.enqueue = AsyncMock(return_value=(42, True))
        queue.enqueue_batch = AsyncMock(return_value={"queued": 2, "duplicates": 0})
        queue.get_stats = AsyncMock(return_value={
            "pending": 10,
            "processing": 2,
            "completed": 100,
            "failed": 5,
            "ready_to_process": 8,
        })
        queue.get_job_status = AsyncMock(return_value={
            "job_id": 1,
            "status": "completed",
            "attempts": 1,
            "created_at": "2024-01-19T12:00:00Z",
            "completed_at": "2024-01-19T12:05:00Z",
            "result": {"pathology": ["Healthy"]},
            "error": None,
        })
        return queue

    @pytest.fixture
    def client_with_queue(self, temp_cache_dir, mock_tagger_class, mock_queue):
        """Create test client with queue enabled."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir),
            'POSTGRES_URL': 'postgresql://mock',
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class), \
            patch('src.api.main.TaggingQueue') as mock_queue_class:

            # Make the queue class return our mock and initialize successfully
            mock_queue_class.return_value = mock_queue
            mock_queue.initialize = AsyncMock()
            mock_queue.close = AsyncMock()

            from src.api.main import app
            with TestClient(app) as client:
                yield client

    def test_enqueue_success(self, client_with_queue):
        """Test successful job enqueue."""
        response = client_with_queue.post("/api/v1/tag/enqueue", json={
            "dataset_id": "ds001234",
            "source_url": "https://github.com/test/ds001234",
            "metadata_snapshot": {"title": "Test Study"}
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["job_id"] == 42
        assert data["dataset_id"] == "ds001234"
        assert data["is_new"] is True

    def test_enqueue_batch_success(self, client_with_queue):
        """Test successful batch enqueue."""
        response = client_with_queue.post("/api/v1/tag/enqueue/batch", json={
            "datasets": [
                {
                    "dataset_id": "ds001",
                    "source_url": "https://github.com/test/ds001",
                    "metadata_snapshot": {"title": "Study 1"}
                },
                {
                    "dataset_id": "ds002",
                    "source_url": "https://github.com/test/ds002",
                    "metadata_snapshot": {"title": "Study 2"}
                }
            ]
        })

        assert response.status_code == 200
        data = response.json()
        assert data["queued"] == 2
        assert data["duplicates"] == 0
        assert data["total"] == 2

    def test_queue_stats_success(self, client_with_queue):
        """Test getting queue statistics."""
        response = client_with_queue.get("/api/v1/queue/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["pending"] == 10
        assert data["processing"] == 2
        assert data["completed"] == 100
        assert data["failed"] == 5
        assert data["ready_to_process"] == 8

    def test_job_status_success(self, client_with_queue):
        """Test getting job status."""
        response = client_with_queue.get("/api/v1/queue/status/ds001234")

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == "ds001234"
        assert data["status"] == "completed"
        assert data["result"]["pathology"] == ["Healthy"]

    def test_job_status_not_found(self, client_with_queue, mock_queue):
        """Test job status for non-existent dataset."""
        mock_queue.get_job_status.return_value = None

        response = client_with_queue.get("/api/v1/queue/status/ds999999")

        assert response.status_code == 404
        assert "No tagging jobs found" in response.json()["detail"]

    def test_health_shows_queue_enabled(self, client_with_queue):
        """Test health endpoint shows queue enabled."""
        response = client_with_queue.get("/health")

        assert response.status_code == 200
        assert response.json()["queue_enabled"] is True

    def test_root_shows_queue_endpoints(self, client_with_queue):
        """Test root endpoint lists queue endpoints when enabled."""
        response = client_with_queue.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["queue_enabled"] is True
        assert "enqueue" in data["endpoints"]
        assert "enqueue_batch" in data["endpoints"]
        assert "queue_stats" in data["endpoints"]
        assert "job_status" in data["endpoints"]


class TestEnqueueValidation:
    """Tests for enqueue request validation."""

    @pytest.fixture
    def mock_queue(self):
        """Create a mock queue."""
        queue = AsyncMock()
        queue.enqueue = AsyncMock(return_value=(1, True))
        queue.initialize = AsyncMock()
        queue.close = AsyncMock()
        return queue

    @pytest.fixture
    def client(self, temp_cache_dir, mock_tagger_class, mock_queue):
        """Create test client with queue enabled."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key-12345',
            'CACHE_DIR': str(temp_cache_dir),
            'POSTGRES_URL': 'postgresql://mock',
        }), patch('src.api.main.OpenRouterTagger', mock_tagger_class), \
            patch('src.api.main.TaggingQueue') as mock_queue_class:

            mock_queue_class.return_value = mock_queue

            from src.api.main import app
            with TestClient(app) as client:
                yield client

    def test_enqueue_requires_dataset_id(self, client):
        """Test enqueue requires dataset_id."""
        response = client.post("/api/v1/tag/enqueue", json={
            "source_url": "https://github.com/test/ds001234",
            "metadata_snapshot": {"title": "Test"}
        })

        assert response.status_code == 422

    def test_enqueue_requires_source_url(self, client):
        """Test enqueue requires source_url."""
        response = client.post("/api/v1/tag/enqueue", json={
            "dataset_id": "ds001234",
            "metadata_snapshot": {"title": "Test"}
        })

        assert response.status_code == 422

    def test_enqueue_requires_metadata_snapshot(self, client):
        """Test enqueue requires metadata_snapshot."""
        response = client.post("/api/v1/tag/enqueue", json={
            "dataset_id": "ds001234",
            "source_url": "https://github.com/test/ds001234",
        })

        assert response.status_code == 422

    def test_enqueue_accepts_empty_metadata(self, client):
        """Test enqueue accepts empty metadata dict."""
        response = client.post("/api/v1/tag/enqueue", json={
            "dataset_id": "ds001234",
            "source_url": "https://github.com/test/ds001234",
            "metadata_snapshot": {}
        })

        assert response.status_code == 200
