"""
Tests for the Postgres tagging queue service.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta

from src.services.queue import TaggingQueue, TaggingJob


class TestTaggingQueueBasics:
    """Tests for basic queue operations."""

    def test_compute_metadata_hash_deterministic(self):
        """Test that same metadata produces same hash."""
        metadata = {"title": "Test Study", "readme": "Description"}

        hash1 = TaggingQueue.compute_metadata_hash(metadata)
        hash2 = TaggingQueue.compute_metadata_hash(metadata)

        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of SHA-256

    def test_compute_metadata_hash_order_independent(self):
        """Test that key order doesn't affect hash."""
        metadata1 = {"a": 1, "b": 2}
        metadata2 = {"b": 2, "a": 1}

        assert TaggingQueue.compute_metadata_hash(metadata1) == TaggingQueue.compute_metadata_hash(metadata2)

    def test_compute_metadata_hash_different_for_different_data(self):
        """Test that different metadata produces different hash."""
        metadata1 = {"title": "Study A"}
        metadata2 = {"title": "Study B"}

        assert TaggingQueue.compute_metadata_hash(metadata1) != TaggingQueue.compute_metadata_hash(metadata2)


class TestTaggingQueueWithMockDB:
    """Tests for queue operations with mocked database."""

    @pytest.fixture
    def mock_cursor(self):
        """Create a mock cursor."""
        cursor = AsyncMock()
        cursor.__aenter__ = AsyncMock(return_value=cursor)
        cursor.__aexit__ = AsyncMock(return_value=None)
        cursor.fetchone = AsyncMock()
        cursor.fetchall = AsyncMock()
        cursor.rowcount = 0
        return cursor

    @pytest.fixture
    def mock_conn(self, mock_cursor):
        """Create a mock connection."""
        conn = AsyncMock()
        conn.cursor = MagicMock(return_value=mock_cursor)
        conn.commit = AsyncMock()
        conn.close = AsyncMock()
        return conn

    @pytest.fixture
    def queue_with_mock_conn(self, mock_conn):
        """Create queue with mocked connection."""
        queue = TaggingQueue("postgresql://mock")
        queue._conn = mock_conn
        return queue

    @pytest.mark.asyncio
    async def test_enqueue_new_job(self, queue_with_mock_conn, mock_cursor):
        """Test enqueueing a new job."""
        # Mock: INSERT returns new ID
        mock_cursor.fetchone.return_value = {"id": 42}

        job_id, is_new = await queue_with_mock_conn.enqueue(
            dataset_id="ds001234",
            source_url="https://github.com/test/ds001234",
            metadata_snapshot={"title": "Test"},
        )

        assert job_id == 42
        assert is_new is True

    @pytest.mark.asyncio
    async def test_enqueue_duplicate_job(self, queue_with_mock_conn, mock_cursor):
        """Test enqueueing a duplicate job returns existing ID."""
        # Mock: INSERT returns None (conflict), then SELECT returns existing
        mock_cursor.fetchone.side_effect = [None, {"id": 99}]

        job_id, is_new = await queue_with_mock_conn.enqueue(
            dataset_id="ds001234",
            source_url="https://github.com/test/ds001234",
            metadata_snapshot={"title": "Test"},
        )

        assert job_id == 99
        assert is_new is False

    @pytest.mark.asyncio
    async def test_claim_job_returns_job(self, queue_with_mock_conn, mock_cursor):
        """Test claiming a job from queue."""
        now = datetime.now(timezone.utc)
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "dataset_id": "ds001234",
            "source_url": "https://github.com/test/ds001234",
            "metadata_snapshot": '{"title": "Test"}',
            "metadata_hash": "abc123",
            "status": "processing",
            "attempts": 1,
            "max_attempts": 3,
            "created_at": now,
            "started_at": now,
            "completed_at": None,
            "next_retry_at": None,
            "result": None,
            "error": None,
        }

        job = await queue_with_mock_conn.claim_job("worker-1")

        assert job is not None
        assert job.id == 1
        assert job.dataset_id == "ds001234"
        assert job.attempts == 1

    @pytest.mark.asyncio
    async def test_claim_job_returns_none_when_empty(self, queue_with_mock_conn, mock_cursor):
        """Test claiming from empty queue returns None."""
        mock_cursor.fetchone.return_value = None

        job = await queue_with_mock_conn.claim_job("worker-1")

        assert job is None

    @pytest.mark.asyncio
    async def test_complete_job(self, queue_with_mock_conn):
        """Test marking job as complete."""
        result = {"pathology": ["Healthy"], "modality": ["Visual"]}

        await queue_with_mock_conn.complete_job(1, result)

        # Verify commit was called
        queue_with_mock_conn._conn.commit.assert_called()

    @pytest.mark.asyncio
    async def test_fail_job_with_retry(self, queue_with_mock_conn, mock_cursor):
        """Test failing a job schedules retry."""
        mock_cursor.fetchone.return_value = {"attempts": 1, "max_attempts": 3}

        await queue_with_mock_conn.fail_job(1, "API Error")

        # Should set status back to pending with next_retry_at
        queue_with_mock_conn._conn.commit.assert_called()

    @pytest.mark.asyncio
    async def test_fail_job_permanently(self, queue_with_mock_conn, mock_cursor):
        """Test job fails permanently after max attempts."""
        mock_cursor.fetchone.return_value = {"attempts": 3, "max_attempts": 3}

        await queue_with_mock_conn.fail_job(1, "Final Error")

        queue_with_mock_conn._conn.commit.assert_called()

    @pytest.mark.asyncio
    async def test_get_stats(self, queue_with_mock_conn, mock_cursor):
        """Test getting queue statistics."""
        mock_cursor.fetchall.return_value = [
            {"status": "pending", "count": 10},
            {"status": "processing", "count": 2},
            {"status": "completed", "count": 100},
            {"status": "failed", "count": 5},
        ]
        mock_cursor.fetchone.return_value = {"count": 8}

        stats = await queue_with_mock_conn.get_stats()

        assert stats["pending"] == 10
        assert stats["processing"] == 2
        assert stats["completed"] == 100
        assert stats["failed"] == 5
        assert stats["ready_to_process"] == 8

    @pytest.mark.asyncio
    async def test_get_job_status(self, queue_with_mock_conn, mock_cursor):
        """Test getting job status for a dataset."""
        now = datetime.now(timezone.utc)
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "status": "completed",
            "attempts": 1,
            "created_at": now,
            "completed_at": now,
            "result": {"pathology": ["Healthy"]},
            "error": None,
        }

        status = await queue_with_mock_conn.get_job_status("ds001234")

        assert status is not None
        assert status["status"] == "completed"
        assert status["result"]["pathology"] == ["Healthy"]

    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self, queue_with_mock_conn, mock_cursor):
        """Test getting status for non-existent job."""
        mock_cursor.fetchone.return_value = None

        status = await queue_with_mock_conn.get_job_status("ds999999")

        assert status is None


class TestTaggingJobDataclass:
    """Tests for TaggingJob dataclass."""

    def test_tagging_job_creation(self):
        """Test creating a TaggingJob."""
        now = datetime.now(timezone.utc)

        job = TaggingJob(
            id=1,
            dataset_id="ds001234",
            source_url="https://github.com/test/ds001234",
            metadata_snapshot={"title": "Test"},
            metadata_hash="abc123",
            status="pending",
            attempts=0,
            max_attempts=3,
            created_at=now,
        )

        assert job.id == 1
        assert job.dataset_id == "ds001234"
        assert job.status == "pending"
        assert job.started_at is None
        assert job.error is None
