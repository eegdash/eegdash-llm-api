"""
Tests for the MongoDB updater service.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

from src.services.mongodb_updater import MongoDBUpdater


class TestMongoDBUpdaterBasics:
    """Tests for basic MongoDB updater operations."""

    @pytest.fixture
    def mock_collection(self):
        """Create a mock MongoDB collection."""
        return MagicMock()

    @pytest.fixture
    def updater_with_mock(self, mock_collection):
        """Create updater with mocked collection."""
        updater = MongoDBUpdater(
            "mongodb://mock",
            "eegdash",
            "datasets"
        )
        updater._collection = mock_collection
        return updater

    def test_needs_update_no_document(self, updater_with_mock, mock_collection):
        """Test needs_update returns True when document doesn't exist."""
        mock_collection.find_one.return_value = None

        result = updater_with_mock.needs_update("ds001234", "config123", "meta456")

        assert result is True

    def test_needs_update_no_tagger_meta(self, updater_with_mock, mock_collection):
        """Test needs_update returns True when no tagger_meta exists."""
        mock_collection.find_one.return_value = {"dataset_id": "ds001234"}

        result = updater_with_mock.needs_update("ds001234", "config123", "meta456")

        assert result is True

    def test_needs_update_config_changed(self, updater_with_mock, mock_collection):
        """Test needs_update returns True when config_hash differs."""
        mock_collection.find_one.return_value = {
            "dataset_id": "ds001234",
            "tagger_meta": {
                "config_hash": "old_config",
                "metadata_hash": "meta456",
            }
        }

        result = updater_with_mock.needs_update("ds001234", "new_config", "meta456")

        assert result is True

    def test_needs_update_metadata_changed(self, updater_with_mock, mock_collection):
        """Test needs_update returns True when metadata_hash differs."""
        mock_collection.find_one.return_value = {
            "dataset_id": "ds001234",
            "tagger_meta": {
                "config_hash": "config123",
                "metadata_hash": "old_meta",
            }
        }

        result = updater_with_mock.needs_update("ds001234", "config123", "new_meta")

        assert result is True

    def test_needs_update_hashes_match(self, updater_with_mock, mock_collection):
        """Test needs_update returns False when hashes match."""
        mock_collection.find_one.return_value = {
            "dataset_id": "ds001234",
            "tagger_meta": {
                "config_hash": "config123",
                "metadata_hash": "meta456",
            }
        }

        result = updater_with_mock.needs_update("ds001234", "config123", "meta456")

        assert result is False


class TestMongoDBUpdaterUpdates:
    """Tests for MongoDB update operations."""

    @pytest.fixture
    def mock_collection(self):
        """Create a mock MongoDB collection."""
        collection = MagicMock()
        collection.update_one.return_value = MagicMock(matched_count=1, modified_count=1)
        return collection

    @pytest.fixture
    def updater_with_mock(self, mock_collection):
        """Create updater with mocked collection."""
        updater = MongoDBUpdater(
            "mongodb://mock",
            "eegdash",
            "datasets"
        )
        updater._collection = mock_collection
        return updater

    def test_update_tags_success(self, updater_with_mock, mock_collection):
        """Test successful tag update."""
        tags = {
            "pathology": ["Healthy"],
            "modality": ["Visual"],
            "type": ["Perception"],
            "confidence": {"pathology": 0.9},
        }

        result = updater_with_mock.update_tags(
            dataset_id="ds001234",
            tags=tags,
            config_hash="config123",
            metadata_hash="meta456",
            model="openai/gpt-4-turbo",
        )

        assert result is True
        mock_collection.update_one.assert_called_once()

        # Check the update includes all tag fields
        call_args = mock_collection.update_one.call_args
        update_doc = call_args[0][1]
        assert "$set" in update_doc
        assert "tags" in update_doc["$set"]
        assert "tagger_meta" in update_doc["$set"]

    def test_update_tags_with_reasoning(self, updater_with_mock, mock_collection):
        """Test tag update includes reasoning."""
        tags = {
            "pathology": ["Healthy"],
            "modality": ["Visual"],
            "type": ["Perception"],
            "confidence": {},
        }
        reasoning = {
            "few_shot_analysis": "Matched example",
            "decision_summary": "Clear visual perception study",
        }

        updater_with_mock.update_tags(
            dataset_id="ds001234",
            tags=tags,
            config_hash="config123",
            metadata_hash="meta456",
            model="openai/gpt-4-turbo",
            reasoning=reasoning,
        )

        call_args = mock_collection.update_one.call_args
        update_doc = call_args[0][1]
        assert "reasoning" in update_doc["$set"]["tags"]

    def test_update_tags_not_found(self, updater_with_mock, mock_collection):
        """Test update returns False when document not found."""
        mock_collection.update_one.return_value = MagicMock(matched_count=0)

        result = updater_with_mock.update_tags(
            dataset_id="ds999999",
            tags={"pathology": ["Unknown"]},
            config_hash="config123",
            metadata_hash="meta456",
            model="openai/gpt-4-turbo",
        )

        assert result is False

    def test_update_tags_conditional_updates_when_different(self, updater_with_mock, mock_collection):
        """Test conditional update when hashes differ."""
        mock_collection.update_one.return_value = MagicMock(matched_count=1, modified_count=1)

        result = updater_with_mock.update_tags_conditional(
            dataset_id="ds001234",
            tags={"pathology": ["Healthy"]},
            config_hash="new_config",
            metadata_hash="new_meta",
            model="openai/gpt-4-turbo",
        )

        assert result is True

        # Check the filter includes hash comparison
        call_args = mock_collection.update_one.call_args
        filter_query = call_args[0][0]
        assert "$or" in filter_query

    def test_update_tags_conditional_skips_when_same(self, updater_with_mock, mock_collection):
        """Test conditional update skips when hashes match."""
        mock_collection.update_one.return_value = MagicMock(matched_count=1, modified_count=0)

        result = updater_with_mock.update_tags_conditional(
            dataset_id="ds001234",
            tags={"pathology": ["Healthy"]},
            config_hash="same_config",
            metadata_hash="same_meta",
            model="openai/gpt-4-turbo",
        )

        assert result is False


class TestMongoDBUpdaterQueries:
    """Tests for MongoDB query operations."""

    @pytest.fixture
    def mock_collection(self):
        """Create a mock MongoDB collection."""
        return MagicMock()

    @pytest.fixture
    def updater_with_mock(self, mock_collection):
        """Create updater with mocked collection."""
        updater = MongoDBUpdater(
            "mongodb://mock",
            "eegdash",
            "datasets"
        )
        updater._collection = mock_collection
        return updater

    def test_get_tags(self, updater_with_mock, mock_collection):
        """Test getting tags for a dataset."""
        mock_collection.find_one.return_value = {
            "tags": {"pathology": ["Healthy"]},
            "tagger_meta": {"config_hash": "abc123"},
        }

        result = updater_with_mock.get_tags("ds001234")

        assert result is not None
        assert result["tags"]["pathology"] == ["Healthy"]

    def test_get_tags_not_found(self, updater_with_mock, mock_collection):
        """Test getting tags for non-existent dataset."""
        mock_collection.find_one.return_value = None

        result = updater_with_mock.get_tags("ds999999")

        assert result is None

    def test_get_untagged_datasets(self, updater_with_mock, mock_collection):
        """Test finding untagged datasets."""
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = [
            {"dataset_id": "ds001", "github_url": "https://github.com/test/ds001"},
            {"dataset_id": "ds002", "github_url": "https://github.com/test/ds002"},
        ]
        mock_collection.find.return_value = mock_cursor

        result = updater_with_mock.get_untagged_datasets("new_config", limit=10)

        assert len(result) == 2
        assert result[0]["dataset_id"] == "ds001"

    def test_get_stats(self, updater_with_mock, mock_collection):
        """Test getting tagging statistics."""
        mock_collection.count_documents.side_effect = [100, 80]

        result = updater_with_mock.get_stats()

        assert result["total_datasets"] == 100
        assert result["tagged"] == 80
        assert result["untagged"] == 20


class TestMongoDBUpdaterConnection:
    """Tests for MongoDB connection management."""

    def test_not_connected_raises_error(self):
        """Test operations raise error when not connected."""
        updater = MongoDBUpdater("mongodb://mock", "eegdash", "datasets")

        with pytest.raises(RuntimeError, match="Not connected"):
            updater.needs_update("ds001234", "config", "meta")

    @patch('src.services.mongodb_updater.MongoClient')
    def test_connect(self, mock_client_class):
        """Test connection establishment."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        updater = MongoDBUpdater("mongodb://mock", "eegdash", "datasets")
        updater.connect()

        mock_client_class.assert_called_once_with("mongodb://mock")
        assert updater._client is mock_client

    @patch('src.services.mongodb_updater.MongoClient')
    def test_close(self, mock_client_class):
        """Test connection close."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        updater = MongoDBUpdater("mongodb://mock", "eegdash", "datasets")
        updater.connect()
        updater.close()

        mock_client.close.assert_called_once()
        assert updater._client is None
