"""Release regressions exercising the real worker, cache and API error paths."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import main as api
from src.services.cache import TaggingCache
from src.services.config import config_hash_for_tagger
from src.services.orchestrator import TaggingOrchestrator
from src.services.queue import TaggingJob, TaggingQueue
from src.services.worker import TaggingWorker


@pytest.fixture
def worker(tmp_path, mock_openrouter_tagger):
    worker = TaggingWorker("postgresql://unused", mongodb_url="mongodb://unused")
    worker._tagger = mock_openrouter_tagger
    worker._queue = Mock(spec=TaggingQueue)
    worker._queue.complete_job = AsyncMock()
    worker._queue.fail_job = AsyncMock()
    worker._queue.compute_metadata_hash = TaggingQueue.compute_metadata_hash
    worker._mongodb = Mock()
    worker._mongodb.update_tags.return_value = True
    worker._config_hash = config_hash_for_tagger(worker._tagger)
    worker._cache = TaggingCache(tmp_path / "cache.json", worker._config_hash)
    return worker


@pytest.fixture
def job(sample_metadata):
    return TaggingJob(1, "ds-test", "https://example.invalid/dataset", sample_metadata,
                      TaggingQueue.compute_metadata_hash(sample_metadata), "processing",
                      1, 3, datetime.now(timezone.utc))


@pytest.mark.parametrize("cached", ["none", "regular", "ground_truth"])
async def test_failed_write_is_retried_and_never_completed(worker, job, sample_tagging_result, cached):
    if cached == "regular":
        key = worker._cache.build_key(job.dataset_id, job.metadata_snapshot, worker.model)
        worker._cache.set(key, sample_tagging_result, job.metadata_snapshot)
    elif cached == "ground_truth":
        worker._cache.set_ground_truth(job.dataset_id, sample_tagging_result)
    worker._mongodb.update_tags.return_value = False
    await worker._process_job(job)
    worker._queue.complete_job.assert_not_awaited()
    worker._queue.fail_job.assert_awaited_once()
    assert "Catalogue write failed" in worker._queue.fail_job.call_args.args[1]
    if cached != "none":
        worker._tagger.tag_with_details.assert_not_called()


async def test_model_failure_is_not_cached_or_written(worker, job):
    worker._tagger.tag_with_details.side_effect = RuntimeError("simulated transport failure")
    await worker._process_job(job)
    worker._mongodb.update_tags.assert_not_called()
    worker._queue.complete_job.assert_not_awaited()
    worker._queue.fail_job.assert_awaited_once()
    assert worker._cache.stats()["total_entries"] == 0


async def test_valid_unknown_is_persisted_and_completed(worker, job):
    worker._tagger.tag_with_details.return_value = {
        **{axis: ["Unknown"] for axis in ("pathology", "modality", "type")},
        "confidence": {axis: 0 for axis in ("pathology", "modality", "type")},
        "reasoning": {"decision_summary": "Insufficient evidence"},
    }
    await worker._process_job(job)
    worker._mongodb.update_tags.assert_called_once()
    worker._queue.complete_job.assert_awaited_once()
    worker._queue.fail_job.assert_not_awaited()
    assert worker._cache.stats()["total_entries"] == 1


async def test_write_retry_reuses_successful_inference(worker, job):
    worker._mongodb.update_tags.side_effect = [False, True]
    await worker._process_job(job)
    await worker._process_job(job)
    worker._tagger.tag_with_details.assert_called_once()
    worker._queue.fail_job.assert_awaited_once()
    worker._queue.complete_job.assert_awaited_once()


@pytest.mark.parametrize("change", ["prompt", "examples", "endpoint", "model", "max_tokens"])
def test_configuration_identity_tracks_effective_inputs(mock_openrouter_tagger, change):
    before = config_hash_for_tagger(mock_openrouter_tagger)
    if change == "prompt":
        mock_openrouter_tagger._build_system_prompt.return_value = "Changed instruction"
    elif change == "examples":
        mock_openrouter_tagger.few_shot_examples = [{"pathology": ["Healthy"]}]
    elif change == "endpoint":
        mock_openrouter_tagger.endpoint = "https://other.invalid/chat/completions"
    elif change == "max_tokens":
        mock_openrouter_tagger.max_tokens = 2000
    else:
        mock_openrouter_tagger.model = "other-model"
    assert config_hash_for_tagger(mock_openrouter_tagger) != before


def test_prompt_edit_invalidates_running_orchestrator_cache(tmp_path, mock_openrouter_tagger, sample_metadata):
    cache = TaggingCache(tmp_path / "cache.json", config_hash_for_tagger(mock_openrouter_tagger))
    service = TaggingOrchestrator(mock_openrouter_tagger, cache)
    with patch.object(service, "_clone_repo"), patch.object(service, "_extract_metadata", return_value=sample_metadata):
        service.tag("ds-test", "https://example.invalid/dataset")
        service.tag("ds-test", "https://example.invalid/dataset")
        assert mock_openrouter_tagger.tag_with_details.call_count == 1
        mock_openrouter_tagger._build_system_prompt.return_value = "Changed prompt"
        service.tag("ds-test", "https://example.invalid/dataset")
    assert mock_openrouter_tagger.tag_with_details.call_count == 2


def test_task_details_participate_in_cache_key(tmp_path):
    cache = TaggingCache(tmp_path / "cache.json", "config")
    assert cache.build_key("ds-test", {"task_details": "A"}, "model") != cache.build_key("ds-test", {"task_details": "B"}, "model")


@pytest.mark.parametrize("failure", [
    {"pathology": ["Unknown"], "reasoning": {"decision_summary": "API call failed"}},
    {"status": "error", "pathology": ["Unknown"]},
])
def test_legacy_failure_is_never_served_as_stale(tmp_path, failure):
    cache = TaggingCache(tmp_path / "cache.json", "new-config")
    key = "ds-test:old-metadata:old-config:model"
    cache._cache[key] = {"result": failure, "timestamp": "2026-01-01"}
    assert cache.get(key) is None
    assert cache.get_any_for_dataset("ds-test") is None
    with pytest.raises(ValueError, match="cannot be cached"):
        cache.set(key, failure, {})


def test_sync_failure_returns_http_error_instead_of_successful_unknown():
    service = Mock()
    service.tag.return_value = {"error": "LLM call failed", "from_cache": False}
    with patch.object(api, "orchestrator", service):
        response = TestClient(api.app).post("/api/v1/tag", json={"dataset_id": "ds-test", "source_url": "https://example.invalid"})
    assert response.status_code == 502
    assert response.json()["detail"] == "LLM call failed"


@pytest.mark.parametrize("route", ["fresh", "cached"])
def test_api_preserves_structured_reasoning_and_request_provenance(sample_tagging_result, route):
    provenance = {"request_hash": "a" * 64, "model": "requested-model",
                  "provider_response_model": "provider-reported", "schema_version": 1}
    result = {**sample_tagging_result, "tagger_meta": provenance,
              "input_metadata": {"readme": "Fixture input"},
              "reasoning": {"evidence_alignment_check": [{"axis": "pathology"}]}}
    service = Mock()
    service.tag.return_value = result
    cache = Mock()
    cache.get_any_for_dataset.return_value = {"result": result}
    with patch.object(api, "orchestrator", service), patch.object(api, "cache", cache):
        client = TestClient(api.app)
        response = (client.post("/api/v1/tag", json={"dataset_id": "ds-test", "source_url": "https://example.invalid"})
                    if route == "fresh" else client.get("/api/v1/tags/ds-test"))
    assert response.status_code == 200
    assert response.json()["request_provenance"] == provenance
    assert response.json()["model"] == "requested-model"
    assert response.json()["input_metadata"] == result["input_metadata"]
    assert response.json()["reasoning"] == result["reasoning"]
