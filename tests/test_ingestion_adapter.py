"""Producer contract: evidence survives through the installed tagger serializer."""
import copy
import json
import subprocess
import sys

import pytest
from eegdash_tagger.tagging import OpenRouterTagger
from src.ingestion import build_enqueue_request


def document():
    return {"dataset_id": "fixture", "name": "Visual memory",
            "description": "A synthetic source description.",
            "external_links": {"source_url": "https://example.invalid/fixture"},
            "tasks": ["recognition"], "events": ["face"],
            "task_details": {"recognition": {"TaskDescription": "Remember visual faces."}},
            "tags": {"pathology": ["Healthy"]}, "dxStatus": "Healthy",
            "study_domain": "Motor", "tagger_meta": {"model": "old-model"}}


def test_source_evidence_survives_actual_model_serializer_without_predictions():
    data = document()
    before = copy.deepcopy(data)
    request = build_enqueue_request(data)
    model_input = json.loads(OpenRouterTagger(api_key="offline-fixture")._build_user_message(request["metadata_snapshot"]))["dataset"]
    assert model_input == request["metadata_snapshot"]
    assert model_input["title"] == data["name"]
    assert model_input["dataset_description"] == data["description"]
    assert model_input["task_details"] == data["task_details"]
    assert not set(model_input) & {"tags", "dxStatus", "study_domain", "tagger_meta", "dataset_id"}
    assert data == before
    request["metadata_snapshot"]["tasks"].append("another")
    assert data == before


def test_canonical_fields_take_precedence_and_url_override_is_explicit():
    data = document() | {"title": "Canonical", "dataset_description": "Canonical description"}
    request = build_enqueue_request(data, source_url="https://example.invalid/explicit")
    assert request["metadata_snapshot"]["title"] == "Canonical"
    assert request["metadata_snapshot"]["dataset_description"] == "Canonical description"
    assert request["source_url"].endswith("/explicit")


@pytest.mark.parametrize("patch", [
    {"dataset_id": ""}, {"external_links": {}},
    {"source_url": "file:///tmp/a", "external_links": {}},
    {"source_url": "https://user:password@example.invalid", "external_links": {}},
    {"tasks": "not-an-array"}, {"task_details": ["not-an-object"]},
    {"tasks": [""]}, {"task_details": {"task": {}}},
    {"task_details": {"bad": float("nan")}},
])
def test_invalid_payload_is_rejected(patch):
    with pytest.raises(ValueError):
        build_enqueue_request(document() | patch)


def test_no_evidence_is_not_inferred_from_tags_or_population_counts():
    with pytest.raises(ValueError, match="no supported"):
        build_enqueue_request({"dataset_id": "fixture", "source_url": "https://example.invalid",
                               "tags": {"pathology": ["Healthy"]}, "demographics": {"subjects_count": 20}})
    with pytest.raises(ValueError, match="no supported"):
        build_enqueue_request({"dataset_id": "fixture", "source_url": "https://example.invalid",
                               "title": "  "})


def test_offline_cli_writes_a_valid_batch_request(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(document()))
    run = subprocess.run([sys.executable, "-m", "src.ingestion", str(path)], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout) == {"datasets": [build_enqueue_request(document())]}
