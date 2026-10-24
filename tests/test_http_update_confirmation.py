"""Business failures must not become successful worker persistence."""
import httpx
import pytest
from src.services.mongodb_http_updater import MongoDBHttpUpdater


@pytest.mark.parametrize("body,expected", [
    ({"success": True, "updated_count": 1, "inserted_count": 0}, True),
    ({"success": True, "updated_count": 0, "inserted_count": 0}, True),
    ({"success": False, "updated_count": 0}, False),
    ({"success": True, "updated_count": 0, "inserted_count": 1}, False),
    ({"message": "failed"}, False),
    ({"success": True, "updated_count": "1", "inserted_count": 0}, False),
    ({"success": True, "updated_count": 1}, False),
    ({"success": True, "updated_count": 1, "inserted_count": False}, False),
    ({"success": True, "updated_count": True, "inserted_count": 0}, False),
    ({"success": True, "updated_count": 2, "inserted_count": 0}, False),
])
@pytest.mark.parametrize("method", ["update_tags", "update_fields"])
def test_http_200_requires_confirmed_write(body, expected, method):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"success": True, "data": {"dataset_id": "ds-test"}})
        return httpx.Response(200, json=body)
    updater = MongoDBHttpUpdater("https://fixture.invalid", "test-only-token")
    updater._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        if method == "update_tags":
            actual = updater.update_tags("ds-test", {}, "config", "metadata", "model")
        else:
            actual = updater.update_fields("ds-test", {"example": True})
        assert actual is expected
    finally:
        updater.close()


def test_http_write_preserves_nested_request_provenance_and_reasoning():
    import json
    writes = []
    provenance = {"request_hash": "abc", "provider_response_model": "reported-model"}
    reasoning = {"evidence_alignment_check": ["first", {"second": True}]}
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"success": True, "data": {"dataset_id": "ds-test"}})
        writes.append(json.loads(request.content))
        return httpx.Response(200, json={"success": True, "updated_count": 1, "inserted_count": 0})
    updater = MongoDBHttpUpdater("https://fixture.invalid", "test-only-token")
    updater._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        assert updater.update_tags("ds-test", {}, "config", "metadata", "model",
                                   reasoning=reasoning, request_provenance=provenance)
        assert writes[0]["tagger_meta"]["request_provenance"] == provenance
        assert writes[0]["tags"]["reasoning"] == reasoning
    finally:
        updater.close()
