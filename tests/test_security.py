"""API key gate and git host allowlist."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, require_api_key
from src.services.orchestrator import is_allowed_source_url, is_valid_dataset_id

OPENNEURO = "https://github.com/OpenNeuroDatasets/ds002718"


@pytest.fixture
def client(temp_cache_dir, mock_tagger_class, monkeypatch):
    app.dependency_overrides.pop(require_api_key, None)
    monkeypatch.delenv("TAGGER_ALLOWED_GIT_PREFIXES", raising=False)
    monkeypatch.setenv("TAGGER_API_KEY", "s3cret")
    with patch.dict("os.environ", {
        "OPENROUTER_API_KEY": "test-key-12345",
        "CACHE_DIR": str(temp_cache_dir),
    }), patch("src.api.main.OpenRouterTagger", mock_tagger_class):
        with TestClient(app) as c:
            yield c


@pytest.mark.parametrize("method,path", [
    ("post", "/api/v1/tag"),
    ("post", "/api/v1/tag/enqueue"),
    ("post", "/api/v1/tag/enqueue/batch"),
    ("delete", "/api/v1/cache"),
    ("delete", "/api/v1/cache/some-key"),
])
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_protected_routes_reject_missing_or_wrong_key(client, method, path, headers):
    r = getattr(client, method)(path, headers=headers)
    assert r.status_code == 401


def test_unconfigured_key_fails_closed(client, monkeypatch):
    monkeypatch.delenv("TAGGER_API_KEY")
    r = client.delete("/api/v1/cache", headers={"X-API-Key": "anything"})
    assert r.status_code == 503


def test_correct_key_passes_gate(client):
    r = client.delete("/api/v1/cache", headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200


def test_reads_stay_public(client):
    assert client.get("/health").status_code == 200


def test_disallowed_source_url_rejected(client):
    r = client.post("/api/v1/tag", headers={"X-API-Key": "s3cret"}, json={
        "dataset_id": "ds1", "source_url": "https://evil.example.com/repo.git",
    })
    assert r.status_code == 422


@pytest.mark.parametrize("url,ok", [
    (OPENNEURO, True),
    ("https://github.com/nemarDatasets/nm000104", True),
    ("https://github.com/nemardatasets/nm000104", True),
    ("https://github.com/OpenNeuroDatasets/../evil/x", False),
    ("http://github.com/OpenNeuroDatasets/ds1", False),
    ("file:///etc", False),
    ("--upload-pack=touch /tmp/x", False),
    ("https://github.com/OpenNeuroDatasetsEvil/x", False),
])
def test_source_url_allowlist(monkeypatch, url, ok):
    monkeypatch.delenv("TAGGER_ALLOWED_GIT_PREFIXES", raising=False)
    assert is_allowed_source_url(url) is ok


@pytest.mark.parametrize("dataset_id,ok", [
    ("ds002718", True), ("nm000104", True), ("EEG.Many-Labs_1", True),
    ("../etc", False), ("a/b", False), ("", False), ("..", False),
])
def test_dataset_id_validation(dataset_id, ok):
    assert is_valid_dataset_id(dataset_id) is ok
