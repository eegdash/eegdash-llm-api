"""API key gate and git host allowlist."""
import threading
import time
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


# --- Review follow-ups -------------------------------------------------------

KEY = {"X-API-Key": "s3cret"}


@pytest.mark.parametrize("url", [
    # userinfo: the real host is after the '@'
    "https://github.com@169.254.169.254/OpenNeuroDatasets/ds1",
    "https://user:pw@github.com/OpenNeuroDatasets/ds1",
    # host suffix / lookalike hosts
    "https://github.com.evil.com/OpenNeuroDatasets/ds1",
    "https://evilgithub.com/OpenNeuroDatasets/ds1",
    # explicit port
    "https://github.com:8443/OpenNeuroDatasets/ds1",
    # percent-encoded dot segments, any case
    "https://github.com/OpenNeuroDatasets/%2e%2e/evil",
    "https://github.com/OpenNeuroDatasets/%2E%2E/evil",
    "https://github.com/OpenNeuroDatasets/.%2e/evil",
    # bare prefix, no repo segment
    "https://github.com/OpenNeuroDatasets",
    "https://github.com/OpenNeuroDatasets/",
    "",
])
def test_source_url_bypasses_rejected(monkeypatch, url):
    monkeypatch.delenv("TAGGER_ALLOWED_GIT_PREFIXES", raising=False)
    assert is_allowed_source_url(url) is False


def test_source_url_host_is_case_insensitive(monkeypatch):
    monkeypatch.delenv("TAGGER_ALLOWED_GIT_PREFIXES", raising=False)
    assert is_allowed_source_url("https://GitHub.COM/OpenNeuroDatasets/ds1") is True


@pytest.mark.parametrize("prefix", ["https://gin.g-node.org/", "https://gin.g-node.org"])
def test_userinfo_bypass_of_configured_host(monkeypatch, prefix):
    monkeypatch.setenv("TAGGER_ALLOWED_GIT_PREFIXES", prefix)
    assert is_allowed_source_url("https://gin.g-node.org/org/repo") is True
    assert is_allowed_source_url(
        "https://gin.g-node.org@169.254.169.254/latest/meta-data"
    ) is False
    assert is_allowed_source_url("https://gin.g-node.org.evil.com/org/repo") is False


def test_prefix_without_trailing_slash_matches_segments_only(monkeypatch):
    monkeypatch.setenv(
        "TAGGER_ALLOWED_GIT_PREFIXES", "https://github.com/OpenNeuroDatasets"
    )
    assert is_allowed_source_url("https://github.com/OpenNeuroDatasets/ds1") is True
    assert is_allowed_source_url("https://github.com/OpenNeuroDatasetsEvil/x") is False


def test_dataset_id_rejects_trailing_newline():
    assert is_valid_dataset_id("ds002718\n") is False


def test_tag_rejects_traversal_dataset_id(client):
    r = client.post("/api/v1/tag", headers=KEY, json={
        "dataset_id": "../x", "source_url": OPENNEURO,
    })
    assert r.status_code == 422


@pytest.mark.parametrize("url", [
    "https://nemar.org/dataexplorer/detail?dataset_id=ds002718",
    "https://openneuro.org/datasets/ds002718",
])
def test_enqueue_accepts_landing_page_urls(client, monkeypatch, url):
    """The worker never clones, so enqueue must not apply the clone allowlist.

    With the queue off, passing validation means 503 (not 422).
    """
    monkeypatch.setattr("src.api.main.queue_enabled", False)
    monkeypatch.setattr("src.api.main.queue", None)
    item = {"dataset_id": "ds002718", "source_url": url, "metadata_snapshot": {}}
    assert client.post("/api/v1/tag/enqueue", headers=KEY, json=item).status_code == 503
    r = client.post("/api/v1/tag/enqueue/batch", headers=KEY, json={"datasets": [item]})
    assert r.status_code == 503


def test_enqueue_still_validates_dataset_id(client):
    r = client.post("/api/v1/tag/enqueue", headers=KEY, json={
        "dataset_id": "../x", "source_url": OPENNEURO, "metadata_snapshot": {},
    })
    assert r.status_code == 422


@pytest.mark.parametrize("url,dataset_id", [
    ("https://evil.example.com/repo.git", "ds1"),
    (OPENNEURO, "../x"),
])
def test_clone_repo_guards_before_subprocess(monkeypatch, tmp_path, url, dataset_id):
    from src.services.orchestrator import TaggingOrchestrator
    monkeypatch.delenv("TAGGER_ALLOWED_GIT_PREFIXES", raising=False)
    orch = TaggingOrchestrator(tagger=None, cache=None)
    with patch("src.services.orchestrator.subprocess.run") as run:
        with pytest.raises(RuntimeError):
            orch._clone_repo(url, dataset_id, str(tmp_path))
        run.assert_not_called()


def test_clone_repo_disables_redirects_and_prompts(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from src.services.orchestrator import TaggingOrchestrator
    monkeypatch.delenv("TAGGER_ALLOWED_GIT_PREFIXES", raising=False)
    orch = TaggingOrchestrator(tagger=None, cache=None)
    with patch("src.services.orchestrator.subprocess.run") as run:
        run.return_value = Mock(returncode=0, stderr=b"")
        orch._clone_repo(OPENNEURO, "ds002718", str(tmp_path))
    args, kwargs = run.call_args
    cmd = args[0]
    assert cmd[:3] == ["git", "-c", "http.followRedirects=false"]
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_oversized_keyless_body_gets_413(client):
    body = b'{"x": "' + b"a" * (5 * 1024 * 1024) + b'"}'
    r = client.post(
        "/api/v1/tag", content=body, headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 413


class _SlowTag:
    """Blocking stand-in for orchestrator.tag; records peak concurrency."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.peak = 0
        self._mu = threading.Lock()

    def __call__(self, **kwargs):
        with self._mu:
            self.active += 1
            self.peak = max(self.peak, self.active)
        self.started.set()
        self.release.wait(timeout=5)
        time.sleep(0.05)
        with self._mu:
            self.active -= 1
        return {"pathology": ["Healthy"], "modality": ["Visual"], "type": ["Perception"]}


def _post_tag(client):
    client.post("/api/v1/tag", headers=KEY, json={
        "dataset_id": "ds002718", "source_url": OPENNEURO,
    })


def test_health_answers_while_sync_tag_runs(client, monkeypatch):
    import src.api.main as main
    slow = _SlowTag()
    monkeypatch.setattr(main.orchestrator, "tag", slow)

    tagger = threading.Thread(target=_post_tag, args=(client,))
    tagger.start()
    assert slow.started.wait(timeout=5)

    health = threading.Thread(target=client.get, args=("/health",))
    health.start()
    health.join(timeout=2)
    finished = not health.is_alive()
    slow.release.set()
    tagger.join(timeout=10)
    health.join(timeout=10)
    assert finished, "/health blocked while a sync tag was running"


def test_sync_tags_are_serialized(client, monkeypatch):
    import src.api.main as main
    slow = _SlowTag()
    monkeypatch.setattr(main.orchestrator, "tag", slow)

    threads = [threading.Thread(target=_post_tag, args=(client,)) for _ in range(3)]
    for t in threads:
        t.start()
    assert slow.started.wait(timeout=5)
    time.sleep(0.2)
    slow.release.set()
    for t in threads:
        t.join(timeout=10)
    assert slow.peak == 1
