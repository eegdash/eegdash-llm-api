"""Opt-in real PostgreSQL plus loopback HTTP integration; never contacts a model.

Set TEST_POSTGRES_URL to a disposable test database. Each test creates and drops
only its unique schema. A local HTTP fixture supplies both model and catalogue.
"""
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import pytest

from src.services.worker import TaggingWorker

pytestmark = pytest.mark.skipif(not os.getenv("TEST_POSTGRES_URL"), reason="requires disposable TEST_POSTGRES_URL")


async def test_real_queue_retries_model_and_write_failures_then_completes(tmp_path, monkeypatch):
    state = {"model_calls": 0, "write_calls": 0, "writes": [], "outage": False}
    labels = {
        "pathology": ["Healthy"], "modality": ["Visual"], "type": ["Perception"],
        "confidence": {"pathology": 0.8, "modality": 0.8, "type": 0.8},
        "reasoning": {"decision_summary": "Synthetic integration fixture"},
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            assert self.path == "/api/eegdash/datasets/fixture-dataset"
            self.reply(200, {"success": True, "data": {"dataset_id": "fixture-dataset"}})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/chat/completions":
                state["model_calls"] += 1
                if state["model_calls"] == 1 or state["outage"]:
                    self.reply(503, {"error": "intentional fixture outage"})
                else:
                    self.reply(200, {"model": "fixture-reported-model", "id": "fixture-response-id",
                                     "choices": [{"message": {"content": json.dumps(labels)}}]})
            elif self.path == "/admin/eegdash/datasets":
                state["write_calls"] += 1
                if state["write_calls"] == 1:
                    self.reply(503, {"error": "intentional fixture write failure"})
                else:
                    state["writes"].append(payload)
                    self.reply(200, {"success": True, "updated_count": 1, "inserted_count": 0})
            else:
                self.reply(404, {})

    schema = "eegdash_test_" + uuid.uuid4().hex
    admin = await psycopg.AsyncConnection.connect(os.environ["TEST_POSTGRES_URL"], autocommit=True)
    await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    connection = make_conninfo(os.environ["TEST_POSTGRES_URL"], options=f"-csearch_path={schema}")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("LLM_API_BASE", base_url + "/v1")
    monkeypatch.delenv("FEW_SHOT_PATH", raising=False)
    monkeypatch.delenv("PROMPT_PATH", raising=False)
    worker = TaggingWorker(connection, eegdash_api_url=base_url,
                          eegdash_admin_token="fixture-token", openrouter_api_key="fixture-token",
                          model="fixture-model", cache_path=tmp_path / "cache.json")
    try:
        await worker.initialize()
        worker._queue.RETRY_DELAYS = [0, 0, 0]
        await worker._queue.enqueue("fixture-dataset", "https://example.invalid/dataset",
                                   {"title": "Fixture", "readme": "Synthetic healthy visual study."})
        for attempt in range(1, 4):
            job = await worker._queue.claim_job("integration-worker")
            assert job is not None and job.attempts == attempt
            await worker._process_job(job)
            status = await worker._queue.get_job_status("fixture-dataset")
            assert status["status"] == ("completed" if attempt == 3 else "pending")
            if attempt == 1:
                assert worker._cache.stats()["total_entries"] == 0
                assert state["write_calls"] == 0
            elif attempt == 2:
                assert worker._cache.stats()["total_entries"] == 1
                assert status["completed_at"] is None
        assert state["model_calls"] == 2  # failed call + success; write retry reuses cache
        assert state["write_calls"] == 2
        assert len(state["writes"]) == 1
        assert state["writes"][0]["tags"]["pathology"] == ["Healthy"]
        provenance = state["writes"][0]["tagger_meta"]["request_provenance"]
        assert provenance["model"] == "fixture-model"
        assert provenance["provider_response_model"] == "fixture-reported-model"
        assert provenance["provider_response_id"] == "fixture-response-id"
        assert len(provenance["request_hash"]) == 64
        assert status["result"]["tags"]["dataset_id"] == "fixture-dataset"
        assert "dataset_id" not in status["result"]["tags"]["input_metadata"]
        assert status["result"]["tags"]["tagger_meta"] == provenance
        assert status["result"]["tags"]["input_metadata"]["readme"] == "Synthetic healthy visual study."
        assert status["result"]["from_cache"] is True
        assert status["result"]["mongodb_updated"] is True

        # A continuing outage consumes the bounded retry budget and cannot be
        # reported as successful completion or write a placeholder Unknown tag.
        state["outage"] = True
        await worker._queue.enqueue("fixture-terminal", "https://example.invalid/terminal",
                                   {"readme": "Different synthetic metadata for terminal failure."})
        for attempt in range(1, 4):
            failed_job = await worker._queue.claim_job("integration-worker")
            assert failed_job.dataset_id == "fixture-terminal" and failed_job.attempts == attempt
            await worker._process_job(failed_job)
        failed_status = await worker._queue.get_job_status("fixture-terminal")
        assert failed_status["status"] == "failed"
        assert failed_status["result"] is None
        assert failed_status["error"]
        assert state["write_calls"] == 2
        assert worker._cache.stats()["total_entries"] == 1
        assert await worker._queue.claim_job("integration-worker") is None
    finally:
        await worker.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        await admin.close()
