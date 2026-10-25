"""Offline adapter from catalogue documents to the existing tag queue contract.

This module never fetches data, calls a model, enqueues a job, or writes a catalogue.
It emits a reviewable batch request; the caller owns durable delivery and retries.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _has_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_content(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_content(item) for item in value)
    return value is not None


def build_enqueue_request(dataset: dict[str, Any], *, source_url: str | None = None) -> dict[str, Any]:
    """Map documented source fields without copying catalogue label predictions.

    Canonical nonempty fields take precedence over aliases. Task/event details and
    already-curated paper abstracts are preserved, not inferred or fetched. Empty
    evidence and missing source identifiers are rejected instead of manufactured.
    """
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an object")
    dataset_id = dataset.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a nonempty string")
    links = dataset.get("external_links") or {}
    if not isinstance(links, dict):
        raise ValueError("external_links must be an object")
    source = source_url or links.get("github_url") or links.get("source_url") or dataset.get("source_url")
    if not isinstance(source, str):
        raise ValueError("an explicit source URL is required; no source is guessed")
    parsed = urlsplit(source)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("source URL must be HTTP(S) without embedded credentials")

    fields = {
        "title": ("title", "name"),
        "dataset_description": ("dataset_description", "description"),
        "readme": ("readme",),
        "participants_overview": ("participants_overview",),
        "tasks": ("tasks",),
        "events": ("events",),
        "task_details": ("task_details",),
        "paper_abstract": ("paper_abstract",),
    }
    snapshot = {}
    for target, candidates in fields.items():
        for candidate in candidates:
            value = dataset.get(candidate)
            if value not in (None, "", [], {}):
                snapshot[target] = deepcopy(value)
                break
    for key in ("title", "dataset_description", "readme", "participants_overview", "paper_abstract"):
        if key in snapshot and not isinstance(snapshot[key], str):
            raise ValueError(f"{key} must be text")
    for key in ("tasks", "events"):
        if key in snapshot and (not isinstance(snapshot[key], list)
                                or not all(isinstance(v, str) and v.strip() for v in snapshot[key])):
            raise ValueError(f"{key} must be a list of nonempty strings")
    if "task_details" in snapshot and (not isinstance(snapshot["task_details"], dict)
                                       or not _has_content(snapshot["task_details"])):
        raise ValueError("task_details must be an object with nonempty values")
    if not _has_content(snapshot):
        raise ValueError("no supported source evidence supplied")
    request = {"dataset_id": dataset_id, "source_url": source, "metadata_snapshot": snapshot}
    # Ensure the queue payload is JSON and cannot contain NaN/Infinity.
    try:
        json.dumps(request, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("source evidence must be finite JSON values") from exc
    return request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="One dataset document or an array of documents")
    args = parser.parse_args()
    try:
        documents = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(documents, list):
            documents = [documents]
        if not documents:
            raise ValueError("at least one dataset document is required")
        requests = [build_enqueue_request(document) for document in documents]
        print(json.dumps({"datasets": requests}, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
