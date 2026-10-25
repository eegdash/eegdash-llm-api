"""Shared cache identity for the effective tagger configuration."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from eegdash_tagger.tagging import OpenRouterTagger
from eegdash_tagger.tagging.taxonomy import LABELS


def config_hash_for_tagger(tagger: OpenRouterTagger) -> str:
    """Hash the prompt actually read and examples actually loaded by this tagger.

    Recompute before each request/job: the library reads its prompt file on each
    call, whereas examples remain the instance's filtered, loaded examples.
    API keys are deliberately excluded. A new hash namespace also isolates
    caches written by releases that stored failed calls as Unknown results.
    """
    identity = {
        "cache_schema": 2,
        "library_version": version("eegdash-tagger"),
        "prompt": tagger._build_system_prompt(),
        "examples": tagger.few_shot_examples,
        "generation": {key: value for key, value in
                       tagger._build_request_payload("", "").items() if key != "messages"},
        "endpoint": tagger.endpoint,
        "vocabulary": LABELS,
    }
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def compute_config_hash(
    few_shot_path: Path | None, model: str, prompt_path: Path | None = None
) -> str:
    """Compute the same identity outside a running service, without an API call."""
    tagger = OpenRouterTagger(
        api_key="configuration-only",
        model=model,
        few_shot_path=few_shot_path,
        prompt_path=prompt_path,
    )
    return config_hash_for_tagger(tagger)
