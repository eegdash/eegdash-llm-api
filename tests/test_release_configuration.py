"""The deployment configuration must describe the same release as the library."""
import json
from pathlib import Path

from eegdash_tagger.tagging import OpenRouterTagger


def test_vendored_configuration_matches_installed_library():
    root = Path(__file__).resolve().parents[1]
    vendor = root / "deploy" / "vendor-config"
    assert (vendor / "prompt.md").read_bytes() == OpenRouterTagger.get_default_prompt_path().read_bytes()
    assert json.loads((vendor / "few_shot_examples.json").read_text()) == json.loads(OpenRouterTagger.get_default_few_shot_path().read_text())
    assert len(OpenRouterTagger(api_key="configuration-only").few_shot_examples) == 15
