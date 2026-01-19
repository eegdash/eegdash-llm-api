"""
Shared fixtures and mocks for API tests.
"""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "title": "Visual Perception EEG Study",
        "dataset_description": "Name: Visual Study\nAuthors: Test Author",
        "readme": "This study investigates visual perception in healthy adults.",
        "participants_overview": "age: [20-30], sex: [M, F]",
        "tasks": ["visual", "rest"],
        "events": ["stimulus_onset", "response"],
        "paper_abstract": ""
    }


@pytest.fixture
def sample_tagging_result():
    """Sample tagging result returned by OpenRouterTagger."""
    return {
        "pathology": ["Healthy"],
        "modality": ["Visual"],
        "type": ["Perception"],
        "confidence": {"pathology": 0.9, "modality": 0.85, "type": 0.8},
        "reasoning": {
            "few_shot_analysis": "Matched visual perception pattern",
            "metadata_analysis": "README mentions visual stimuli",
            "paper_abstract_analysis": "No abstract provided",
            "decision_summary": "Clear visual perception study"
        }
    }


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_openrouter_tagger(sample_tagging_result):
    """Mock OpenRouterTagger to avoid API calls."""
    mock_tagger = Mock()
    mock_tagger.model = "openai/gpt-4-turbo"
    mock_tagger.tag_with_details.return_value = sample_tagging_result
    return mock_tagger


@pytest.fixture
def mock_git_clone():
    """Mock git clone operation."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0, stderr=b"")
        yield mock_run


@pytest.fixture
def mock_dataset_summary(sample_metadata):
    """Mock DatasetSummary returned by build_dataset_summary_from_path."""
    mock_summary = Mock()
    mock_summary.to_llm_json.return_value = sample_metadata
    mock_summary.dataset_description = sample_metadata.get("dataset_description", "")
    return mock_summary


@pytest.fixture
def mock_metadata_extraction(mock_dataset_summary):
    """Mock the metadata extraction from cloned repo."""
    with patch('src.services.orchestrator.build_dataset_summary_from_path') as mock_build:
        mock_build.return_value = mock_dataset_summary
        yield mock_build


@pytest.fixture
def mock_abstract_fetcher():
    """Mock abstract fetching to avoid network calls."""
    with patch('src.services.orchestrator.extract_dois_from_references') as mock_dois, \
         patch('src.services.orchestrator.fetch_abstract_with_cache') as mock_abstract:
        mock_dois.return_value = []
        mock_abstract.return_value = ""
        yield  # Don't need to return anything, patches are active during yield


@pytest.fixture
def mock_tagger_class(sample_tagging_result, temp_cache_dir):
    """Mock OpenRouterTagger class with class methods."""
    mock_class = Mock()

    # Mock instance
    mock_instance = Mock()
    mock_instance.model = "openai/gpt-4-turbo"
    mock_instance.tag_with_details.return_value = sample_tagging_result
    mock_class.return_value = mock_instance

    # Mock class methods - return real paths in temp dir
    few_shot_path = temp_cache_dir / "few_shot_examples.json"
    prompt_path = temp_cache_dir / "prompt.md"

    # Create dummy files
    few_shot_path.write_text('{"examples": []}')
    prompt_path.write_text("Test prompt")

    mock_class.get_default_few_shot_path.return_value = few_shot_path
    mock_class.get_default_prompt_path.return_value = prompt_path

    return mock_class
