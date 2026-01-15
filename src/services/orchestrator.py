"""
Main workflow orchestration for tagging.

This module coordinates the tagging workflow with cache-first strategy.
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from eegdash_tagger.tagging import OpenRouterTagger, ParsedMetadata
from eegdash_tagger.metadata import build_dataset_summary_from_path
from eegdash_tagger.scraping import extract_dois_from_references, fetch_abstract_with_cache

from .cache import TaggingCache


class TaggingOrchestrator:
    """
    Orchestrates the tagging workflow with cache-first strategy.

    Flow:
    1. Clone repository
    2. Extract BIDS metadata
    3. Fetch paper abstracts
    4. Check cache
    5. On miss: call LLM, store result
    6. Return result
    """

    def __init__(
        self,
        tagger: OpenRouterTagger,
        cache: TaggingCache,
        allow_on_demand: bool = True,
        serve_stale_on_error: bool = True,
        clone_timeout: int = 120,
        abstract_cache_path: Path = None
    ):
        """
        Initialize the orchestrator.

        Args:
            tagger: OpenRouterTagger instance for LLM calls
            cache: TaggingCache instance for result caching
            allow_on_demand: If False, only serve cached results
            serve_stale_on_error: Serve stale results on LLM errors
            clone_timeout: Timeout for git clone in seconds
            abstract_cache_path: Path to abstract cache file
        """
        self.tagger = tagger
        self.cache = cache
        self.allow_on_demand = allow_on_demand
        self.serve_stale_on_error = serve_stale_on_error
        self.clone_timeout = clone_timeout
        self.abstract_cache_path = abstract_cache_path or Path("data/cache/abstract_cache.json")

    def tag(
        self,
        dataset_id: str,
        source_url: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Tag a dataset with cache-first strategy.

        Args:
            dataset_id: Dataset identifier (e.g., "ds004398")
            source_url: Git repository URL
            force_refresh: Skip cache lookup and force new LLM call

        Returns:
            Tagging result dict with keys:
            - dataset_id, pathology, modality, type, confidence, reasoning
            - from_cache: bool
            - stale: bool (if serving stale result)
            - cache_key: str (for debugging)
            - error: str (if error occurred)
        """
        # Clone and extract metadata
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                repo_path = self._clone_repo(source_url, dataset_id, tmpdir)
                metadata = self._extract_metadata(repo_path)
            except Exception as e:
                # Clone failed - try to serve stale result
                if self.serve_stale_on_error:
                    stale = self.cache.get_any_for_dataset(dataset_id)
                    if stale:
                        return {
                            **stale["result"],
                            "dataset_id": dataset_id,
                            "from_cache": True,
                            "stale": True,
                            "error": f"Clone failed: {str(e)}"
                        }
                return self._error_response(dataset_id, f"Clone failed: {str(e)}")

        # Check for sufficient data
        if not metadata.get('readme') and not metadata.get('dataset_description'):
            return self._insufficient_data_response(dataset_id)

        # Build cache key
        cache_key = self.cache.build_key(
            dataset_id=dataset_id,
            metadata=metadata,
            model=self.tagger.model
        )

        # Check cache (unless force refresh)
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return {
                    **cached,
                    "dataset_id": dataset_id,
                    "from_cache": True,
                    "cache_key": cache_key
                }

        # Cache miss - check if on-demand allowed
        if not self.allow_on_demand:
            stale = self.cache.get_any_for_dataset(dataset_id)
            if stale:
                return {
                    **stale["result"],
                    "dataset_id": dataset_id,
                    "from_cache": True,
                    "stale": True
                }
            return self._cache_miss_response(dataset_id)

        # Call LLM
        try:
            parsed = self._build_parsed_metadata(metadata)
            result = self.tagger.tag_with_details(parsed, dataset_id)

            # Store in cache
            self.cache.set(cache_key, result, metadata)

            return {
                **result,
                "dataset_id": dataset_id,
                "from_cache": False,
                "cache_key": cache_key
            }

        except Exception as e:
            if self.serve_stale_on_error:
                stale = self.cache.get_any_for_dataset(dataset_id)
                if stale:
                    return {
                        **stale["result"],
                        "dataset_id": dataset_id,
                        "from_cache": True,
                        "stale": True,
                        "error": f"LLM call failed: {str(e)}"
                    }
            return self._error_response(dataset_id, f"LLM call failed: {str(e)}")

    def _clone_repo(self, source_url: str, dataset_id: str, tmpdir: str) -> str:
        """
        Shallow clone repository.

        Args:
            source_url: Git repository URL
            dataset_id: Dataset identifier (used as directory name)
            tmpdir: Temporary directory path

        Returns:
            Path to cloned repository

        Raises:
            RuntimeError: If clone fails
        """
        repo_path = os.path.join(tmpdir, dataset_id)
        env = os.environ.copy()
        env['GIT_LFS_SKIP_SMUDGE'] = '1'

        result = subprocess.run(
            ['git', 'clone', '--depth', '1', source_url, repo_path],
            env=env,
            capture_output=True,
            timeout=self.clone_timeout
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())

        return repo_path

    def _extract_metadata(self, repo_path: str) -> Dict[str, Any]:
        """
        Extract BIDS metadata and fetch paper abstracts.

        Args:
            repo_path: Path to cloned repository

        Returns:
            Metadata dict ready for tagging
        """
        # Parse BIDS structure
        summary = build_dataset_summary_from_path(repo_path)
        metadata = summary.to_llm_json()

        # Fetch paper abstracts from DOIs
        paper_abstract = ""
        if summary.dataset_description:
            dois = extract_dois_from_references(summary.dataset_description, None)
            for doi in dois[:2]:  # Max 2 abstracts
                abstract = fetch_abstract_with_cache(doi, self.abstract_cache_path)
                if abstract:
                    paper_abstract += f"\n\n{abstract}"

        metadata['paper_abstract'] = paper_abstract.strip()
        return metadata

    def _build_parsed_metadata(self, metadata: Dict[str, Any]) -> ParsedMetadata:
        """
        Convert metadata dict to ParsedMetadata TypedDict.

        Args:
            metadata: Extracted metadata dict

        Returns:
            ParsedMetadata instance
        """
        return ParsedMetadata(
            title=metadata.get('title', ''),
            dataset_description=metadata.get('dataset_description', ''),
            readme=metadata.get('readme', ''),
            participants_overview=metadata.get('participants_overview', ''),
            tasks=metadata.get('tasks', []),
            events=metadata.get('events', []),
            paper_abstract=metadata.get('paper_abstract', '')
        )

    def _insufficient_data_response(self, dataset_id: str) -> Dict[str, Any]:
        """Response when dataset has insufficient metadata."""
        return {
            "dataset_id": dataset_id,
            "pathology": ["Unknown"],
            "modality": ["Unknown"],
            "type": ["Unknown"],
            "confidence": {"pathology": 0, "modality": 0, "type": 0},
            "reasoning": {},
            "from_cache": False,
            "error": "Insufficient metadata - no readme or dataset_description found"
        }

    def _cache_miss_response(self, dataset_id: str) -> Dict[str, Any]:
        """Response when cache miss and on-demand disabled."""
        return {
            "dataset_id": dataset_id,
            "pathology": ["Unknown"],
            "modality": ["Unknown"],
            "type": ["Unknown"],
            "confidence": {"pathology": 0, "modality": 0, "type": 0},
            "reasoning": {},
            "from_cache": False,
            "error": "Cache miss and on-demand tagging disabled"
        }

    def _error_response(self, dataset_id: str, error: str) -> Dict[str, Any]:
        """Generic error response."""
        return {
            "dataset_id": dataset_id,
            "pathology": ["Unknown"],
            "modality": ["Unknown"],
            "type": ["Unknown"],
            "confidence": {"pathology": 0, "modality": 0, "type": 0},
            "reasoning": {},
            "from_cache": False,
            "error": error
        }
