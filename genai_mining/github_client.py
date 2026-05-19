"""
GitHub REST API client and raw JSON cache.

GitHubClient  — HTTP session with rate-limit handling, transient-error
                retry (500/502/503/504), and token-pool rotation.
RawCache      — Stores paginated API responses as JSON files for
                reproducibility without re-hitting the API.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from genai_mining.token_pool import TokenPool


class GitHubClient:
    """
    Minimal GitHub REST API client.

    Features:
    - Round-robin token rotation via TokenPool
    - Automatic rate-limit sleep + token rotation on 403
    - Exponential-backoff retry for transient 5xx errors (up to 3 attempts)
    - Pagination helper that fetches all pages from a list endpoint
    """

    _RETRYABLE_STATUS = {500, 502, 503, 504}
    _MAX_RETRIES = 3

    def __init__(self, pool: TokenPool, sleep_seconds: float = 0.2) -> None:
        self._pool = pool
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self._rotate_token()

    def _rotate_token(self) -> None:
        """Switch the session Authorization header to the next token."""
        self.session.headers.update({
            "Authorization": f"Bearer {self._pool.next()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "genai-repository-mining-pipeline",
        })

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Perform a GET request with rate-limit handling and transient-error retry.
        """
        attempt = 0
        while True:
            response = self.session.get(url, params=params)

            if response.status_code == 403 and "rate limit" in response.text.lower():
                reset = response.headers.get("X-RateLimit-Reset")
                if reset:
                    wait = max(0, int(reset) - int(time.time())) + 5
                    print(f"[rate-limit] Sleeping for {wait}s (token rotation after)...")
                    time.sleep(wait)
                self._rotate_token()
                continue

            if response.status_code in self._RETRYABLE_STATUS:
                attempt += 1
                if attempt <= self._MAX_RETRIES:
                    backoff = 2 ** attempt
                    print(
                        f"[retry] {response.status_code} on {url} "
                        f"— retrying in {backoff}s (attempt {attempt}/{self._MAX_RETRIES})"
                    )
                    time.sleep(backoff)
                    continue
                raise RuntimeError(
                    f"GitHub API error {response.status_code} after {self._MAX_RETRIES} retries: "
                    f"{response.text[:500]}"
                )

            if response.status_code >= 400:
                raise RuntimeError(
                    f"GitHub API error {response.status_code}: {response.text[:500]}"
                )

            time.sleep(self.sleep_seconds)
            return response.json()

    def paginate(self, url: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Retrieve all pages from a GitHub list endpoint.
        """
        params = dict(params or {})
        params.setdefault("per_page", 100)

        results: List[Any] = []
        page = 1

        while True:
            params["page"] = page
            data = self.get(url, params=params)

            if not isinstance(data, list):
                raise RuntimeError(
                    f"Expected list response from {url}, got {type(data)}"
                )

            if not data:
                break

            results.extend(data)
            page += 1

        return results


class RawCache:
    """
    Stores raw GitHub API responses as JSON files on disk.

    Caching enables re-running feature extraction without repeating API calls,
    which is important for reproducibility and rate-limit conservation.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def repo_dir(self, repo: str) -> Path:
        safe_repo = repo.replace("/", "__")
        path = self.root / safe_repo
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, repo: str, name: str, data: Any) -> None:
        path = self.repo_dir(repo) / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def read_json(self, repo: str, name: str) -> Optional[Any]:
        path = self.repo_dir(repo) / f"{name}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
