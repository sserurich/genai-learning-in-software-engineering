"""
Data models (dataclasses) shared across the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RepoGroupInfo:
    """Maps a repository to its study group metadata."""
    repo_name: str
    team_id: str
    group: str


@dataclass
class PullRequestRawBundle:
    """All raw GitHub API data collected for a single pull request."""
    pr: Dict[str, Any]
    commits: List[Dict[str, Any]]
    files: List[Dict[str, Any]]
    issue_comments: List[Dict[str, Any]]
    review_comments: List[Dict[str, Any]]
    reviews: List[Dict[str, Any]]
    check_runs: List[Dict[str, Any]]


@dataclass
class MiningFailure:
    """Structured record of a per-PR failure logged during mining."""
    timestamp: str
    repo: str
    pr_number: str
    stage: str
    url: str
    error_type: str
    status_code: str
    reason: str
    traceback: str
