"""
Repository miner: fetches all PR sub-resources from the GitHub API and
assembles PullRequestRawBundle objects, logging failures per sub-resource
so that a single bad PR never aborts the entire run.
"""

from __future__ import annotations

import re
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from genai_mining.config import GITHUB_API
from genai_mining.github_client import GitHubClient, RawCache
from genai_mining.models import MiningFailure, PullRequestRawBundle


def _cached_or_fetch(
    cache: RawCache,
    client: GitHubClient,
    repo: str,
    pr_number: int,
    name: str,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
) -> List[Any]:
    """Fetch a paginated PR sub-resource, serving from cache when available."""
    cache_name = f"pr_{pr_number}_{name}"
    if use_cache:
        cached = cache.read_json(repo, cache_name)
        if cached is not None:
            return cached
    data = client.paginate(url, params=params)
    cache.write_json(repo, cache_name, data)
    return data


def mine_repository(
    client: GitHubClient,
    cache: RawCache,
    repo: str,
    state: str = "all",
    use_cache: bool = True,
) -> Tuple[List[PullRequestRawBundle], List[MiningFailure]]:
    """
    Mine all PRs and related artifacts from a single repository.

    Returns ``(bundles, failures)``.  Each sub-resource (commits, files,
    comments, reviews, check runs) is fetched independently; a failure on
    one sub-resource is logged and the loop continues with empty data for
    that field rather than aborting.

    Check runs are fully paginated via ``paginate()`` so PRs with more than
    100 checks are never silently truncated.
    """
    owner_repo_url = f"{GITHUB_API}/repos/{repo}"
    failures: List[MiningFailure] = []

    if use_cache:
        cached_prs = cache.read_json(repo, "pulls")
    else:
        cached_prs = None

    if cached_prs is not None:
        pulls = cached_prs
    else:
        pulls = client.paginate(f"{owner_repo_url}/pulls", params={"state": state})
        cache.write_json(repo, "pulls", pulls)

    bundles: List[PullRequestRawBundle] = []

    for pr in pulls:
        number = pr["number"]
        print(f"[mine] {repo} PR #{number}")

        def _fetch(name: str, url: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
            return _cached_or_fetch(
                cache, client, repo, number, name, url, params, use_cache
            )

        def _log_failure(stage: str, url: str, exc: Exception) -> None:
            status_code = ""
            if isinstance(exc, RuntimeError) and exc.args:
                m = re.search(r"error (\d+):", exc.args[0])
                if m:
                    status_code = m.group(1)
            failures.append(MiningFailure(
                timestamp=datetime.now(timezone.utc).isoformat(),
                repo=repo,
                pr_number=str(number),
                stage=stage,
                url=url,
                error_type=type(exc).__name__,
                status_code=status_code,
                reason=str(exc),
                traceback=traceback.format_exc(),
            ))

        try:
            commits = _fetch("commits", f"{owner_repo_url}/pulls/{number}/commits")
        except Exception as exc:
            _log_failure("fetch_commits", f"{owner_repo_url}/pulls/{number}/commits", exc)
            commits = []

        try:
            files = _fetch("files", f"{owner_repo_url}/pulls/{number}/files")
        except Exception as exc:
            _log_failure("fetch_files", f"{owner_repo_url}/pulls/{number}/files", exc)
            files = []

        try:
            issue_comments = _fetch(
                "issue_comments", f"{owner_repo_url}/issues/{number}/comments"
            )
        except Exception as exc:
            _log_failure(
                "fetch_issue_comments",
                f"{owner_repo_url}/issues/{number}/comments",
                exc,
            )
            issue_comments = []

        try:
            review_comments = _fetch(
                "review_comments", f"{owner_repo_url}/pulls/{number}/comments"
            )
        except Exception as exc:
            _log_failure(
                "fetch_review_comments",
                f"{owner_repo_url}/pulls/{number}/comments",
                exc,
            )
            review_comments = []

        try:
            reviews = _fetch("reviews", f"{owner_repo_url}/pulls/{number}/reviews")
        except Exception as exc:
            _log_failure(
                "fetch_reviews", f"{owner_repo_url}/pulls/{number}/reviews", exc
            )
            reviews = []

        head_sha = pr.get("head", {}).get("sha")
        if head_sha:
            try:
                check_runs = _fetch(
                    "check_runs",
                    f"{owner_repo_url}/commits/{head_sha}/check-runs",
                )
            except Exception as exc:
                _log_failure(
                    "fetch_check_runs",
                    f"{owner_repo_url}/commits/{head_sha}/check-runs",
                    exc,
                )
                check_runs = []
        else:
            check_runs = []

        bundles.append(PullRequestRawBundle(
            pr=pr,
            commits=commits,
            files=files,
            issue_comments=issue_comments,
            review_comments=review_comments,
            reviews=reviews,
            check_runs=check_runs,
        ))

    return bundles, failures
