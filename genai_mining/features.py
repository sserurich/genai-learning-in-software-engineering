"""
Feature extraction: converts a PullRequestRawBundle into a flat dataset row.

All text-mining helpers (keyword counters, disclosure detection, CI parsing,
suspicion heuristics) live here so they can be unit-tested independently of
the GitHub API layer.
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

from genai_mining.config import (
    DISCLOSURE_CONTEXT_PATTERNS,
    DISCLOSURE_SEEDS,
    REASONING_KEYWORDS,
    SUSPICION_MAX_DESCRIPTION_LEN,
    SUSPICION_MIN_FILES_CHANGED,
    SUSPICION_MIN_LINES_ADDED,
    SUSPICION_MIN_SIGNALS_REQUIRED,
    SUSPICION_SINGLE_COMMIT_LINES,
    TEST_FILE_PATTERNS,
    VERIFICATION_KEYWORDS,
)
from genai_mining.models import PullRequestRawBundle, RepoGroupInfo


DISCLOSURE_SEED_REGEX = {
    seed: re.compile(r"\b" + re.escape(seed) + r"\b", flags=re.IGNORECASE)
    for seed in DISCLOSURE_SEEDS
}
DISCLOSURE_CONTEXT_REGEX = [
    re.compile(pattern, flags=re.IGNORECASE) for pattern in DISCLOSURE_CONTEXT_PATTERNS
]
TEST_FILE_REGEX = [
    re.compile(pattern, flags=re.IGNORECASE) for pattern in TEST_FILE_PATTERNS
]
ALTERNATIVES_REGEX = re.compile(
    r"\b(alternative|alternatives|tradeoff|tradeoffs)\b",
    flags=re.IGNORECASE,
)

JUSTIFICATION_KEYWORDS = tuple(
    sorted({"because", "reason", "approach", "decided", "why"} & set(REASONING_KEYWORDS))
)


@lru_cache(maxsize=1024)
def _word_boundary_pattern(keyword: str) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape(keyword.lower()) + r"\b")


@lru_cache(maxsize=1024)
def _ignorecase_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE)


# =============================================================================
# Text utilities
# =============================================================================

def normalize_text(value: Optional[str]) -> str:
    return value or ""


def lower_text(value: Optional[str]) -> str:
    return normalize_text(value).lower()


def count_keywords(text: str, keywords: Iterable[str]) -> int:
    """Count keyword occurrences using word-boundary matching."""
    count = 0
    lower = text.lower()
    for keyword in keywords:
        count += len(_word_boundary_pattern(keyword).findall(lower))
    return count


def contains_any_pattern(value: str, patterns: Iterable[str]) -> bool:
    return any(_ignorecase_pattern(pattern).search(value) for pattern in patterns)


def split_repo_name(repo: str) -> str:
    """Return the repo portion of an ``owner/repo`` string."""
    return repo.split("/")[-1]


def unique_nonempty(values: Iterable[Optional[str]]) -> List[str]:
    seen: set = set()
    output: List[str] = []
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


# =============================================================================
# PR text collection
# =============================================================================

def collect_pr_text_locations(bundle: PullRequestRawBundle) -> Dict[str, str]:
    """
    Collect developer-authored text from PR artifacts, grouped by source location.
    """
    pr = bundle.pr

    pr_body = normalize_text(pr.get("body"))

    issue_text = "\n".join(
        normalize_text(c.get("body")) for c in bundle.issue_comments
    )

    review_comment_text = "\n".join(
        normalize_text(c.get("body")) for c in bundle.review_comments
    )

    review_body_text = "\n".join(
        normalize_text(r.get("body")) for r in bundle.reviews
    )

    commit_messages = "\n".join(
        normalize_text(c.get("commit", {}).get("message")) for c in bundle.commits
    )

    return {
        "PR Body": pr_body,
        "Discussion Comment": issue_text,
        "Review Comment": review_comment_text + "\n" + review_body_text,
        "Commit Message": commit_messages,
    }


# =============================================================================
# GenAI disclosure detection
# =============================================================================

def detect_ai_disclosure(text_locations: Dict[str, str]) -> Tuple[str, int, str, str]:
    """
    Detect explicit GenAI disclosure using seed keywords and contextual patterns.

    ``ai_keyword_count`` reflects only unambiguous seed-keyword hits (named
    tools).  Context patterns contribute to the ``ai_disclosed`` flag but are
    not counted, preventing double-counting when a sentence matches both a seed
    and a pattern (e.g. "using chatgpt").

    Returns:
        ai_disclosed         : "Yes" / "No"
        ai_keyword_count     : seed hits only
        ai_mention_locations : semicolon-separated location labels
        ai_tool_mentioned    : semicolon-separated tool names (sorted)
    """
    matched_locations: List[str] = []
    tools_seen: set = set()
    total_seed_count = 0
    context_pattern_matched = False

    for location, text in text_locations.items():
        location_seed_count = 0
        location_pattern_matched = False

        for seed, pattern in DISCLOSURE_SEED_REGEX.items():
            hits = len(pattern.findall(text))
            if hits:
                location_seed_count += hits
                tool_label = "ChatGPT" if seed == "chatgpt" else seed.capitalize()
                tools_seen.add(tool_label)

        for pattern in DISCLOSURE_CONTEXT_REGEX:
            if pattern.search(text):
                location_pattern_matched = True
                break

        if location_seed_count > 0 or location_pattern_matched:
            matched_locations.append(location)

        total_seed_count += location_seed_count
        context_pattern_matched = context_pattern_matched or location_pattern_matched

    ai_disclosed = "Yes" if (total_seed_count > 0 or context_pattern_matched) else "No"
    return (
        ai_disclosed,
        total_seed_count,
        ";".join(unique_nonempty(matched_locations)),
        ";".join(sorted(tools_seen)),
    )


# =============================================================================
# CI and test-file helpers
# =============================================================================

def is_test_file(filename: str) -> bool:
    return any(pattern.search(filename) for pattern in TEST_FILE_REGEX)


def extract_ci_status(check_runs: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Return ``(ci_present, ci_result)`` from a list of check-run objects.

    ci_result values: "Passed" | "Failed" | "Unavailable"
    """
    if not check_runs:
        return "No", "Unavailable"

    conclusions = [
        c.get("conclusion")
        for c in check_runs
        if c.get("conclusion") is not None
    ]

    if not conclusions:
        return "Yes", "Unavailable"

    failing = {"failure", "cancelled", "timed_out", "action_required", "startup_failure"}
    if any(c in failing for c in conclusions):
        return "Yes", "Failed"

    if all(c == "success" for c in conclusions):
        return "Yes", "Passed"

    return "Yes", "Unavailable"


# =============================================================================
# Commit / review timing helpers
# =============================================================================

def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def commits_after_first_review(
    commits: List[Dict[str, Any]],
    reviews: List[Dict[str, Any]],
    review_comments: List[Dict[str, Any]],
) -> int:
    """Count commits made after the earliest review or review-comment timestamp."""
    review_times: List[datetime] = []

    for review in reviews:
        dt = parse_datetime(review.get("submitted_at"))
        if dt:
            review_times.append(dt)

    for comment in review_comments:
        dt = parse_datetime(comment.get("created_at"))
        if dt:
            review_times.append(dt)

    if not review_times:
        return 0

    first_review_time = min(review_times)
    count = 0

    for commit in commits:
        commit_date = (
            commit.get("commit", {}).get("committer", {}).get("date")
        )
        dt = parse_datetime(commit_date)
        if dt and dt > first_review_time:
            count += 1

    return count


# =============================================================================
# Collaboration helpers
# =============================================================================

def detect_author_response(
    pr_author: str,
    issue_comments: List[Dict[str, Any]],
    review_comments: List[Dict[str, Any]],
) -> str:
    """Return "Yes" if the PR author posted at least one comment."""
    for comment in issue_comments + review_comments:
        if comment.get("user", {}).get("login") == pr_author:
            return "Yes"
    return "No"


def detect_change_requested(reviews: List[Dict[str, Any]]) -> str:
    for review in reviews:
        if review.get("state") == "CHANGES_REQUESTED":
            return "Yes"
    return "No"


# =============================================================================
# Suspicion heuristic
# =============================================================================

def conservative_ai_suspicion(
    ai_disclosed: str,
    pr_description_length: int,
    commits_per_pr: int,
    files_changed: int,
    lines_added: int,
    review_comments_count: int,
    test_changes_present: str,
) -> Tuple[str, str]:
    """
    Conservative heuristic for flagging PRs that may need manual review.

    This is NOT proof of AI usage.  It flags under-documented or unusual PRs
    based on configurable thresholds (see ``config.py``).  Multiple weak
    signals (``SUSPICION_MIN_SIGNALS_REQUIRED``) must co-occur before
    ``ai_suspected`` is set to "Yes".
    """
    if ai_disclosed == "Yes":
        return "No", ""

    reasons = []

    if lines_added >= SUSPICION_MIN_LINES_ADDED and pr_description_length < SUSPICION_MAX_DESCRIPTION_LEN:
        reasons.append("large code addition with short PR description")

    if files_changed >= SUSPICION_MIN_FILES_CHANGED and pr_description_length < SUSPICION_MAX_DESCRIPTION_LEN:
        reasons.append("many files changed with limited explanation")

    if commits_per_pr <= 1 and lines_added >= SUSPICION_SINGLE_COMMIT_LINES:
        reasons.append("large implementation introduced in one commit")

    if review_comments_count == 0 and lines_added >= SUSPICION_MIN_LINES_ADDED:
        reasons.append("substantial implementation with no review discussion")

    if test_changes_present == "No" and lines_added >= SUSPICION_MIN_LINES_ADDED:
        reasons.append("large implementation without test changes")

    if len(reasons) >= SUSPICION_MIN_SIGNALS_REQUIRED:
        return "Yes", "; ".join(reasons)

    return "No", ""


# =============================================================================
# Main feature-extraction entry point
# =============================================================================

def extract_features_for_pr(
    bundle: PullRequestRawBundle,
    repo: str,
    semester: str,
    group_info: RepoGroupInfo,
) -> Dict[str, Any]:
    """Convert one PullRequestRawBundle into one flat dataset row (dict)."""
    pr = bundle.pr
    pr_author = pr.get("user", {}).get("login", "")

    text_locations = collect_pr_text_locations(bundle)
    all_text = "\n".join(text_locations.values())
    pr_body = normalize_text(pr.get("body"))

    ai_disclosed, ai_keyword_count, ai_locations, ai_tools = detect_ai_disclosure(
        text_locations
    )

    commits_count = len(bundle.commits)
    files_changed = len(bundle.files)
    lines_added = sum(f.get("additions", 0) for f in bundle.files)
    lines_deleted = sum(f.get("deletions", 0) for f in bundle.files)

    test_changes_present = (
        "Yes"
        if any(is_test_file(f.get("filename", "")) for f in bundle.files)
        else "No"
    )

    ci_present, ci_result = extract_ci_status(bundle.check_runs)

    verification_count = count_keywords(all_text, VERIFICATION_KEYWORDS)
    reasoning_count = count_keywords(all_text, REASONING_KEYWORDS)

    pr_description_length = len(pr_body.strip())

    reviewers = unique_nonempty(
        r.get("user", {}).get("login")
        for r in bundle.reviews
        if r.get("user", {}).get("login")
    )

    unique_participants = unique_nonempty(
        [pr_author]
        + [c.get("user", {}).get("login") for c in bundle.issue_comments]
        + [c.get("user", {}).get("login") for c in bundle.review_comments]
        + [r.get("user", {}).get("login") for r in bundle.reviews]
    )

    after_review = commits_after_first_review(
        bundle.commits, bundle.reviews, bundle.review_comments
    )

    correction_after_review = "Yes" if after_review > 0 else "No"
    change_requested = detect_change_requested(bundle.reviews)
    author_response = detect_author_response(
        pr_author, bundle.issue_comments, bundle.review_comments
    )

    # Use REASONING_KEYWORDS as the single source of truth.
    justification_present = "Yes" if (
        reasoning_count >= 2
        or any(
            _word_boundary_pattern(keyword).search(pr_body.lower())
            for keyword in JUSTIFICATION_KEYWORDS
        )
    ) else "No"

    alternatives_discussed = "Yes" if ALTERNATIVES_REGEX.search(all_text) else "No"

    ai_suspected, suspicion_reason = conservative_ai_suspicion(
        ai_disclosed=ai_disclosed,
        pr_description_length=pr_description_length,
        commits_per_pr=commits_count,
        files_changed=files_changed,
        lines_added=lines_added,
        review_comments_count=len(bundle.review_comments),
        test_changes_present=test_changes_present,
    )

    disclosure_gap = "Yes" if ai_suspected == "Yes" and ai_disclosed == "No" else "No"
    is_merged = "Yes" if pr.get("merged_at") else "No"

    return {
        "semester": semester,
        "group": group_info.group,
        "team_id": group_info.team_id,
        "repo_name": split_repo_name(repo),
        "pr_number": pr.get("number"),
        "pr_url": pr.get("html_url"),
        "pr_author": pr_author,
        "created_at": pr.get("created_at"),
        "merged_at": pr.get("merged_at") or "",
        "closed_at": pr.get("closed_at") or "",
        "is_merged": is_merged,
        "ai_disclosed": ai_disclosed,
        "ai_keyword_count": ai_keyword_count,
        "ai_mention_locations": ai_locations,
        "ai_tool_mentioned": ai_tools,
        "ai_suspected": ai_suspected,
        "suspicion_reason": suspicion_reason,
        "disclosure_gap": disclosure_gap,
        "genai_usage_type": "",
        "usage_evidence": "",
        "integration_style": "",
        "commits_per_pr": commits_count,
        "commits_after_first_review": after_review,
        "files_changed": files_changed,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "test_changes_present": test_changes_present,
        "ci_present": ci_present,
        "ci_result": ci_result,
        "verification_keywords_count": verification_count,
        "correction_after_review": correction_after_review,
        "pr_description_length": pr_description_length,
        "reasoning_keywords_count": reasoning_count,
        "justification_present": justification_present,
        "alternatives_discussed": alternatives_discussed,
        "reviewers_count": len(reviewers),
        "review_comments_count": len(bundle.review_comments),
        "discussion_comments_count": len(bundle.issue_comments),
        "unique_participants_count": len(unique_participants),
        "change_requested": change_requested,
        "author_response_present": author_response,
        "patchtrack_category": "",
        "patchtrack_evidence": "",
    }
