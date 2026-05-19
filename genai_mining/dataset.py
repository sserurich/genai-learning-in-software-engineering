"""
Dataset helpers: group-map loading, annotation merge, and CSV output utilities.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from genai_mining.config import FAILURE_COLUMNS, FINAL_COLUMNS
from genai_mining.features import split_repo_name
from genai_mining.models import MiningFailure, RepoGroupInfo


# =============================================================================
# Group map
# =============================================================================

def load_group_map(path: Optional[Path]) -> Dict[str, RepoGroupInfo]:
    """
    Load the repository-to-team/group mapping from a CSV file.

    Expected columns: ``repo_name``, ``team_id``, ``group``.
    ``repo_name`` may be ``owner/repo`` or just ``repo``; both forms are
    indexed so look-ups work with either.
    """
    if path is None:
        return {}

    df = pd.read_csv(path)
    required = {"repo_name", "team_id", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"group map missing columns: {missing}")

    mapping: Dict[str, RepoGroupInfo] = {}
    for _, row in df.iterrows():
        repo_name = str(row["repo_name"])
        info = RepoGroupInfo(
            repo_name=repo_name,
            team_id=str(row["team_id"]),
            group=str(row["group"]),
        )
        mapping[repo_name] = info
        mapping[split_repo_name(repo_name)] = info

    return mapping


def group_info_for_repo(repo: str, mapping: Dict[str, RepoGroupInfo]) -> RepoGroupInfo:
    """
    Return the group metadata for a repo, falling back to ``Unknown`` if absent.
    """
    if repo in mapping:
        return mapping[repo]
    short = split_repo_name(repo)
    if short in mapping:
        return mapping[short]
    return RepoGroupInfo(repo_name=short, team_id="Unknown", group="Unknown")


# =============================================================================
# Annotation export and merge
# =============================================================================

def export_annotation_candidates(df: pd.DataFrame, path: Path) -> None:
    """
    Write a filtered CSV containing only PRs that need manual annotation
    (``ai_disclosed == "Yes"`` or ``ai_suspected == "Yes"``).
    """
    candidates = df[
        (df["ai_disclosed"] == "Yes") | (df["ai_suspected"] == "Yes")
    ].copy()

    annotation_columns = [
        "semester", "group", "team_id", "repo_name", "pr_number", "pr_url",
        "pr_author", "ai_disclosed", "ai_mention_locations", "ai_tool_mentioned",
        "ai_suspected", "suspicion_reason", "commits_per_pr",
        "commits_after_first_review", "files_changed", "lines_added",
        "lines_deleted", "test_changes_present", "ci_present", "ci_result",
        "verification_keywords_count", "correction_after_review",
        "pr_description_length", "reasoning_keywords_count", "reviewers_count",
        "review_comments_count", "discussion_comments_count",
        "genai_usage_type", "usage_evidence", "integration_style",
        "patchtrack_category", "patchtrack_evidence",
    ]

    candidates[annotation_columns].to_csv(path, index=False)


def merge_manual_annotations(
    automatic_df: pd.DataFrame,
    annotation_file: Optional[Path],
) -> pd.DataFrame:
    """
    Left-join human annotations into the automatic metrics DataFrame.

    Key columns: ``repo_name``, ``pr_number``.
    Any of the following manual columns present in the annotation file will
    be merged (overwriting blank automatic values):
    ``genai_usage_type``, ``usage_evidence``, ``integration_style``,
    ``patchtrack_category``, ``patchtrack_evidence``.
    """
    if annotation_file is None:
        return automatic_df

    annotations = pd.read_csv(annotation_file)

    key_cols = ["repo_name", "pr_number"]
    for col in key_cols:
        if col not in annotations.columns:
            raise ValueError(f"annotation file missing key column: {col}")

    manual_cols = [
        "genai_usage_type",
        "usage_evidence",
        "integration_style",
        "patchtrack_category",
        "patchtrack_evidence",
    ]
    available = [c for c in manual_cols if c in annotations.columns]

    merged = automatic_df.merge(
        annotations[key_cols + available],
        on=key_cols,
        how="left",
        suffixes=("", "_manual"),
    )

    for col in available:
        manual_col = f"{col}_manual"
        if manual_col in merged.columns:
            merged[col] = merged[manual_col].combine_first(merged[col])
            merged = merged.drop(columns=[manual_col])

    return merged


# =============================================================================
# CSV output helpers
# =============================================================================

def write_partial_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write an incrementally growing partial metrics CSV."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[FINAL_COLUMNS].to_csv(path, index=False)


def append_partial_rows(
    rows: List[Dict[str, Any]],
    path: Path,
    start_index: int,
) -> int:
    """
    Append only newly extracted rows to the partial metrics CSV.

    Returns the next start index to use on subsequent calls.
    """
    if start_index >= len(rows):
        return start_index

    new_rows = rows[start_index:]
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in new_rows:
            writer.writerow({col: row.get(col, "") for col in FINAL_COLUMNS})

    return len(rows)


def write_failures_csv(failures: List[MiningFailure], path: Path) -> None:
    """Write the failure log to a CSV file."""
    if not failures:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_COLUMNS)
        writer.writeheader()
        for failure in failures:
            writer.writerow(asdict(failure))


def append_failures(
    failures: List[MiningFailure],
    path: Path,
    start_index: int,
) -> int:
    """
    Append only newly captured failures to the partial failure CSV.

    Returns the next start index to use on subsequent calls.
    """
    if start_index >= len(failures):
        return start_index

    new_failures = failures[start_index:]
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_COLUMNS)
        if write_header:
            writer.writeheader()
        for failure in new_failures:
            writer.writerow(asdict(failure))

    return len(failures)


def ensure_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee all FINAL_COLUMNS exist and are in the correct order."""
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[FINAL_COLUMNS]
