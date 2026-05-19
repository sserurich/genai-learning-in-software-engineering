"""
Top-level pipeline: orchestrates token loading, mining, feature extraction,
and all CSV outputs in a single ``build_dataset`` call.
"""

from __future__ import annotations

import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from genai_mining.config import FINAL_COLUMNS
from genai_mining.dataset import (
    append_failures,
    append_partial_rows,
    ensure_final_columns,
    export_annotation_candidates,
    group_info_for_repo,
    load_group_map,
    merge_manual_annotations,
    write_failures_csv,
)
from genai_mining.features import extract_features_for_pr
from genai_mining.github_client import GitHubClient, RawCache
from genai_mining.miner import mine_repository
from genai_mining.models import MiningFailure
from genai_mining.token_pool import TokenPool


def _parse_date_arg(label: str, value: Optional[str]) -> Optional[date]:
    """Parse optional YYYY-MM-DD date CLI inputs."""
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {label} '{value}'. Expected YYYY-MM-DD.") from exc


def build_dataset(
    repos: List[str],
    semester: str,
    group_map_file: Optional[Path],
    annotation_file: Optional[Path],
    outdir: Path,
    tokens_file: Optional[Path] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> None:
    """
    Run the full mining and dataset preparation pipeline.

    Steps:
    1. Load tokens (env vars + optional file) and build a GitHubClient.
    2. For each repository, mine all PRs and extract features.
    3. After each repository, write partial CSVs so progress is never lost.
    4. Write final ``automatic_metrics.csv``, ``annotation_candidates.csv``,
       ``final_dataset.csv``, and ``mining_failures.csv``.
    """
    pool = TokenPool.from_env_and_file(tokens_file)
    print(f"[tokens] Using {len(pool)} token(s).")

    start_bound = _parse_date_arg("--start-date", start_date)
    end_bound = _parse_date_arg("--end-date", end_date)
    if start_bound and end_bound and start_bound > end_bound:
        raise ValueError("--start-date must be less than or equal to --end-date.")

    outdir.mkdir(parents=True, exist_ok=True)
    raw_cache = RawCache(outdir / "raw_data")

    client = GitHubClient(pool=pool)
    group_map = load_group_map(group_map_file)

    rows: List[Dict[str, Any]] = []
    all_failures: List[MiningFailure] = []
    rows_checkpoint = 0
    failures_checkpoint = 0

    partial_metrics_path = outdir / "automatic_metrics.partial.csv"
    partial_failures_path = outdir / "mining_failures.partial.csv"

    if start_bound or end_bound:
        print(
            "[filter] PR created_at range: "
            f"{start_bound.isoformat() if start_bound else 'MIN'}"
            " to "
            f"{end_bound.isoformat() if end_bound else 'MAX'}"
        )

    for repo in repos:
        group_info = group_info_for_repo(repo, group_map)
        bundles, repo_failures = mine_repository(
            client=client,
            cache=raw_cache,
            repo=repo,
            state="all",
            use_cache=use_cache,
            start_date=start_bound,
            end_date=end_bound,
        )
        all_failures.extend(repo_failures)

        for bundle in bundles:
            try:
                row = extract_features_for_pr(
                    bundle=bundle,
                    repo=repo,
                    semester=semester,
                    group_info=group_info,
                )
                rows.append(row)
            except Exception as exc:
                pr_number = bundle.pr.get("number", "?")
                print(
                    f"[error] feature_extraction {repo} PR #{pr_number}: {exc}",
                    file=sys.stderr,
                )
                all_failures.append(MiningFailure(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    repo=repo,
                    pr_number=str(pr_number),
                    stage="feature_extraction",
                    url="",
                    error_type=type(exc).__name__,
                    status_code="",
                    reason=str(exc),
                    traceback=traceback.format_exc(),
                ))

        # Persist only the newly added records after every repository.
        rows_checkpoint = append_partial_rows(
            rows,
            partial_metrics_path,
            rows_checkpoint,
        )
        failures_checkpoint = append_failures(
            all_failures,
            partial_failures_path,
            failures_checkpoint,
        )
        print(
            f"[partial] {repo}: {len(rows)} rows, {len(all_failures)} failures so far."
        )

    # -------------------------------------------------------------------------
    # Build automatic metrics DataFrame
    # -------------------------------------------------------------------------
    if not rows:
        print("[warning] No pull requests were successfully mined. Outputs will be empty.")
        automatic_df = pd.DataFrame(columns=FINAL_COLUMNS)
    else:
        automatic_df = pd.DataFrame(rows)

    automatic_df = ensure_final_columns(automatic_df)

    automatic_path = outdir / "automatic_metrics.csv"
    automatic_df.to_csv(automatic_path, index=False)
    print(f"[output] automatic metrics → {automatic_path}")

    annotation_candidates_path = outdir / "annotation_candidates.csv"
    export_annotation_candidates(automatic_df, annotation_candidates_path)
    print(f"[output] annotation candidates → {annotation_candidates_path}")

    # -------------------------------------------------------------------------
    # Merge manual annotations (if provided)
    # -------------------------------------------------------------------------
    try:
        final_df = merge_manual_annotations(automatic_df, annotation_file)
    except Exception as exc:
        print(f"[error] merge_manual_annotations: {exc}", file=sys.stderr)
        all_failures.append(MiningFailure(
            timestamp=datetime.now(timezone.utc).isoformat(),
            repo="",
            pr_number="",
            stage="merge_manual_annotations",
            url="",
            error_type=type(exc).__name__,
            status_code="",
            reason=str(exc),
            traceback=traceback.format_exc(),
        ))
        final_df = automatic_df

    final_df = ensure_final_columns(final_df)

    final_path = outdir / "final_dataset.csv"
    final_df.to_csv(final_path, index=False)
    print(f"[output] final dataset → {final_path}")

    # -------------------------------------------------------------------------
    # Write failure log
    # -------------------------------------------------------------------------
    failures_path = outdir / "mining_failures.csv"
    write_failures_csv(all_failures, failures_path)
    if all_failures:
        print(f"[output] {len(all_failures)} failure(s) → {failures_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\nSummary")
    print("-------")
    print(f"Repositories mined : {len(repos)}")
    print(f"Pull requests      : {len(final_df)}")
    if len(final_df) > 0:
        print(f"AI disclosed PRs   : {(final_df['ai_disclosed'] == 'Yes').sum()}")
        print(f"AI suspected PRs   : {(final_df['ai_suspected'] == 'Yes').sum()}")
        print(f"Disclosure gaps    : {(final_df['disclosure_gap'] == 'Yes').sum()}")
    print(f"Mining failures    : {len(all_failures)}")
