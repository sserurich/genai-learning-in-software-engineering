#!/usr/bin/env python3
"""
CLI entry point for the GenAI repository mining pipeline.

All pipeline logic lives in the ``genai_mining`` package.  This file only
parses command-line arguments and calls ``build_dataset``.

Usage
-----
    python3 mine_genai_repos.py \\
        --repos owner/repo1 owner/repo2 \\
        --semester Spring2026 \\
        --group-map group_map.csv \\
        --tokens-file tokens.txt \\
        --outdir outputs

See README.md for full documentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from genai_mining.pipeline import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine GitHub repositories and prepare GenAI PR-level dataset."
    )

    parser.add_argument(
        "--repos",
        nargs="+",
        required=True,
        help="GitHub repositories in owner/repo format.",
    )

    parser.add_argument(
        "--semester",
        required=True,
        help="Semester label, e.g. Spring2026.",
    )

    parser.add_argument(
        "--group-map",
        type=Path,
        default=None,
        help="CSV mapping repos to team_id and Treatment/Control group.",
    )

    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help="Optional manual annotation CSV to merge into final dataset.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs"),
        help="Output directory.",
    )

    parser.add_argument(
        "--tokens-file",
        type=Path,
        default=None,
        help=(
            "Path to a text file containing one GitHub token per line. "
            "Combined with GITHUB_TOKEN / GITHUB_TOKENS env vars. "
            "Blank lines and lines starting with # are ignored."
        ),
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached raw JSON and re-fetch from GitHub.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_dataset(
        repos=args.repos,
        semester=args.semester,
        group_map_file=args.group_map,
        annotation_file=args.annotation_file,
        outdir=args.outdir,
        tokens_file=args.tokens_file,
        use_cache=not args.no_cache,
    )
