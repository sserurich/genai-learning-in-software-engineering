"""
TokenPool: round-robin GitHub token management.

Supports three token sources (all are merged):
  1. GITHUB_TOKENS env var  — comma-separated list of tokens
  2. GITHUB_TOKEN env var   — single token
  3. --tokens-file argument — one token per line; blank lines and
                              lines starting with # are ignored
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Iterator, List, Optional


class TokenPool:
    """
    Round-robin pool of GitHub personal access tokens.

    All token sources are merged; the pool cycles through them in order so
    that rate-limit pressure is distributed across accounts.
    """

    def __init__(self, tokens: List[str]) -> None:
        if not tokens:
            raise RuntimeError(
                "No GitHub tokens provided. "
                "Set GITHUB_TOKEN, GITHUB_TOKENS, or pass --tokens-file."
            )
        self._cycle: Iterator[str] = itertools.cycle(tokens)
        self._tokens = tokens

    def next(self) -> str:
        """Return the next token in the rotation."""
        return next(self._cycle)

    def __len__(self) -> int:
        return len(self._tokens)

    @classmethod
    def from_env_and_file(cls, tokens_file: Optional[Path] = None) -> "TokenPool":
        """
        Build a TokenPool from environment variables and/or a tokens file.

        Priority / merge order:
        1. GITHUB_TOKENS env var (comma-separated)
        2. GITHUB_TOKEN env var (single token)
        3. tokens_file (one token per line)
        """
        tokens: List[str] = []

        for t in os.environ.get("GITHUB_TOKENS", "").split(","):
            t = t.strip()
            if t:
                tokens.append(t)

        single = os.environ.get("GITHUB_TOKEN", "").strip()
        if single and single not in tokens:
            tokens.append(single)

        if tokens_file is not None:
            for line in tokens_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and line not in tokens:
                    tokens.append(line)

        return cls(tokens)
