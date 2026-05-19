"""
Configuration: all keyword lists, regex patterns, column schemas, and tunable
suspicion-heuristic thresholds for the GenAI repository mining pipeline.

Suspicion thresholds tie directly to the research protocol.  Change them
deliberately and document the rationale in the research notes.
"""

from __future__ import annotations

GITHUB_API = "https://api.github.com"

# ---------------------------------------------------------------------------
# GenAI disclosure detection
# ---------------------------------------------------------------------------

DISCLOSURE_SEEDS = [
    "chatgpt",
    "copilot",
    "claude",
    "cursor",
    "gemini",
    "codeium",
]

DISCLOSURE_CONTEXT_PATTERNS = [
    r"generated\s+by\s+(chatgpt|copilot|claude|cursor|gemini|codeium|ai|llm)",
    r"suggested\s+by\s+(chatgpt|copilot|claude|cursor|gemini|codeium|ai|llm)",
    r"using\s+(chatgpt|copilot|claude|cursor|gemini|codeium)",
    r"based\s+on\s+(ai|llm|genai)\s+output",
    r"ai[-\s]?generated",
    r"llm[-\s]?generated",
    r"genai",
    r"large\s+language\s+model",
]

# ---------------------------------------------------------------------------
# Keyword lists for verification and reasoning signals
# ---------------------------------------------------------------------------

VERIFICATION_KEYWORDS = [
    "test",
    "tested",
    "verify",
    "verified",
    "validate",
    "validated",
    "checked",
    "coverage",
    "ci",
    "build",
    "passing",
    "passed",
]

REASONING_KEYWORDS = [
    "because",
    "reason",
    "why",
    "alternative",
    "alternatives",
    "tradeoff",
    "tradeoffs",
    "decided",
    "approach",
    "design",
    "architecture",
    "limitation",
    "issue",
]

# ---------------------------------------------------------------------------
# Test-file detection patterns
# ---------------------------------------------------------------------------

TEST_FILE_PATTERNS = [
    r"(^|/|\\)test(s)?(/|\\)",
    r"(^|/|\\)__tests__(/|\\)",
    r"test_.*\.py$",
    r".*_test\.py$",
    r".*\.test\.(js|jsx|ts|tsx)$",
    r".*\.spec\.(js|jsx|ts|tsx)$",
    r".*Test\.(java|kt|cs)$",
]

# ---------------------------------------------------------------------------
# Output column schemas
# ---------------------------------------------------------------------------

FINAL_COLUMNS = [
    # General Metadata
    "semester",
    "group",
    "team_id",
    "repo_name",
    "pr_number",
    "pr_url",
    "pr_author",
    "created_at",
    "merged_at",
    "closed_at",
    "is_merged",

    # GenAI Disclosure and Transparency
    "ai_disclosed",
    "ai_keyword_count",
    "ai_mention_locations",
    "ai_tool_mentioned",

    # Transparency Gaps
    "ai_suspected",
    "suspicion_reason",
    "disclosure_gap",

    # GenAI Usage and Integration
    "genai_usage_type",
    "usage_evidence",
    "integration_style",

    # Development and Revision Signals
    "commits_per_pr",
    "commits_after_first_review",
    "files_changed",
    "lines_added",
    "lines_deleted",

    # Verification Behavior
    "test_changes_present",
    "ci_present",
    "ci_result",
    "verification_keywords_count",
    "correction_after_review",

    # Justification and Reasoning
    "pr_description_length",
    "reasoning_keywords_count",
    "justification_present",
    "alternatives_discussed",

    # Collaboration and Review
    "reviewers_count",
    "review_comments_count",
    "discussion_comments_count",
    "unique_participants_count",
    "change_requested",
    "author_response_present",

    # PatchTrack Classification
    "patchtrack_category",
    "patchtrack_evidence",
]

FAILURE_COLUMNS = [
    "timestamp",
    "repo",
    "pr_number",
    "stage",
    "url",
    "error_type",
    "status_code",
    "reason",
    "traceback",
]

# ---------------------------------------------------------------------------
# Suspicion heuristic thresholds
# ---------------------------------------------------------------------------

SUSPICION_MIN_LINES_ADDED = 500        # lines_added threshold for "large addition"
SUSPICION_MAX_DESCRIPTION_LEN = 200    # pr_description_length considered "short"
SUSPICION_MIN_FILES_CHANGED = 8        # files_changed threshold for "many files"
SUSPICION_SINGLE_COMMIT_LINES = 400    # lines_added where one commit is suspicious
SUSPICION_MIN_SIGNALS_REQUIRED = 2     # how many weak signals trigger ai_suspected
