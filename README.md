# Robust GenAI Repository Mining Pipeline

This pipeline supports the **GLSE** (GenAI Learning in Software Engineering) study, mining GitHub PR workflows to produce a PR-level dataset for analyzing GenAI usage in student projects (UNLV CS 472/672).

## Create virtual environment and install dependencies

Create a virtual environment first, then install dependencies.

### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows (Command Prompt)

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer manual package install instead of requirements file:

```bash
python -m pip install pandas requests
```

## Architecture

The pipeline is organized as a modular Python package:

```
genai_mining/
├── __init__.py              # Main entry point (exports build_dataset)
├── config.py                # Constants (API endpoints, keywords, thresholds)
├── models.py                # Dataclasses (RepoGroupInfo, PullRequestRawBundle, MiningFailure)
├── token_pool.py            # Round-robin GitHub token management
├── github_client.py         # HTTP client + disk cache for GitHub API
├── miner.py                 # PR fetching (commits, files, comments, reviews, check runs)
├── features.py              # Feature extraction (disclosure detection, suspicion heuristic, CI parsing)
├── dataset.py               # Group map loading, annotation merge, CSV export helpers
└── pipeline.py              # Main orchestrator (wires all modules together)
```

**Key functions:**
- `genai_mining.pipeline.build_dataset()` — top-level entry point (called by `mine_genai_repos.py`)
- `genai_mining.miner.mine_repository()` — fetches all PR sub-resources from GitHub
- `genai_mining.features.extract_features_for_pr()` — extracts all metrics for a single PR
- `genai_mining.dataset.*` — helpers for group maps, annotation merging, and CSV I/O

**Token management:**
- `TokenPool` (in `token_pool.py`) auto-discovers tokens from `GITHUB_TOKEN`, `GITHUB_TOKENS` env vars, and optional `--tokens-file`
- Tokens are rotated round-robin to distribute API load

**Error handling:**
- Each PR sub-resource fetch is wrapped in try/except
- Failures are logged to `mining_failures.csv` with full details (stage, error, traceback)
- Pipeline never aborts — continues even after failures
- GitHub 5xx errors (500, 502, 503, 504) are retried up to 3 times with exponential backoff

## Token options

All token sources are **merged** — you can combine any of the options below and the pipeline will round-robin across all discovered tokens.

### Option 1: Single token (environment variable)

```bash
export GITHUB_TOKEN="ghp_your_token_here"
```

### Option 2: Multiple tokens (comma-separated environment variable)

```bash
export GITHUB_TOKENS="ghp_token1,ghp_token2,ghp_token3"
```

### Option 3: Multiple tokens via file

Create `tokens.txt` — one token per line. Blank lines and lines starting with `#` are ignored:

```text
# primary account
ghp_token1
# secondary account
ghp_token2
ghp_token3
```

Then pass:

```bash
--tokens-file tokens.txt
```

## Group map

Create `group_map.csv`:

```csv
repo_name,team_id,group
judge-portal,Team-08,Treatment
student-ecommerce-app,Team-14,Control
```

`repo_name` may be either `owner/repo` or just `repo`.

## Run mining

```bash
python3 mine_genai_repos.py \
  --repos Freddie-Pike/judge-portal NL-Eats-Community-Outreach-Inc/judge-portal \
  --semester Spring2026 \
  --group-map group_map.csv \
  --tokens-file tokens.txt \
  --outdir outputs
```

To bypass the local JSON cache and re-fetch everything from GitHub:

```bash
python3 mine_genai_repos.py \
  --repos owner/repo1 owner/repo2 \
  --semester Spring2026 \
  --no-cache \
  --outdir outputs
```

## Outputs

```text
outputs/
  raw_data/
  automatic_metrics.csv
  automatic_metrics.partial.csv
  annotation_candidates.csv
  final_dataset.csv
  mining_failures.csv
  mining_failures.partial.csv
```

## Failure handling

If a PR or sub-resource fails, the script logs the failure and continues. The whole run never aborts due to a single PR error.

The failure log (`mining_failures.csv`) contains:

```csv
timestamp,repo,pr_number,stage,url,error_type,status_code,reason,traceback
```

Stages logged in the failure file:

- `fetch_commits`
- `fetch_files`
- `fetch_issue_comments`
- `fetch_review_comments`
- `fetch_reviews`
- `fetch_check_runs`
- `feature_extraction`
- `merge_manual_annotations`

Transient GitHub errors (`500`, `502`, `503`, `504`) are retried up to 3 times with exponential backoff before being logged as failures.

## Manual annotation workflow

Open:

```text
outputs/annotation_candidates.csv
```

Only PRs where `ai_disclosed = Yes` or `ai_suspected = Yes` appear in this file.

Fill the following columns using the controlled vocabularies below:

| Column | Valid values |
|---|---|
| `genai_usage_type` | `Code Generation`, `Debugging`, `Documentation`, `Testing`, `Refactoring`, `Design/Architecture`, `Other`, `` (leave blank if unknown) |
| `usage_evidence` | Free text — quote or paraphrase the disclosure evidence |
| `integration_style` | `Wholesale` (AI output pasted with no changes), `Partial` (AI output edited/merged), `Prompted` (developer used AI interactively), `Unknown` |
| `patchtrack_category` | `AI-Assisted`, `Human-Only`, `Uncertain` |
| `patchtrack_evidence` | Free text — brief justification for the patchtrack classification |

Save as `manual_annotations.csv` (keep `repo_name` and `pr_number` columns as keys), then rerun:

```bash
python3 mine_genai_repos.py \
  --repos owner/repo1 owner/repo2 \
  --semester Spring2026 \
  --group-map group_map.csv \
  --tokens-file tokens.txt \
  --annotation-file manual_annotations.csv \
  --outdir outputs
```

## Important validity note

The script does not claim to detect hidden GenAI use.

- `ai_disclosed` means explicit disclosure only.
- `ai_keyword_count` counts only named tool mentions (e.g. "chatgpt", "copilot"). Context patterns such as "ai-generated" contribute to `ai_disclosed` detection but are **not** counted here, so the metric is not inflated by overlapping matches.
- `ai_suspected` is only a conservative manual-review flag based on configurable thresholds (`SUSPICION_MIN_LINES_ADDED`, `SUSPICION_MAX_DESCRIPTION_LEN`, etc. in the script).
- Manual GenAI labels should only be applied to PRs where `ai_disclosed = Yes` or `ai_suspected = Yes`.
