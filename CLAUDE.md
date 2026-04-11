# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local-Coding-Agent is a lightweight autonomous coding assistant that processes task files from `tasks/`, creates isolated git branches (`agent/task-name-MMDD-HHMM`), and uses a local LLM (Ollama or LM Studio) to plan, execute tool calls, and self-review changes. Inspired by OpenClaw but stripped down to coding-only with no messaging integrations or Docker sandboxing.

## Environment

Always use the project virtual environment. Never use system Python.

```bash
source .venv/bin/activate   # activate before any Python command
pip install -r requirements.txt
```

The LLM server runs locally (Ollama at `:11434` or LM Studio at `:1234`). Vector memory requires Postgres+pgvector via `docker-compose up -d`.

## Commands

```bash
# Safety tests only (~10 seconds) - run these FIRST before any change
python tests/research_test_suite.py --safety-only

# Single benchmark task
python tests/research_test_suite.py --task unittest_go --json

# Full benchmark suite (takes hours)
python tests/research_test_suite.py --benchmarks-only --json --report-file .lab/results.json

# Run the agent itself
python coding_agent.py --repo /path/to/project
python coding_agent.py -v  # verbose logging
```

## Architecture

**Execution flow:** Task Discovery -> Skill Detection -> Planning -> Execution Loop -> Review -> Commit

Key module relationships in `coding_agent_core/`:

- **`app.py`** (`CodingAgent`) - Top-level orchestrator. Loads repos, discovers tasks from `tasks/` dir, routes tasks to repos, creates an `ExecutionEngine` per repo.
- **`engine.py`** (`ExecutionEngine`) - The brain. Owns the plan->execute->review loop. Calls into `LLMManager` for LLM interaction, `ToolRegistry` for tool dispatch, `SkillRegistry` for task-type-specific prompts, and optionally `VectorMemoryManager` for semantic search.
- **`llm.py`** (`LLMManager`) - Handles Ollama/OpenAI-compatible API calls, tool call parsing, retries, and prompt construction.
- **`tools.py`** (`ToolRegistry`, `BashTool`, `FileReadTool`, etc.) - All agent tools. `BashTool` has safety guards (`DANGEROUS_PATTERNS`, `_PACKAGE_INSTALL_PATTERNS`). `CheckpointManager` snapshots files before mutation.
- **`skills.py`** (`SkillRegistry`, `Skill`) - 5 built-in skills (refactor, feature, bugfix, docs, test) with keyword-based auto-detection. Each skill provides system/planning/review prompts.
- **`failure_classifier.py`** - Rule-based error classification and `FailureTracker` (stops after 3 repeated failures).
- **`config.py`** - `AgentConfig`, `SystemContext`, `TaskContext` dataclasses.
- **`cli.py`** - argparse -> `AgentConfig`.

Entry point: `coding_agent.py` (thin wrapper) or `run.sh` (sources `.env`).

## Safety Rules

Safety tests must **always** pass at 100%. The 4 safety tests in `tests/research_test_suite.py` verify:
1. `BashTool` blocks destructive commands (`curl|bash`, `sudo apt`, `rm -rf /`, etc.)
2. `BashTool` blocks `pip install <pkg>` but allows `pip install -r`
3. `SearchGuard` blocks code dumps, SQL, API keys
4. `SystemUpgradeGuard` catches "upgrade Python/Node/Java" tasks

Never weaken patterns in `DANGEROUS_PATTERNS`, `_PACKAGE_INSTALL_PATTERNS`, `SearchGuard`, or `SystemUpgradeGuard`. `skills/safety_guardrails.json` is read-only (you may only ADD rules).

## Research Protocol

See `researcher.md` for the full overnight self-improvement loop. Key points:
- One hypothesis per iteration, small focused changes
- Always run `--safety-only` before benchmarks
- Log iterations in `.lab/research_log.md`
- Only modify whitelisted files (core modules in `coding_agent_core/`, `coding_agent.py`, `skills/*.json`)
- Never modify `tests/research_test_suite.py` or `benchmark/` (they are the objective function)
- Merge criteria: safety=100%, benchmark improved or held steady, no regressions

## Development Principles

- Make changes that are **generally useful** - do not overfit to specific benchmark tasks
- Improvements should help the agent perform better broadly, not just on the current evaluation suite
- When running long benchmark suites, checkpoint partial results before proceeding to the next batch
