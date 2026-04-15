# Coding Agent

Local-Coding-Agent is a lightweight autonomous coding assistant for local repositories. It watches `tasks/`, creates isolated branches (`agent/task-name-MMDD-HHMM`), and uses a local LLM to plan, execute, review, and commit changes.

The original inspiration came from [OpenClaw](https://github.com/OpenClaw), but this project is intentionally independent and focused on coding workflows only.

## What It Does

- Processes task files from `tasks/`
- Auto-detects task type with built-in skills (`refactor`, `feature`, `bugfix`, `docs`, `test`)
- Runs an execution loop: plan -> execute -> review -> iterate
- Uses guarded tools for file edits, bash, and git actions
- Creates local git branches for review (no automatic push)
- Supports optional semantic memory via Postgres + pgvector

## Quick Start

### 1) Set up the environment

Always use the project virtual environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start your local LLM

Run a compatible local API server before launching the agent:

- Ollama (default local endpoint: `http://localhost:11434`)
- LM Studio (typical endpoint: `http://localhost:1234`)

See [LLM_SETUP.md](./LLM_SETUP.md) for details.

### 3) Optional: enable vector memory

If you want semantic codebase memory, start Postgres + pgvector:

```bash
docker-compose up -d
```

### 4) Prepare a target repository

```bash
cd /path/to/your-repo
mkdir -p tasks skills .coding-agent
```

Add one or more task files, for example:

```bash
cat > tasks/refactor-auth.txt <<'EOF'
Refactor the auth module to use dependency injection.
Keep existing tests passing.
EOF
```

### 5) Run the agent

```bash
python /path/to/Local-Coding-Agent/coding_agent.py --repo /path/to/your-repo
```

Useful variants:

```bash
python coding_agent.py -v
python coding_agent.py --model codellama
```

### 6) Review generated branches

```bash
git branch | grep '^  agent/\|^agent/'
git checkout agent/refactor-auth-0217-0830
git diff main
```

## LLM Model Validation

Run these from the Local-Coding-Agent repo (with `.venv` activated):

```bash
# Safety checks first (~10s)
python tests/research_test_suite.py --safety-only

# Single benchmark task
python tests/research_test_suite.py --task unittest_go --json

# Full benchmark suite (long-running)
python tests/research_test_suite.py --benchmarks-only --json --report-file .lab/results.json
```

## Architecture

Execution flow:

```text
Task Discovery -> Skill Detection -> Planning -> Execution Loop -> Review -> Commit
```

Core modules in `coding_agent_core/`:

- `app.py` (`CodingAgent`): top-level orchestrator and repo/task routing
- `engine.py` (`ExecutionEngine`): plan/execute/review loop controller
- `llm.py` (`LLMManager`): local LLM API interaction and tool call parsing
- `tools.py` (`ToolRegistry` + tools): tool dispatch and safety-guarded execution
- `skills.py` (`SkillRegistry`): built-in skills and prompt templates
- `failure_classifier.py`: repeated-failure detection and classification
- `config.py`: runtime configuration dataclasses
- `cli.py`: CLI argument parsing into config

Entry points: `coding_agent.py` and `run.sh`.

## Safety Model

Safety behavior is part of the product, not an optional mode:

- `BashTool` blocks destructive shell patterns
- package installation via raw `pip install <pkg>` is blocked (while `pip install -r` is allowed)
- `SearchGuard` blocks sensitive or unsafe retrieval patterns
- `SystemUpgradeGuard` blocks upgrade-task requests for major runtimes

Do not weaken safety guard patterns. Extend rules instead when needed.

## Skills

Built-in skills:

- `refactor`
- `feature`
- `bugfix`
- `docs`
- `test`

Custom skills can be added in `skills/` as JSON definitions. Built-ins are loaded by default; do not duplicate them in custom files.

## Repository Layout (Target Repo)

Typical structure expected by the agent:

```text
your-project/
  tasks/
    my-task.txt
  skills/
    my-custom-skill.json
  .coding-agent/
    archive/
    agent.log
    reports/
```

## Configuration

Use `--help` for the full, current CLI surface:

```bash
python coding_agent.py --help
```

Common flags include repository selection, task and skills directories, base branch, model endpoint/model name, iteration limits, and multi-repo discovery options.

For multi-repo workflows, see [MULTI_REPO.md](./MULTI_REPO.md).

## Research Workflow Notes

If you are iterating on agent quality:

- run `--safety-only` before benchmarks
- keep changes small and hypothesis-driven
- log iterations in `.lab/research_log.md`
- avoid changing benchmark objective files directly

See `researcher.md` for the complete protocol.

## Requirements

- Python 3.8+
- dependencies in `requirements.txt`
- local LLM server (Ollama or LM Studio)
- optional: Docker Compose for pgvector-backed memory

## License

MIT
