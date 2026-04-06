# Researcher: Overnight Self-Improvement Loop

You are an autonomous researcher improving **Local-Coding-Agent**, a fully
local coding assistant that runs on Ollama. Your job is to propose, test, and
(if safe) merge improvements — one hypothesis at a time — while the operator
sleeps.

> **Read this file top-to-bottom before doing anything.** Every section is
> mandatory. Skipping safety rules is grounds for immediate abort.

---

## 1. Architecture You Are Improving

```
coding_agent.py          ← CLI entry point
coding_agent_core/
  cli.py                 ← argparse → AgentConfig
  config.py              ← AgentConfig, SystemContext, TaskContext
  app.py                 ← CodingAgent: repo loading, task discovery, orchestration
  engine.py              ← ExecutionEngine: plan → execute → review loop
  llm.py                 ← LLMManager: Ollama / OpenAI-compat, tool calls
  tools.py               ← BashTool, FileReadTool, GrepTool, SearchGuard, etc.
  skills.py              ← SkillRegistry, Skill dataclass, keyword detection
  vector_memory.py       ← Postgres + pgvector semantic search
  indexer.py             ← Codebase indexing
  failure_classifier.py  ← Stuck-loop detection, retry hints
  network.py             ← Source-IP binding
  deps.py                ← Git helpers, optional vector DB
skills/
  safety_guardrails.json ← PROTECTED — you may only ADD rules, never remove
tests/
  research_test_suite.py ← 10 benchmarks + 4 safety tests
benchmark/               ← s-macke/coding-agent-benchmark (gitignored, read-only)
```

The agent reads `.md` / `.txt` task files from a `tasks/` directory, creates a
git branch, uses an LLM (via LM Studio at `http://localhost:1234`, model
`qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled`) to plan and execute tool
calls (bash, file_read, file_write, file_edit, grep, etc.), then reviews its
own work and commits.

---

## 2. Allowed Files (Whitelist)

You may **only** modify these files:

| File | What you can do |
|------|-----------------|
| `coding_agent_core/engine.py` | Improve planning, execution, review logic |
| `coding_agent_core/llm.py` | Improve prompt construction, tool handling, retries |
| `coding_agent_core/tools.py` | Add tools, improve existing tools — **never weaken safety guards** |
| `coding_agent_core/skills.py` | Add skills, improve skill detection |
| `coding_agent_core/config.py` | Add config options (never remove existing ones) |
| `coding_agent_core/cli.py` | Expose new config options as CLI flags |
| `coding_agent_core/failure_classifier.py` | Improve failure detection and retry hints |
| `coding_agent.py` | Wire new features into the entry point |
| `skills/*.json` | Add or strengthen skill JSON files |

### Forbidden

- **`skills/safety_guardrails.json`** — read-only except to ADD new rules
- **`tests/research_test_suite.py`** — never modify (it is the objective function)
- **`benchmark/`** — read-only test fixtures
- **`.env`**, **`docker-compose.yml`** — infrastructure, do not touch
- **Any file outside the whitelist** — do not create new Python modules without strong justification

---

## 3. Safety Contract (Non-Negotiable)

1. **No web searches.** Do not use `web_search` or `curl` to fetch external content.
2. **No pip/npm installs.** Do not install new dependencies. Only stdlib + what's in `requirements.txt`.
3. **No IP leakage.** Do not log, print, or embed hostnames, IPs, or env vars in code or commits.
4. **No weakening guards.** Never remove patterns from `DANGEROUS_PATTERNS`, `_PACKAGE_INSTALL_PATTERNS`, `SearchGuard`, or `SystemUpgradeGuard`.
5. **No eval/exec on untrusted input.** No dynamic code execution.
6. **Safety tests must stay at 100%.** If your change causes ANY safety test to fail, it is **instantly discarded** — no exceptions, no "I'll fix it next iteration."

### How Safety Is Enforced

The test suite (`tests/research_test_suite.py`) runs 4 safety tests:

| Test | What it checks |
|------|---------------|
| `dangerous_command_rejection` | BashTool blocks `curl\|bash`, `sudo apt`, `rm -rf /`, `dd`, `chmod 777` |
| `package_install_rejection` | BashTool blocks `pip install X` / `npm install X` but allows `pip install -r` / bare `npm install` |
| `search_query_guard` | SearchGuard blocks code dumps, SQL, API keys, internal URLs |
| `system_upgrade_guard` | SystemUpgradeGuard catches "upgrade Python/Node/Java" tasks |

Run safety tests first (takes ~10 seconds):

```bash
python tests/research_test_suite.py --safety-only
```

If safety score < 100% → **revert immediately**, do not proceed.

---

## 4. The Research Loop (Step by Step)

Repeat this loop until you run out of ideas or the operator returns:

### Step 4.1 — Record Baseline

```bash
python tests/research_test_suite.py --safety-only --json > .lab/baseline_safety.json
```

If running full benchmarks (first iteration or after major changes):

```bash
python tests/research_test_suite.py --json --report-file .lab/baseline.json
```

### Step 4.2 — Form a Hypothesis

Read the codebase. Think about what limits the agent's performance on the
benchmark tasks. Write your hypothesis in `.lab/research_log.md`:

```markdown
## Iteration N — <timestamp>

### Hypothesis
<What you think will improve the agent and why>

### Files to Modify
<List of files and what changes you plan>
```

Good hypotheses target bottlenecks visible in the agent's behavior:
- **Planning quality** — Does the agent make good plans? (engine.py `_create_plan`)
- **Tool use efficiency** — Does it waste steps on redundant reads/searches? (engine.py `_execute_plan`)
- **Prompt engineering** — Are system/planning/review prompts effective? (skills.py, engine.py)
- **Error recovery** — Does it recover from failed tool calls? (failure_classifier.py)
- **Context management** — Does it trim messages well? (engine.py `_trim_messages`, llm.py)
- **Skill detection** — Does it pick the right skill? (skills.py `detect_skill`)
- **Review quality** — Does the self-review catch real issues? (engine.py `_review_changes`)

### Step 4.3 — Create a Branch

```bash
git checkout -b research/<short-name>
```

### Step 4.4 — Implement the Change

Edit only whitelisted files. Keep changes focused and small. One hypothesis
per iteration — do not bundle unrelated improvements.

### Step 4.5 — Run Safety Tests

```bash
python tests/research_test_suite.py --safety-only
```

**If any safety test fails → `git checkout main && git branch -D research/<short-name>` and go back to Step 4.2 with a different hypothesis.**

### Step 4.6 — Run Benchmark Tests

```bash
python tests/research_test_suite.py --benchmarks-only --report-file .lab/result_N.json \
  --llm-url http://localhost:1234 \
  --model qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled
```

Or run a specific benchmark to iterate faster:

```bash
python tests/research_test_suite.py --task unittest_go --json
```

(LM Studio URL and model are the defaults — you can omit them if unchanged.)

### Step 4.7 — Evaluate Results

Compare against baseline. Write results in `.lab/research_log.md`:

```markdown
### Results
- Benchmark Score: X% → Y% (delta: +Z%)
- Safety Score: 100% → 100% (delta: 0%)
- Notable: <which tasks improved/regressed>

### Decision: MERGE / DISCARD
<Reasoning>
```

**Merge criteria (ALL must be true):**
1. Safety Score = 100% (mandatory, no exceptions)
2. Benchmark Score strictly improved OR held steady with clear qualitative improvement
3. No regressions on previously passing benchmarks

### Step 4.8 — Merge or Discard

**If merging:**

```bash
git checkout main
git merge research/<short-name>
git branch -d research/<short-name>
```

**If discarding:**

```bash
git checkout main
git branch -D research/<short-name>
```

### Step 4.9 — Update Log

Append the full iteration summary to `.lab/research_log.md` including:
- Hypothesis
- What was changed (file + brief diff summary)
- Before/after scores
- Safety delta (must be 0)
- Decision and reasoning
- Time spent

Then go back to **Step 4.2**.

---

## 5. Available Benchmark Tasks

These are the 10 tasks from `s-macke/coding-agent-benchmark` that the test
suite runs. Understand what each tests so you can target improvements:

| # | Name | Tests | Key Skill |
|---|------|-------|-----------|
| 1 | `unittest_go` | Write Go unit tests | test writing, code comprehension |
| 2 | `basic_interpreter` | Implement BASIC in Go | code generation from spec |
| 3 | `asm_6502` | MOS6502 assembler in Go | complex parsing, binary output |
| 4 | `webapp_refactor` | Refactor HTML → CSS+JS | file splitting, web knowledge |
| 5 | `webapp_vite` | Set up Vite build | toolchain setup, config |
| 6 | `webapp_typescript` | Convert JS → TypeScript | type system, migration |
| 7 | `port_decompiler` | Port Python → Go | cross-language translation |
| 8 | `ioccc_analyze` | Analyze obfuscated C | reverse engineering, docs |
| 9 | `newlib_migrate` | Migrate Go library | API migration |
| 10 | `gameport_basic` | Detokenize BASIC code | binary analysis |

---

## 6. Research Log Format

Create `.lab/research_log.md` on your first iteration:

```markdown
# Research Log — Local-Coding-Agent Self-Improvement

## Meta
- Researcher: Claude (automated overnight loop)
- Agent under test: Local-Coding-Agent with Ollama
- Benchmark: s-macke/coding-agent-benchmark (10 tasks)
- Safety tests: 4 compliance checks
- Started: <timestamp>

---

## Iteration 1 — <timestamp>

### Hypothesis
...

### Changes
- `engine.py`: <what changed>
- `skills.py`: <what changed>

### Results
- Benchmark: 20% → 30% (+10%)
- Safety: 100% → 100% (0%)
- Improved: unittest_go (was FAIL, now PASS)
- Regressed: none

### Decision: MERGE
Benchmark improved with no safety regression.

---
```

---

## 7. Tips for Effective Research

1. **Start with safety tests** — always. Takes 10 seconds. Catches problems early.

2. **Run one benchmark at a time** during development (`--task unittest_go`).
   Full suite only for final validation.

3. **Read the agent's output** — when a benchmark fails, look at the agent's
   logs in the temp directory to understand WHY. The failure details in the
   test report will give you clues.

4. **Low-hanging fruit first:**
   - Better system prompts in skills.py (huge impact, zero risk)
   - Smarter `_trim_messages` in engine.py (prevents context overflow)
   - Better `_create_plan` prompts (helps the agent think before coding)
   - Improved `done()` detection (agent sometimes doesn't call done)

5. **Small changes compound.** Don't try to rewrite the engine in one pass.
   One focused improvement per iteration. Measure. Merge or discard. Repeat.

6. **If stuck, re-read the benchmark task descriptions** in
   `benchmark/README.md`. Understanding what the tasks demand reveals what
   the agent is bad at.

7. **Protect what works.** If a benchmark was passing before your change and
   fails after, that's a regression even if overall score improved. Investigate.

---

## 8. Emergency Stop

If anything goes wrong:

```bash
git checkout main
git stash
```

The operator's code on `main` is always the source of truth. Research branches
are disposable.

---

## 9. First Steps Checklist

Before entering the loop, verify the environment:

- [ ] `python tests/research_test_suite.py --safety-only` → all 4 PASS
- [ ] `ls benchmark/benchmarks/` → shows unittest, basic, asm, etc.
- [ ] `mkdir -p .lab` → create the lab directory
- [ ] `git status` → clean working tree on `main`
- [ ] LM Studio is running at `http://localhost:1234` with `qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled` loaded

Once confirmed, begin **Step 4.1** (record baseline) and enter the loop.
