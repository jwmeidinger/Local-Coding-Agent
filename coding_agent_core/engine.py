from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from .config import AgentConfig, SystemContext, TaskContext
from .deps import GitCommandError, VECTOR_MEMORY_AVAILABLE, VectorMemoryManager
from .failure_classifier import (
    FailureInfo,
    FailureTracker,
    classify_failure,
    classify_review_rejection,
    get_retry_guidance,
    _normalize_signature,
)
from .llm import ChatResponse, LLMManager, ToolCall
from .skills import Skill, SkillRegistry
from .tools import SystemUpgradeGuard, ToolRegistry


class ExecutionEngine:
    """Core execution engine - the brain of the agent."""
    
    def __init__(self, config: AgentConfig, repo, repo_path: Path):
        self.config = config
        self.repo = repo
        self.repo_path = repo_path
        self.llm = LLMManager(config)
        self.skills = SkillRegistry(config.skills_dir)
        self.tools = ToolRegistry(config, repo)
        self.logger = logging.getLogger("coding-agent")
        
        # Detect system context
        self.system_context = SystemContext.detect(repo_path)
        self.logger.info(f"System context: {self.system_context.os_name} {self.system_context.os_version}")
        
        # Initialize vector memory system
        if VECTOR_MEMORY_AVAILABLE:
            try:
                self.memory_manager = VectorMemoryManager(
                    repo_path,
                    embed_url=config.llm_url,
                    embed_model=config.embed_model,
                    embed_dims=config.embed_dims,
                )
                self.logger.info(
                    "Vector memory initialized for %s (embed_model=%r, dims=%d)",
                    repo_path.name, config.embed_model or "none (fallback)", config.embed_dims,
                )
            except Exception as e:
                self.logger.warning(
                    "Vector memory unavailable (%r). Falling back to no memory. "
                    "Start Postgres with `docker-compose up -d` to enable it.",
                    e,
                )
                self.memory_manager = None
        else:
            self.logger.warning("Vector memory not available. Install psycopg2-binary")
            self.memory_manager = None
        
        self.modified_files = []
        self.current_branch = None

        # Suggest build command if not configured
        if not config.build_command:
            self._suggest_build_command()
    
    def _suggest_build_command(self) -> None:
        """Detect build/check commands from project config files and log suggestions."""
        import json as _json

        candidates = []
        pkg_manager = "npm run"

        # Detect package manager
        if (self.repo_path / "pnpm-lock.yaml").exists():
            pkg_manager = "pnpm run"
        elif (self.repo_path / "yarn.lock").exists():
            pkg_manager = "yarn"
        elif (self.repo_path / "bun.lockb").exists():
            pkg_manager = "bun run"

        # Check package.json scripts
        pkg_json = self.repo_path / "package.json"
        if pkg_json.exists():
            try:
                data = _json.loads(pkg_json.read_text(encoding="utf-8"))
                scripts = data.get("scripts", {})

                # Words that indicate a script runs tests — never use as build_command
                _test_indicators = ("test", "jest", "vitest", "mocha", "coverage", "spec")

                def _is_test_script(script_body: str) -> bool:
                    body_lower = script_body.lower()
                    return any(w in body_lower for w in _test_indicators)

                # Prefer fast check-only commands over full builds
                preferred = ["typecheck", "type-check", "lint", "tsc", "build:check"]
                for name in preferred:
                    if name in scripts and not _is_test_script(scripts[name]):
                        candidates.append((f"{pkg_manager} {name}", f'package.json "{name}"'))

                # "check" is ambiguous — only use if it doesn't run tests
                if "check" in scripts and not _is_test_script(scripts["check"]):
                    candidates.append((f"{pkg_manager} check", 'package.json "check"'))

                if "build" in scripts and not _is_test_script(scripts["build"]):
                    candidates.append((f"{pkg_manager} build", 'package.json "build"'))
            except Exception:
                pass

        # Check for tsconfig — only suggest if no package.json scripts found
        # NEVER suggest raw `npx tsc` — it has PATH issues on Windows.
        # Always prefer the package manager's script runner.
        if not candidates:
            if (self.repo_path / "tsconfig.json").exists() or (self.repo_path / "tsconfig.electron.json").exists():
                candidates.append((f"{pkg_manager} build", "tsconfig.json detected (add a 'build' script to package.json)"))

        # Check Makefile
        makefile = self.repo_path / "Makefile"
        if makefile.exists():
            try:
                content = makefile.read_text(encoding="utf-8", errors="ignore")
                for target in ["check", "lint", "typecheck", "build"]:
                    if f"\n{target}:" in content or content.startswith(f"{target}:"):
                        candidates.append((f"make {target}", f'Makefile "{target}" target'))
            except Exception:
                pass

        # Check pyproject.toml / setup.cfg
        if (self.repo_path / "pyproject.toml").exists():
            candidates.append(("python -m py_compile", "pyproject.toml detected"))
            if (self.repo_path / "mypy.ini").exists() or (self.repo_path / ".mypy.ini").exists():
                candidates.append(("mypy .", "mypy config detected"))

        # Check Cargo.toml (Rust)
        if (self.repo_path / "Cargo.toml").exists():
            candidates.append(("cargo check", "Cargo.toml detected"))

        # Check go.mod (Go)
        if (self.repo_path / "go.mod").exists():
            candidates.append(("go build ./...", "go.mod detected"))

        if candidates:
            best = candidates[0][0]
            self.config.build_command = best
            self.logger.info(
                "Auto-detected build command: %s (from %s). "
                "Override with --build-command if needed.",
                best, candidates[0][1],
            )
            if len(candidates) > 1:
                self.logger.info("Other candidates:")
                for cmd, source in candidates[1:]:
                    self.logger.info(f"  {cmd}  (from {source})")

    def _get_project_summary(self) -> str:
        """Read key project config files and return a brief summary for context.

        This saves the agent 2-3 file_read steps by front-loading information
        it almost always needs: available scripts, language/framework, entry points.
        """
        import json as _json
        sections = []

        # --- package.json ---
        pkg_json = self.repo_path / "package.json"
        if pkg_json.exists():
            try:
                data = _json.loads(pkg_json.read_text(encoding="utf-8"))
                parts = []
                if data.get("name"):
                    parts.append(f"Name: {data['name']}")
                if data.get("version"):
                    parts.append(f"Version: {data['version']}")

                # Scripts (very useful for the agent to know)
                scripts = data.get("scripts", {})
                if scripts:
                    script_list = ", ".join(f'"{k}"' for k in scripts.keys())
                    parts.append(f"Scripts: {script_list}")

                    # Explicitly surface the test command so the agent doesn't guess
                    if "test" in scripts:
                        parts.append(
                            f'Test command: `npm test -- <testFilePattern>` '
                            f'(runs: {scripts["test"][:80]})'
                        )

                # Main entry point
                for key in ("main", "module", "entry"):
                    if data.get(key):
                        parts.append(f"Entry: {data[key]}")
                        break

                # Key dependencies (just names, not versions)
                deps = list(data.get("dependencies", {}).keys())
                if deps:
                    shown = deps[:10]
                    dep_str = ", ".join(shown)
                    if len(deps) > 10:
                        dep_str += f" (+{len(deps) - 10} more)"
                    parts.append(f"Dependencies: {dep_str}")

                dev_deps = list(data.get("devDependencies", {}).keys())
                frameworks = [d for d in dev_deps if d in (
                    "typescript", "react", "vue", "svelte", "angular",
                    "next", "nuxt", "electron", "jest", "vitest", "mocha",
                    "eslint", "prettier", "webpack", "vite", "rollup", "esbuild",
                )]
                if frameworks:
                    parts.append(f"Dev tools: {', '.join(frameworks)}")

                if parts:
                    sections.append("## package.json\n" + "\n".join(f"- {p}" for p in parts))
            except Exception:
                pass

        # --- tsconfig.json ---
        for tsconfig_name in ("tsconfig.json", "tsconfig.electron.json", "tsconfig.app.json"):
            tsconfig = self.repo_path / tsconfig_name
            if tsconfig.exists():
                try:
                    # tsconfig can have comments, so just extract key bits
                    content = tsconfig.read_text(encoding="utf-8")
                    parts = [f"TypeScript config: {tsconfig_name}"]
                    if '"outDir"' in content:
                        import re
                        m = re.search(r'"outDir"\s*:\s*"([^"]+)"', content)
                        if m:
                            parts.append(f"Output: {m.group(1)}")
                    if '"strict": true' in content:
                        parts.append("Strict mode enabled")
                    sections.append("## TypeScript\n" + "\n".join(f"- {p}" for p in parts))
                except Exception:
                    pass
                break  # Only show one tsconfig

        # --- Makefile ---
        makefile = self.repo_path / "Makefile"
        if makefile.exists():
            try:
                content = makefile.read_text(encoding="utf-8", errors="ignore")
                targets = []
                for line in content.split('\n'):
                    if line and not line.startswith('\t') and not line.startswith('#') and ':' in line:
                        target = line.split(':')[0].strip()
                        if target and not target.startswith('.'):
                            targets.append(target)
                if targets:
                    sections.append(f"## Makefile\n- Targets: {', '.join(targets[:15])}")
            except Exception:
                pass

        # --- pyproject.toml ---
        pyproject = self.repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                parts = ["Python project (pyproject.toml)"]
                if "pytest" in content:
                    parts.append("Testing: pytest")
                if "mypy" in content:
                    parts.append("Type checking: mypy")
                if "ruff" in content or "flake8" in content:
                    parts.append("Linting: " + ("ruff" if "ruff" in content else "flake8"))
                sections.append("## Python\n" + "\n".join(f"- {p}" for p in parts))
            except Exception:
                pass

        # --- requirements.txt ---
        req_txt = self.repo_path / "requirements.txt"
        if req_txt.exists() and not pyproject.exists():
            try:
                lines = [l.strip().split("==")[0].split(">=")[0] for l in
                         req_txt.read_text(encoding="utf-8").splitlines()
                         if l.strip() and not l.startswith("#")]
                if lines:
                    shown = lines[:10]
                    dep_str = ", ".join(shown)
                    if len(lines) > 10:
                        dep_str += f" (+{len(lines) - 10} more)"
                    sections.append(f"## requirements.txt\n- Packages: {dep_str}")
            except Exception:
                pass

        if sections:
            return "\n## Project Context (auto-detected)\n" + "\n\n".join(sections) + "\n"
        return ""

    def execute_task(self, task_description: str, task_id: str, repo_path: Path = None) -> bool:
        """Execute a single task from start to finish."""
        self.logger.info(f"Starting execution of task: {task_id}")

        # Ensure workspace directory exists (may have been cleaned up after previous task)
        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Ensure .coding-agent is in .git/info/exclude so it never gets committed
        self._ensure_git_exclude()

        # Reset token usage tracking for this task
        self.llm.reset_usage()
        
        # Check for system upgrade attempts
        is_safe, reason, report = SystemUpgradeGuard.is_safe_task(task_description)
        if not is_safe and report:
            self.logger.warning(f"Task blocked: {reason}")
            self._create_abort_report(task_id, task_description, reason, report)
            return False
        
        # Check/initialize codebase index
        if self.memory_manager:
            try:
                summary = self.memory_manager.get_codebase_summary()
                self.logger.info(f"Codebase memory: {summary}")
                
                # Always re-index on startup to get fresh data
                self.logger.info("Re-indexing codebase to get latest files...")
                self.memory_manager.index_codebase()
                summary = self.memory_manager.get_codebase_summary()
                self.logger.info(f"Codebase memory after indexing: {summary}")
            except Exception as e:
                self.logger.warning(f"Could not get/update codebase summary: {e}")
        
        # Create branch
        branch_name = self._create_branch(task_id)
        self.current_branch = branch_name
        
        # Reset modified files tracking
        self.modified_files = []
        
        # Initialize context
        context = TaskContext(
            task_id=task_id,
            task_description=task_description,
            branch_name=branch_name,
            repo_path=self.repo_path,
            system_info=self.system_context.to_prompt(),
        )
        
        # Detect skill
        skill = self.skills.detect_skill(task_description)
        self.logger.info(f"Using skill: {skill.name}")
        
        # Initialize failure tracker for stop-condition detection
        failure_tracker = FailureTracker(max_repeats=3)
        task_passed = False
        
        # Execution loop
        for iteration in range(1, self.config.max_iterations + 1):
            context.iteration = iteration
            self.logger.info(f"Iteration {iteration}/{self.config.max_iterations}")
            
            # Plan (or replan after feedback)
            if not context.plan or context.review_feedback:
                # On replan after failure, retrieve similar past failures
                if context.review_feedback and self.memory_manager and failure_tracker.last:
                    try:
                        ft = failure_tracker.last.failure_type
                        similar_failures = self.memory_manager.find_similar_failures(
                            ft, context.task_description, limit=2
                        )
                        if similar_failures:
                            past_ctx = self.memory_manager.format_past_runs(similar_failures)
                            context.review_feedback += (
                                f"\n\n{past_ctx}"
                            )
                            self.logger.info(
                                "Injected %d similar past failures into replan context",
                                len(similar_failures),
                            )
                    except Exception as e:
                        self.logger.warning(f"Past-failure retrieval failed: {e}")

                context.plan = self._create_plan(context, skill)
                self.logger.info(f"Plan created:\n{context.plan}")
            
            # Execute
            success = self._execute_plan(context, skill)
            if not success:
                self.logger.warning(f"Execution failed on iteration {iteration}")
                # Surface inner-loop failure context to the replan
                # Look for STUCK entries in the execution log
                stuck_entries = [
                    e for e in context.execution_log if e.startswith("STUCK:")
                ]
                if stuck_entries:
                    stuck_summary = "\n".join(stuck_entries[-3:])
                    context.review_feedback = (
                        f"The previous iteration was aborted because the agent got "
                        f"stuck in a loop:\n{stuck_summary}\n\n"
                        f"## Strategy for this iteration\n"
                        f"1. Use web_search to look up the specific framework/mocking "
                        f"issue (e.g. 'jest mock axios isAxiosError', "
                        f"'jest fake timers nested setTimeout async')\n"
                        f"2. If specific tests keep failing after a genuine attempt, "
                        f"REMOVE or SKIP those tests with `it.skip(...)` and add a "
                        f"TODO comment explaining why. Passing 20/24 tests is better "
                        f"than failing all 24.\n"
                        f"3. Do NOT repeat the same approach that failed last time."
                    )
                continue
            
            # Skip review if nothing was written — go straight to next iteration
            if not self._has_changes():
                self.logger.warning(
                    "Iteration %d produced no file changes — skipping review, retrying",
                    iteration,
                )
                context.review_feedback = (
                    "CRITICAL: The previous iteration read files but never called "
                    "file_edit or file_write. You MUST write code this iteration. Do NOT just read "
                    "files again. Use the information you already gathered to make the "
                    "changes immediately with file_edit."
                )
                if iteration >= self.config.max_iterations:
                    self.logger.warning("Max iterations reached without any file changes")
                    break
                continue
            
            # Review
            review_result = self._review_changes(context, skill)
            
            if self._review_passed(review_result):
                self.logger.info("Task passed review")
                task_passed = True
                break
            else:
                # Classify the review rejection
                failure_info = classify_review_rejection(review_result)
                should_stop = failure_tracker.record(failure_info)

                # Build targeted feedback: review text + retry guidance
                guidance = get_retry_guidance(failure_info.failure_type)
                context.review_feedback = (
                    f"{review_result}\n\n"
                    f"## Retry guidance\n{guidance}"
                )

                if self.config.verbose:
                    self.logger.info(f"Review feedback:\n{review_result}")
                else:
                    self.logger.info(f"Review feedback: {review_result[:200]}...")

                if should_stop:
                    self.logger.warning(
                        "Stopping: same failure repeated %d times without improvement "
                        "(type=%s)", failure_tracker.max_repeats,
                        failure_info.failure_type,
                    )
                    break
                if iteration >= self.config.max_iterations:
                    self.logger.warning("Max iterations reached without PASS")
                    break
        
        # --- Determine outcome for memory ---
        if task_passed:
            outcome = "success"
        elif self._has_changes():
            outcome = "partial"
        else:
            outcome = "failure"

        fail_type, fail_summary = failure_tracker.summary_for_storage()

        # Build a meaningful resolution string
        resolution = None
        if task_passed and fail_type:
            # The task passed after failures — capture what fixed it
            parts = [f"Passed after {len(failure_tracker._history)} failure(s)."]
            if context.done_message:
                # Truncate the done message to the most useful bit
                done_short = context.done_message.strip().splitlines()[0][:120]
                parts.append(f"Agent summary: {done_short}")
            resolution = " ".join(parts)

        # Build a summarized execution log for storage
        # Keep the last 8 steps, each truncated — enough to understand
        # the approach without blowing up storage
        exec_log_summary = None
        if context.execution_log:
            tail = context.execution_log[-8:]
            log_lines = [entry[:100] for entry in tail]
            exec_log_summary = "\n".join(log_lines)
            # Cap at ~1000 chars total
            if len(exec_log_summary) > 1000:
                exec_log_summary = exec_log_summary[:1000] + "\n...(truncated)"

        # Check for changes
        if not self._has_changes():
            self.logger.warning("No changes were made")
            # Still record the failed attempt in memory
            if self.memory_manager:
                try:
                    self.memory_manager.update_for_task(
                        [], task_description, self.current_branch, skill.name,
                        outcome=outcome, failure_type=fail_type,
                        failure_summary=fail_summary,
                        execution_log=exec_log_summary,
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to update memory: {e}")
            self.logger.info(self.llm.get_usage_summary())
            return False
        
        # Get list of modified files before committing
        _stray_names = {
            "cat", "test-output.txt", "test-output.txt;",
            "test_output.txt", "jest_output.txt",
            "temp-write-script.js",
        }
        try:
            modified = self.repo.git.diff("--name-only").split('\n')
            self.modified_files = [
                f.strip() for f in modified
                if f.strip()
                and not f.strip().startswith(".coding-agent")
                and f.strip() not in _stray_names
            ]
            new_files = [
                f for f in self.repo.untracked_files
                if not f.startswith(".coding-agent")
                and f not in _stray_names
            ]
            self.modified_files.extend(new_files)
        except GitCommandError:
            self.modified_files = []
        
        # Commit
        if self.config.auto_commit:
            self._commit_changes(task_id)
        
        # Update memory with modified files and outcome
        if self.memory_manager:
            self.logger.info(f"Updating memory for {len(self.modified_files)} modified files")
            try:
                self.memory_manager.update_for_task(
                    self.modified_files,
                    task_description,
                    self.current_branch,
                    skill.name,
                    outcome=outcome,
                    failure_type=fail_type,
                    failure_summary=fail_summary,
                    resolution=resolution,
                    execution_log=exec_log_summary,
                )
            except Exception as e:
                self.logger.warning(f"Failed to update memory: {e}")
        
        # Log token usage summary for this task
        self.logger.info(self.llm.get_usage_summary())

        return True
    
    def _create_abort_report(self, task_id: str, task_description: str, reason: str, report: str) -> None:
        """Create an abort report file when a task is blocked."""
        try:
            # Create reports directory
            reports_dir = self.repo_path / ".coding-agent" / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Create report filename
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            report_file = reports_dir / f"ABORT-{task_id[:20]}-{timestamp}.md"
            
            # Write report
            report_file.write_text(report, encoding="utf-8")
            self.logger.info(f"Abort report created: {report_file}")
            
            # Also log to main log
            self.logger.warning(f"Task '{task_id}' was aborted. See {report_file} for details.")
        except Exception as e:
            self.logger.error(f"Failed to create abort report: {e}")
    
    def _create_branch(self, task_id: str) -> str:
        """Create a git branch for the task with collision detection and dirty tree handling."""
        slug = re.sub(r"[^a-z0-9-]+", "-", task_id.lower()).strip("-")
        timestamp = datetime.now().strftime("%m%d-%H%M")
        base_branch_name = f"{self.config.branch_prefix}{slug}"
        stashed = False

        try:
            # Check for uncommitted changes
            if self.repo.is_dirty(untracked_files=True):
                self.logger.warning("Working tree has uncommitted changes, stashing them...")
                self.repo.git.stash('push', '-m', f'auto-stash before {task_id}')
                stashed = True

            # Generate unique branch name
            existing_branches = [b.name for b in self.repo.branches]
            branch_name = f"{base_branch_name}-{timestamp}"
            counter = 1
            while branch_name in existing_branches:
                branch_name = f"{base_branch_name}-{timestamp}-{counter}"
                counter += 1

            # Checkout base branch and create new branch
            self.repo.git.checkout(self.config.base_branch)
            self.repo.git.checkout("-b", branch_name)
            self.logger.info(f"Created branch: {branch_name}")
            return branch_name

        except Exception as e:
            if stashed:
                try:
                    self.repo.git.stash('pop')
                    self.logger.info("Restored stashed changes after branch creation failure")
                except GitCommandError as stash_err:
                    self.logger.warning(f"Could not restore stash: {stash_err}")
            raise RuntimeError(f"Failed to create branch: {e}") from e
    
    def _create_plan(self, context: TaskContext, skill: Skill) -> str:
        """Create an execution plan grounded in the actual file tree.

        Runs file_tree first so the LLM cannot hallucinate filenames.
        """
        # Always get the real file tree so the plan is grounded
        file_tree = self.tools.execute('file_tree(path=".")')
        self.logger.info("File tree collected for planning phase")

        # Optionally enrich with vector search results
        context_info = ""
        if self.memory_manager:
            try:
                results = self.memory_manager.search_codebase(
                    context.task_description,
                    limit=10
                )
                if results:
                    context_info = "Relevant Code Files (from memory):\n"
                    for r in results[:5]:
                        context_info += f"\n  {r['file_path']} (score: {r['combined_score']:.2f})\n"
                        context_info += f"     {r.get('summary') or ''}\n"
                        kf = r.get('key_functions') or []
                        if kf:
                            context_info += f"     Functions: {', '.join(kf[:5])}\n"
                    context_info += "\n"
            except Exception as e:
                self.logger.warning(f"Vector search failed: {e}")

        # Retrieve similar past runs for context
        past_runs_context = ""
        if self.memory_manager:
            try:
                past_runs = self.memory_manager.find_similar_tasks(
                    context.task_description, limit=3
                )
                # Only include runs with meaningful similarity
                relevant = [r for r in past_runs if r.get("similarity", 0) > 0.3]
                if relevant:
                    past_runs_context = self.memory_manager.format_past_runs(relevant)
                    self.logger.info(
                        "Found %d relevant past runs for planning", len(relevant)
                    )
            except Exception as e:
                self.logger.warning(f"Past-run retrieval failed: {e}")

        prompt = f"""{context.system_info}

## Project File Tree (REAL — use only these paths)
{file_tree}

{context_info}{past_runs_context}{skill.planning_prompt.format(task_description=context.task_description)}

IMPORTANT RULES:
- You may ONLY reference files/directories that appear in the file tree above.
- If the task mentions a file that does NOT exist, say so and adapt.
- Keep the plan SHORT and ACTIONABLE (max 30 lines).
- List: (1) files to read, (2) files to create or modify, (3) commands to run.
- Do NOT write code in the plan."""
        if context.review_feedback:
            prompt += f"\n\nPrevious review feedback to address:\n{context.review_feedback}"

        return self.llm.generate(prompt, skill.system_prompt)
    
    @staticmethod
    def _summarize_result(tool_call: str, result: str, summary_chars: int = 600) -> str:
        """Create a short summary of a tool result for older history entries.

        For file reads: keeps the header line (path + line count) plus the first
        and last few lines so the agent remembers the file structure.
        For other tools: keeps the first ``summary_chars`` characters.
        """
        if len(result) <= summary_chars:
            return result

        lines = result.splitlines()

        if tool_call.startswith("file_read(") and len(lines) > 20:
            header = lines[0]  # e.g. "[src/app.js — 450 lines]"
            top = "\n".join(lines[1:11])
            bottom = "\n".join(lines[-5:])
            omitted = len(lines) - 16
            return (
                f"{header}\n{top}\n"
                f"  ... ({omitted} lines — already read, use start_line/end_line to revisit) ...\n"
                f"{bottom}"
            )

        return result[:summary_chars] + f"\n...(summary, {len(result) - summary_chars} chars omitted)"

    def _execute_plan(self, context: TaskContext, skill: Skill) -> bool:
        """Execute the plan step by step using multi-turn chat with tools.

        Uses native tool calling when the model supports it, with automatic
        fallback to text-based extraction when it doesn't.

        Context strategy (multi-turn):
        - Each tool call and result becomes a real message in the conversation.
        - Older tool results are summarized to manage context window size.
        - The LLM sees the full conversation history, not a reconstructed prompt.
        """
        # --- Collect file tree ONCE at the start ---
        file_tree = self.tools.execute('file_tree(path=".")')

        relevant_files = []
        if self.memory_manager:
            try:
                results = self.memory_manager.search_codebase(
                    context.task_description,
                    limit=5
                )
                relevant_files = [r['file_path'] for r in results if r['combined_score'] > 0.5]
            except Exception as e:
                self.logger.warning(f"Could not search codebase: {e}")

        memory_context = ""
        if relevant_files:
            memory_context = f"""Most Relevant Files (from memory):
{chr(10).join([f"  - {f}" for f in relevant_files[:5]])}

"""

        plan_text = context.plan or ""
        if len(plan_text) > 3000:
            plan_text = plan_text[:3000] + "\n...(plan truncated)"

        # --- Build initial messages ---
        # --- Windows-specific instructions ---
        windows_rules = ""
        if self.system_context.os_name == "Windows":
            windows_rules = """
## Windows Environment Rules
- You are running on Windows. Use Windows-compatible paths and commands.
- Do NOT use Unix paths like ./node_modules/.bin/jest or /c/Users/...
- Use backslash paths or forward-slash relative paths: node_modules\\.bin\\jest.cmd
- To run tests, ALWAYS use `npm test -- <testFilePattern>`. Do NOT use `npx jest` — it does not work on Windows.
- Do NOT use shell pipes like | head, | tail, | grep — they are unreliable on Windows.
- Do NOT use cd /c/... syntax — that is Git Bash Unix-style and won't work.
- Do NOT chain commands with semicolons (;) or use $? — use separate bash calls instead.
- Do NOT install packages (npm install <pkg>, pip install <pkg>, etc.) — only use what's already available.
"""

        system_message = skill.system_prompt + f"""

{context.system_info}

You are working in repository: {context.repo_path}
All file paths are relative to this directory.
{windows_rules}
## Rules
1. Use file_tree output (provided below) instead of repeated list_files calls.
2. Use file_edit to modify existing files (NOT file_write — file_write is for NEW files only).
3. If file_edit fails (text not found), re-read the file to get the exact text.
4. Do NOT invent file paths — only use paths from the file tree.
5. When all work is done, call git_diff() to review, then done().
6. FOLLOW EXISTING PATTERNS in the code. If the codebase already handles epic_title/epic_url, use the same pattern for milestone_title/milestone_url. Do NOT web_search for API docs when you can see the pattern in the code you already read.
7. Limit web_search to 2 calls maximum. If you need API docs, read ONE page. Do not search repeatedly for the same topic.
8. Do NOT install new packages or dependencies. Only use packages already in package.json / requirements.txt.
9. NEVER use `as any` to call methods that don't exist in the codebase or type definitions. If TypeScript says a method doesn't exist, it doesn't exist at runtime either. Casting to `any` just hides the error — it will crash when the code actually runs. Instead, look for the correct method name in the existing code or type definitions.
10. Do NOT include proprietary code, internal URLs, file contents, API keys, or project-specific identifiers in web_search queries. Keep searches to general technical questions only (e.g. "gitbeaker pagination API" not "how to paginate api.GroupLabels.all in our gitlab-reports app").
"""

        # --- Auto-detect project context from config files ---
        project_summary = self._get_project_summary()

        # --- Previous iteration context ---
        # If this is iteration 2+, tell the agent what it already discovered
        prev_iteration_context = ""
        if context.iteration > 1:
            parts = []
            if context.files_read:
                parts.append(
                    "Files you already read in previous iterations (DO NOT re-read these):\n"
                    + "\n".join(f"  - {f}" for f in sorted(context.files_read))
                )
            if context.done_message:
                parts.append(
                    f"Your previous completion summary:\n{context.done_message}"
                )
            if context.review_feedback:
                parts.append(
                    f"Reviewer feedback to address:\n{context.review_feedback}"
                )
            if parts:
                prev_iteration_context = (
                    "\n## Previous Iteration Context\n"
                    + "\n\n".join(parts)
                    + "\n\nIMPORTANT: Use the context above. Do NOT re-read files you already read. "
                    "Focus on addressing the reviewer's feedback.\n"
                )

        user_message = f"""{memory_context}## Project File Tree
{file_tree}
{project_summary}{prev_iteration_context}
Task: {context.task_description}

Plan:
{plan_text}

Begin working. Call your first tool now."""

        # The conversation history — this is what makes it multi-turn
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        # Get native tool schemas for the LLM
        tool_schemas = self.tools.get_tool_schemas_native()

        max_steps = 25
        max_consecutive_errors = getattr(self.config, "max_consecutive_errors", 2)

        no_tool_count = 0
        error_count = 0
        write_count = 0          # Successful (non-reverted) writes
        write_attempts = 0       # Total write attempts (including reverted)
        build_verify_enabled = self.config.verify_after_write and bool(self.config.build_command)
        # Use persistent set from context — survives across iterations
        files_already_read = context.files_read
        # Cache grep results: key = (pattern, path, include), value = result text
        grep_cache: dict[tuple, str] = {}
        # Track web_search queries to prevent search spirals
        web_search_queries: list[str] = []
        MAX_WEB_SEARCHES = 3

        # Inner-loop failure tracker — detects repeated tool-level failures
        # within a single _execute_plan invocation
        inner_tracker = FailureTracker(max_repeats=3)
        MAX_EDIT_FAILURES = 4  # Stop nudging after this many file_edit errors

        # Test run tracking — detects edit→test→fail→edit→test→fail loops
        test_run_count = 0           # Total test invocations
        failed_test_runs = 0         # Consecutive failed test runs
        last_test_failure_sig = ""   # Normalized signature of last test failure
        MAX_TEST_RUNS = 4            # Cap on test invocations per execution
        edits_since_last_test = 0    # Track edits between test runs
        MAX_EDITS_BEFORE_TEST = 4    # Nudge to test after this many edits

        # Pre-compiled patterns for detecting test failures in bash output
        _JEST_FAIL_PATTERNS = [
            re.compile(r"FAIL\s+\S+", re.IGNORECASE),
            re.compile(r"Tests:\s+\d+ failed", re.IGNORECASE),
            re.compile(r"Test Suites:.*failed", re.IGNORECASE),
            re.compile(r"FAILED", re.IGNORECASE),
        ]

        # Step budget thresholds for escalating write nudges
        NUDGE_SOFT = int(max_steps * 0.5)
        NUDGE_HARD = int(max_steps * 0.75)
        NUDGE_FINAL = int(max_steps * 0.9)

        for step in range(max_steps):
            # --- Inject nudge if the agent hasn't written anything ---
            # But DON'T nag if the model HAS been trying (write_attempts > 0)
            # — that means build failures are reverting its work, not laziness.
            if write_count == 0 and write_attempts == 0 and step >= NUDGE_SOFT:
                remaining = max_steps - step
                if step >= NUDGE_FINAL:
                    nudge = (
                        f"FINAL WARNING: You have {remaining} steps left and have "
                        f"NOT written any code yet. Call file_edit or file_write NOW "
                        f"or the task will fail."
                    )
                elif step >= NUDGE_HARD:
                    nudge = (
                        f"WARNING: You have used {step}/{max_steps} steps reading "
                        f"files but have NOT called file_edit yet. You MUST start "
                        f"making changes NOW."
                    )
                else:
                    nudge = (
                        f"Note: You have used {step} of {max_steps} steps. "
                        f"Start making changes with file_edit soon."
                    )
                messages.append({"role": "user", "content": nudge})

            # --- Nudge if too many edits without running tests ---
            # Only fires after at least one failed test run (agent is in fix loop)
            if (failed_test_runs > 0
                    and edits_since_last_test >= MAX_EDITS_BEFORE_TEST):
                nudge = (
                    f"You have made {edits_since_last_test} edits since the last "
                    f"test run. Run tests now to verify your changes before "
                    f"making more edits."
                )
                messages.append({"role": "user", "content": nudge})

            # --- Manage context window: summarize old tool results ---
            self._trim_messages(messages)

            # --- Call the LLM ---
            response = self.llm.chat_with_tools(messages, tool_schemas)

            # Handle empty response (timeout / error)
            if response.is_empty:
                error_count += 1
                self.logger.warning(
                    "LLM returned empty response (%d/%d consecutive errors)",
                    error_count, max_consecutive_errors,
                )
                if error_count >= max_consecutive_errors:
                    self.logger.error(
                        "Aborting: %d consecutive LLM failures.", error_count
                    )
                    return False
                # Add a retry nudge
                messages.append({
                    "role": "user",
                    "content": "Your last response was empty. Call a tool now.",
                })
                continue

            error_count = 0

            # --- Handle text-only response (no tool call) ---
            if not response.is_tool_call:
                if self.config.verbose:
                    context.execution_log.append(f"Step {step + 1}:\n{response.text}")
                else:
                    context.execution_log.append(f"Step {step + 1}: {response.text[:200]}...")

                # Append the assistant's text reply to history
                messages.append({"role": "assistant", "content": response.text})

                no_tool_count += 1
                if no_tool_count >= 3:
                    self.logger.warning(
                        "Model not producing tool calls after %d attempts", no_tool_count
                    )
                    return True

                messages.append({
                    "role": "user",
                    "content": "You MUST call a tool now. Do not explain — just call the tool.",
                })
                continue

            # --- Handle tool call ---
            no_tool_count = 0
            tc = response.tool_call
            call_display = tc.to_call_string()
            self.logger.info(f"Step {step + 1}: {call_display}")
            context.execution_log.append(f"Step {step + 1}: {call_display}")

            # Done tool
            if tc.name == "done":
                self.logger.info("Task completed - done tool called")
                # Store the agent's completion summary for the reviewer
                context.done_message = tc.arguments.get("message", "")
                # Append assistant message for clean history
                self._append_tool_call_messages(messages, response, "DONE")
                return True

            # Deduplicate full-file reads
            if tc.name == "file_read":
                read_path = tc.arguments.get("path", "")
                has_range = "start_line" in tc.arguments or "end_line" in tc.arguments
                if read_path and not has_range and read_path in files_already_read:
                    msg = (
                        f"You already read {read_path}. The content is in the conversation. "
                        f"Use start_line/end_line to revisit specific sections."
                    )
                    self._append_tool_call_messages(messages, response, msg)
                    self.logger.info(f"Skipped duplicate read of {read_path}")
                    continue
                if read_path and not has_range:
                    files_already_read.add(read_path)

            # Deduplicate grep calls with similar patterns
            if tc.name == "grep":
                grep_key = (
                    tc.arguments.get("pattern", ""),
                    tc.arguments.get("path", "."),
                    tc.arguments.get("include", ""),
                )
                skip_grep = False
                # Check for exact duplicate
                if grep_key in grep_cache:
                    msg = (
                        f"You already searched for this pattern. Previous results:\n"
                        f"{grep_cache[grep_key]}"
                    )
                    self._append_tool_call_messages(messages, response, msg)
                    self.logger.info(f"Skipped duplicate grep: {grep_key[0]}")
                    skip_grep = True
                else:
                    # Check for similar pattern (substring of a previous search)
                    current_pattern = grep_key[0]
                    for cached_key, cached_result in grep_cache.items():
                        cached_pattern = cached_key[0]
                        same_scope = grep_key[1] == cached_key[1]
                        if same_scope and current_pattern and cached_pattern:
                            if current_pattern in cached_pattern or cached_pattern in current_pattern:
                                msg = (
                                    f"Similar search already done (pattern: '{cached_pattern}'). "
                                    f"Previous results:\n{cached_result}\n\n"
                                    f"Use file_read with start_line/end_line to examine specific matches."
                                )
                                self._append_tool_call_messages(messages, response, msg)
                                self.logger.info(
                                    f"Skipped similar grep: '{current_pattern}' ~= '{cached_pattern}'"
                                )
                                skip_grep = True
                                break
                if skip_grep:
                    continue

            # Limit and deduplicate web_search calls
            if tc.name == "web_search":
                query = tc.arguments.get("query", "")
                query_lower = query.lower().strip()

                # Check if we've already searched for something very similar
                skip_search = False
                for prev_query in web_search_queries:
                    prev_lower = prev_query.lower().strip()
                    # Exact or near-duplicate
                    if query_lower == prev_lower:
                        skip_search = True
                        break
                    # Significant overlap (>60% of words in common)
                    q_words = set(query_lower.split())
                    p_words = set(prev_lower.split())
                    if q_words and p_words:
                        overlap = len(q_words & p_words) / max(len(q_words), len(p_words))
                        if overlap > 0.6:
                            skip_search = True
                            break

                if skip_search:
                    msg = (
                        f"You already searched for something very similar. "
                        f"Previous searches: {web_search_queries}\n"
                        f"Use the information you already have from the code. "
                        f"Follow existing patterns in the codebase instead of searching."
                    )
                    self._append_tool_call_messages(messages, response, msg)
                    self.logger.info(f"Skipped duplicate web_search: {query}")
                    continue

                if len(web_search_queries) >= MAX_WEB_SEARCHES:
                    msg = (
                        f"Web search limit reached ({MAX_WEB_SEARCHES} searches). "
                        f"You have enough information. Follow existing code patterns "
                        f"and start making changes with file_edit."
                    )
                    self._append_tool_call_messages(messages, response, msg)
                    self.logger.info(f"Web search limit reached, blocked: {query}")
                    continue

                web_search_queries.append(query)

            # Track mutations
            is_mutation = tc.name in ("file_write", "file_edit")
            if is_mutation:
                write_count += 1
                write_attempts += 1
                edits_since_last_test += 1

            # --- Execute the tool ---
            result = self.tools.execute_by_name(tc.name, tc.arguments)

            # Cache grep results for deduplication
            if tc.name == "grep" and "Error" not in result:
                grep_key = (
                    tc.arguments.get("pattern", ""),
                    tc.arguments.get("path", "."),
                    tc.arguments.get("include", ""),
                )
                grep_cache[grep_key] = result

            # --- Classify inner-loop failures from tool results ---
            tool_had_error = False
            if is_mutation and "Error" in result:
                # file_edit / file_write returned an error (e.g. text not found)
                tool_failure = classify_failure(result)
                should_stop_inner = inner_tracker.record(tool_failure)
                tool_had_error = True

                if should_stop_inner:
                    self.logger.warning(
                        "Inner loop: same tool error repeated %d times (type=%s). "
                        "Aborting execution for this iteration.",
                        inner_tracker.max_repeats, tool_failure.failure_type,
                    )
                    result += (
                        "\n\n⛔ This same error has occurred multiple times. "
                        "Re-read the file to get the current exact content, "
                        "then try a different approach."
                    )
                    self._append_tool_call_messages(messages, response, result)
                    # Surface inner-loop failure info to the outer loop
                    context.execution_log.append(
                        f"STUCK: {tool_failure.failure_type} — {tool_failure.summary[:100]}"
                    )
                    return False

                # Inject targeted guidance on repeated (but not yet stuck) failures
                if inner_tracker._seen.get(
                    (tool_failure.failure_type, 
                     _normalize_signature(tool_failure.summary)), 0
                ) >= 2:
                    guidance = get_retry_guidance(tool_failure.failure_type)
                    result += f"\n\n## Retry guidance\n{guidance}"

            elif tc.name == "bash":
                cmd = tc.arguments.get("command", "").lower()
                is_test_run = any(kw in cmd for kw in [
                    "npm test", "npx jest", "pytest", "python -m pytest",
                    "cargo test", "go test", "mvn test", "gradle test",
                ])

                if is_test_run:
                    test_run_count += 1
                    edits_since_last_test = 0

                    # Detect test failure from output
                    test_failed = any(pat.search(result) for pat in _JEST_FAIL_PATTERNS)

                    if test_failed:
                        test_failure = classify_failure(result)
                        inner_tracker.record(test_failure)
                        current_sig = _normalize_signature(test_failure.summary)
                        same_as_last = (current_sig == last_test_failure_sig)
                        last_test_failure_sig = current_sig
                        failed_test_runs += 1

                        # Parse test pass/fail counts from output
                        tests_passed = 0
                        tests_failed_count = 0
                        tests_total = 0
                        count_match = re.search(
                            r"Tests:\s+(\d+)\s+failed.*?(\d+)\s+passed.*?(\d+)\s+total",
                            result
                        )
                        if not count_match:
                            # Try pytest format: "X passed, Y failed"
                            count_match = re.search(
                                r"(\d+)\s+passed.*?(\d+)\s+failed",
                                result
                            )
                            if count_match:
                                tests_passed = int(count_match.group(1))
                                tests_failed_count = int(count_match.group(2))
                                tests_total = tests_passed + tests_failed_count
                        else:
                            tests_failed_count = int(count_match.group(1))
                            tests_passed = int(count_match.group(2))
                            tests_total = int(count_match.group(3))

                        pass_rate = (
                            tests_passed / tests_total if tests_total > 0 else 0
                        )

                        if same_as_last and failed_test_runs >= 3:
                            # Same test failure 3+ times — force a different approach
                            self.logger.warning(
                                "Test failure repeated %d times with same error. "
                                "Injecting re-read guidance.",
                                failed_test_runs,
                            )
                            # Allow re-reading source files — the agent NEEDS to
                            # re-read to break out of the loop. Clear the dedup set.
                            files_already_read.clear()
                            self.logger.info(
                                "Cleared file read cache to allow re-reading after "
                                "repeated test failures."
                            )
                            # Unlock web search — the agent is stuck on a pattern
                            # it can't figure out from the codebase alone
                            web_search_queries.clear()
                            MAX_WEB_SEARCHES = max(MAX_WEB_SEARCHES, 2)
                            self.logger.info(
                                "Unlocked web search (limit=%d, cleared query cache) "
                                "after repeated test failures.",
                                MAX_WEB_SEARCHES,
                            )
                            result += (
                                "\n\n⛔ SAME TEST FAILURE %d TIMES IN A ROW. "
                                "Your previous edits are NOT fixing the root cause.\n"
                                "STOP editing and do the following:\n"
                                "1. Use web_search to look up the specific error message "
                                "or testing pattern that is failing — search for the "
                                "framework name + the error (web search limit has been "
                                "increased)\n"
                                "2. Use file_read to re-read the SOURCE file being tested "
                                "— the read cache has been cleared so you can re-read it\n"
                                "3. If mocking is the issue, look at how other test files "
                                "in this project mock the same dependency\n"
                                "4. If specific tests STILL fail after trying a new "
                                "approach, use the test framework's skip mechanism to "
                                "skip them and add a TODO comment — passing most tests "
                                "is better than failing all"
                                % failed_test_runs
                            )
                        elif same_as_last and failed_test_runs >= 2:
                            # Same failure twice — inject guidance
                            guidance = get_retry_guidance(test_failure.failure_type)
                            result += (
                                f"\n\n⚠ This is the same test failure as last time. "
                                f"Your last edit did not fix it.\n"
                                f"## What to do differently\n{guidance}\n"
                                f"Consider re-reading the source file to understand "
                                f"what the code actually does before editing the test again."
                            )
                    else:
                        # Tests passed — reset failure tracking
                        failed_test_runs = 0
                        last_test_failure_sig = ""

                    # Cap total test runs to prevent burn
                    if test_run_count >= MAX_TEST_RUNS and failed_test_runs > 0:
                        # Check if most tests pass — partial success is better
                        # than total failure
                        if pass_rate >= 0.8 and tests_passed > 0:
                            self.logger.info(
                                "Partial success: %d/%d tests passing (%.0f%%). "
                                "Instructing agent to skip failing tests and commit.",
                                tests_passed, tests_total, pass_rate * 100,
                            )
                            result += (
                                f"\n\n✅ PARTIAL SUCCESS: {tests_passed}/{tests_total} "
                                f"tests are passing ({pass_rate:.0%}).\n"
                                f"You have been unable to fix the remaining "
                                f"{tests_failed_count} test(s) after {test_run_count} attempts.\n"
                                f"DO THIS NOW:\n"
                                f"1. Use file_edit to mark ONLY the failing tests as "
                                f"skipped using the test framework's skip mechanism "
                                f"— do NOT remove them\n"
                                f"2. Add a TODO comment on each skipped test explaining "
                                f"why it was skipped and what needs manual review\n"
                                f"3. Run tests ONE more time to confirm all pass\n"
                                f"4. Call done() with a summary noting which tests "
                                f"were skipped and why"
                            )
                            # Give the agent more steps to do the skip+verify
                            MAX_TEST_RUNS += 1
                        else:
                            self.logger.warning(
                                "Test run limit reached (%d runs, %d consecutive failures). "
                                "Aborting inner loop to trigger replan.",
                                test_run_count, failed_test_runs,
                            )
                            result += (
                                f"\n\n⛔ You have run tests {test_run_count} times with "
                                f"{failed_test_runs} consecutive failures. "
                                f"Aborting this iteration to try a different approach."
                            )
                            self._append_tool_call_messages(messages, response, result)
                            context.execution_log.append(
                                f"STUCK: test loop — {test_run_count} runs, "
                                f"{failed_test_runs} failures, last: {last_test_failure_sig[:80]}"
                            )
                            return False

                elif "Error" in result and "error" in result.lower():
                    # Non-test bash command failed — classify for tracking
                    bash_failure = classify_failure(result)
                    inner_tracker.record(bash_failure)

            # --- Auto-verify after mutations ---
            if is_mutation and not tool_had_error and build_verify_enabled:
                verify_output, is_tool_broken = self._run_build_check()
                if is_tool_broken:
                    # The build tool itself can't run (PATH issue, missing binary, etc.)
                    # Keep the edit — the code is probably fine, the environment is broken.
                    build_verify_enabled = False
                    self.logger.warning(
                        "Build tool is not available (%s). "
                        "Disabling auto-verify for the rest of this task. "
                        "The edit has been KEPT.",
                        verify_output[:100],
                    )
                    result += (
                        f"\n\n⚠ Build tool not available ({self.config.build_command}). "
                        f"Auto-verify disabled. Your edit has been kept."
                    )
                elif verify_output:
                    # Real build error — the code change broke something. Revert.
                    written_path = tc.arguments.get("path", "")
                    if written_path:
                        revert_result = self.tools.execute_by_name("revert_file", {"path": written_path})

                        # Classify the build failure for targeted guidance
                        build_failure = classify_failure(verify_output)
                        should_stop_inner = inner_tracker.record(build_failure)
                        guidance = get_retry_guidance(build_failure.failure_type)
                        file_hint = ""
                        if build_failure.file_hint and build_failure.line_hint:
                            file_hint = (
                                f"\nError location: {build_failure.file_hint}"
                                f" line {build_failure.line_hint}"
                            )
                        elif build_failure.file_hint:
                            file_hint = f"\nError location: {build_failure.file_hint}"

                        result += (
                            f"\n\n⚠ BUILD FAILED after this change — file has been auto-reverted.\n"
                            f"Failure type: {build_failure.failure_type}\n"
                            f"Build output:\n{verify_output}\n"
                            f"Revert: {revert_result}{file_hint}\n\n"
                            f"## What to do next\n{guidance}"
                        )
                        write_count -= 1
                        self.logger.warning(
                            "Auto-reverted %s after build failure (type=%s)",
                            written_path, build_failure.failure_type,
                        )

                        if should_stop_inner:
                            self.logger.warning(
                                "Inner loop: same build failure repeated %d times. "
                                "Aborting execution for this iteration.",
                                inner_tracker.max_repeats,
                            )
                            context.execution_log.append(
                                f"STUCK: {build_failure.failure_type} — {build_failure.summary[:100]}"
                            )
                            self._append_tool_call_messages(messages, response, result)
                            return False

            if self.config.verbose:
                self.logger.info(f"Result:\n{result}")
            else:
                # For bash commands, show the last 10 lines (test summaries,
                # error messages, etc. are always at the end)
                if tc.name == "bash" and len(result) > 200:
                    tail_lines = result.rstrip().splitlines()[-10:]
                    tail_text = "\n".join(tail_lines)
                    self.logger.info(
                        f"Result (last 10 lines):\n{tail_text}"
                    )
                else:
                    self.logger.info(f"Result: {result[:200]}...")

            if "not found" in result.lower():
                result += "\n\nIMPORTANT: Use file_tree or list_files to find the correct path."

            # --- Append tool call + result to conversation history ---
            self._append_tool_call_messages(messages, response, result)

        if write_count == 0:
            self.logger.warning(
                "Execution used all %d steps without calling file_write/file_edit "
                "(read-loop detected: %d files read).",
                max_steps, len(files_already_read),
            )

        return True

    def _append_tool_call_messages(
        self,
        messages: list,
        response: ChatResponse,
        result: str,
    ) -> None:
        """Append the assistant's tool call + tool result to the conversation.

        For native tool calling, uses the proper message format.
        For text fallback, simulates it with assistant + user messages.
        """
        tc = response.tool_call
        if not tc:
            return

        if self.llm.supports_tools and tc.raw_id:
            # Native format: assistant message with tool_calls + tool result message
            import json
            messages.append({
                "role": "assistant",
                "content": response.text or None,
                "tool_calls": [{
                    "id": tc.raw_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.raw_id,
                "content": result,
            })
        elif self.llm.supports_tools and self.llm.server_type == "ollama":
            # Ollama native: uses a slightly different format
            import json
            messages.append({
                "role": "assistant",
                "content": response.text or "",
                "tool_calls": [{
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "content": result,
            })
        else:
            # Text fallback: simulate with assistant text + user providing result
            call_str = tc.to_call_string()
            messages.append({
                "role": "assistant",
                "content": response.text or call_str,
            })
            messages.append({
                "role": "user",
                "content": f"Tool result for {tc.name}:\n{result}\n\nCall the next tool (or done() if finished):",
            })

    def _trim_messages(self, messages: list, max_chars: int = 0) -> None:
        """Trim older tool results in the conversation to manage context size.

        Keeps the system message and last few exchanges intact but summarizes
        older tool results to prevent context window overflow.
        """
        max_chars = max_chars or getattr(self.config, "max_prompt_chars", 80000)

        # Estimate total size
        total = sum(len(str(m.get("content", ""))) for m in messages)
        if total <= max_chars:
            return

        # Summarize tool results from oldest to newest (skip system + first user)
        # Never touch the last 6 messages (current exchange)
        for i in range(2, max(2, len(messages) - 6)):
            msg = messages[i]
            content = msg.get("content") or ""
            if msg.get("role") in ("tool", "user") and len(content) > 500:
                # Summarize long tool results
                if content.startswith("[") and "lines]" in content[:80]:
                    # File read result — keep header + first/last lines
                    lines = content.splitlines()
                    if len(lines) > 20:
                        header = lines[0]
                        top = "\n".join(lines[1:8])
                        bottom = "\n".join(lines[-3:])
                        msg["content"] = (
                            f"{header}\n{top}\n"
                            f"  ... ({len(lines) - 11} lines summarized) ...\n"
                            f"{bottom}"
                        )
                else:
                    msg["content"] = content[:400] + "\n...(trimmed)"

            total = sum(len(str(m.get("content", ""))) for m in messages)
            if total <= max_chars:
                return

        # If still too big, drop oldest messages (except system + first user)
        while total > max_chars and len(messages) > 8:
            removed = messages.pop(2)
            total -= len(str(removed.get("content", "")))

    # Patterns that indicate the build TOOL is broken, not the code
    _BUILD_TOOL_BROKEN_PATTERNS = [
        "is not recognized as an internal or external command",  # Windows cmd
        "not recognized as a cmdlet",  # PowerShell
        "command not found",  # Unix
        "No such file or directory",  # Unix missing binary
        "Cannot find module",  # Node missing module
        "MODULE_NOT_FOUND",  # Node
        "ENOENT",  # Node file not found
    ]

    # Patterns that indicate the build command is actually running tests,
    # not doing a typecheck/lint. These are too slow and produce false failures.
    _TEST_RUNNER_PATTERNS = [
        "Test Suites:",   # Jest summary
        "Tests:",         # Jest summary
        "test-coverage",  # npm script name
        "PASS src/",      # Jest pass line
        "FAIL src/",      # Jest fail line
        "% Stmts",        # Coverage table header
    ]

    def _run_build_check(self) -> tuple[str, bool]:
        """Run the configured build command.

        Returns (error_output, is_tool_broken):
          - ("", False) on success
          - (output, False) on real build/type error (code is wrong)
          - (output, True) if the build tool itself can't run (PATH, missing binary, etc.)
        """
        if not self.config.build_command:
            return "", False
        try:
            import subprocess
            self.logger.info(
                "Running build check: %s", self.config.build_command
            )
            result = subprocess.run(
                self.config.build_command,
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                timeout=120,
                encoding="utf-8",
                errors="replace",
            )

            output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()

            # Detect if the "build" command is secretly running tests
            if any(p in output for p in self._TEST_RUNNER_PATTERNS):
                self.logger.warning(
                    "Build command '%s' appears to be running tests, not typechecking. "
                    "Disabling auto-verify. Use --build-command to set a fast typecheck.",
                    self.config.build_command,
                )
                return output, True  # Treat as tool-broken to disable

            if result.returncode == 0:
                return "", False

            if len(output) > 2000:
                output = output[:2000] + "\n...(build output truncated)"

            # Check if the failure is the tool itself, not the code
            is_tool_broken = any(
                pattern in output for pattern in self._BUILD_TOOL_BROKEN_PATTERNS
            )
            return output, is_tool_broken

        except subprocess.TimeoutExpired:
            self.logger.warning(
                "Build command timed out after 120s: %s",
                self.config.build_command,
            )
            return (
                f"Build command timed out after 120 seconds: "
                f"{self.config.build_command}"
            ), True
        except FileNotFoundError:
            return f"Build command not found: {self.config.build_command}", True
        except Exception as e:
            self.logger.warning(f"Build check failed to run: {e}")
            return "", False  # Don't block on unknown failures
    
    def _review_changes(self, context: TaskContext, skill: Skill) -> str:
        """Review the changes made."""
        try:
            diff = self.repo.git.diff()
            status = self.repo.git.status()
        except GitCommandError:
            diff = "Could not get diff"
            status = "Could not get status"
        
        # Also include content of new (untracked) files — git diff misses these
        untracked_section = ""
        try:
            untracked = self.repo.untracked_files
            # Filter out workspace/checkpoint files
            source_untracked = [
                f for f in untracked
                if not f.startswith(".coding-agent")
                and not f.startswith("cat")
                and not f.endswith(".txt;")
            ]
            if source_untracked:
                parts = []
                for fpath in source_untracked[:5]:  # Limit to 5 files
                    try:
                        full = self.repo_path / fpath
                        content = full.read_text(encoding="utf-8", errors="replace")
                        if len(content) > 2000:
                            content = content[:2000] + "\n...(file truncated)"
                        parts.append(f"--- NEW FILE: {fpath} ---\n{content}")
                    except Exception:
                        parts.append(f"--- NEW FILE: {fpath} (could not read) ---")
                untracked_section = "\n\nNew (untracked) files:\n" + "\n\n".join(parts)
        except Exception:
            pass

        max_diff = 4000 if not self.config.verbose else 12000
        combined_diff = diff + untracked_section
        diff_to_send = combined_diff if len(combined_diff) <= max_diff else (
            combined_diff[:max_diff] + f"\n...(diff truncated, {len(combined_diff) - max_diff} chars omitted)"
        )
        
        recent_log = context.execution_log[-3:]
        log_text = "\n".join(entry[:300] for entry in recent_log)

        # Include the agent's completion rationale so the reviewer
        # understands WHY certain changes were (or weren't) made.
        agent_summary = ""
        if context.done_message:
            agent_summary = f"""
Agent's completion summary:
{context.done_message}
"""

        # Include files the agent read — the reviewer should consider
        # information from these files even if they weren't modified.
        files_read_section = ""
        if context.files_read:
            files_read_section = (
                "\nFiles the agent read during execution (may contain relevant context "
                "even if not modified):\n"
                + "\n".join(f"  - {f}" for f in sorted(context.files_read))
                + "\n"
            )

        prompt = f"""Review the changes made for this task:

Task: {context.task_description}

Git Status:
{status}

Changes (diff):
{diff_to_send}
{agent_summary}{files_read_section}
Recent steps:
{log_text}

IMPORTANT: When evaluating whether a requirement is satisfied, consider ALL
information available — not just the diff. If the agent read a file and found
that a requirement is already implemented there (e.g., caching already exists
in a helper module), that counts as satisfied even if the diff doesn't show
changes to that file. The agent's completion summary above explains its reasoning.

{skill.review_prompt}
"""
        
        return self.llm.generate(prompt, skill.system_prompt)
    
    @staticmethod
    def _review_passed(review_result: str) -> bool:
        """Check whether the review status is PASS.

        Uses a precise regex so that incidental uses of the word "pass" in
        prose (e.g. "pass data to the handler") don't trigger a false positive.
        """
        if not review_result:
            return False
        # Match "STATUS: PASS" (with optional brackets/stars/whitespace)
        # but NOT "STATUS: PASS_WITH_SUGGESTIONS" or "STATUS: NEEDS_WORK"
        return bool(re.search(
            r'STATUS\s*:\s*\[?\s*\*{0,2}\s*PASS\s*\*{0,2}\s*\]?'
            r'(?:\s|$|[^A-Z_])',
            review_result,
            re.IGNORECASE | re.MULTILINE,
        ))

    def _has_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        return self.repo.is_dirty(untracked_files=True)
    
    def _ensure_git_exclude(self) -> None:
        """Ensure .coding-agent is excluded via .git/info/exclude (repo-local, never committed)."""
        exclude_path = self.repo_path / ".git" / "info" / "exclude"
        marker = ".coding-agent"

        lines: list[str] = []
        if exclude_path.exists():
            try:
                lines = exclude_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                pass

        # Already present — nothing to do
        for line in lines:
            stripped = line.strip()
            if stripped == marker or stripped == f"/{marker}" or stripped == f"{marker}/":
                return

        # Append the entry
        try:
            exclude_path.parent.mkdir(parents=True, exist_ok=True)
            with exclude_path.open("a", encoding="utf-8") as f:
                # Ensure we start on a new line
                if lines and lines[-1] != "":
                    f.write("\n")
                f.write(f"{marker}/\n")
            self.logger.debug("Added .coding-agent/ to .git/info/exclude")
        except Exception as e:
            self.logger.warning("Could not update .git/info/exclude: %s", e)

    def _commit_changes(self, task_id: str) -> None:
        """Commit the changes, excluding agent workspace files."""
        try:
            import shutil

            # 1. Ensure .coding-agent is git-excluded so it never gets committed
            self._ensure_git_exclude()

            # 2. If .coding-agent was previously tracked, untrack it
            try:
                self.repo.git.rm("-r", "--cached", ".coding-agent")
                self.logger.debug("Untracked .coding-agent from git index")
            except GitCommandError:
                pass  # Not tracked — nothing to do

            # 3. Remove stray files the agent may have left
            for stray in (
                "cat", "test-output.txt", "test-output.txt;",
                "test_output.txt", "jest_output.txt",
                "temp-write-script.js",
            ):
                stray_path = self.repo_path / stray
                if stray_path.exists():
                    try:
                        stray_path.unlink()
                    except Exception:
                        pass

            # 4. Stage and commit
            self.repo.git.add(A=True)
            task_name = re.sub(r'\.(txt|md)$', '', task_id)
            self.repo.git.commit(m=f"Agent: {task_name}")
            self.logger.info("Changes committed successfully")

            # 5. Ensure workspace dir still exists for subsequent tasks
            workspace_dir = self.repo_path / ".coding-agent"
            workspace_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to commit: {e}")