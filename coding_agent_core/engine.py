from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from .config import AgentConfig, SystemContext, TaskContext
from .deps import GitCommandError, VECTOR_MEMORY_AVAILABLE, VectorMemoryManager
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
                # Prefer fast check-only commands over full builds
                preferred = ["typecheck", "check", "lint", "tsc", "build:check"]
                for name in preferred:
                    if name in scripts:
                        candidates.append((f"{pkg_manager} {name}", f'package.json "{name}"'))
                if "build" in scripts:
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
            self.logger.info("No --build-command set. Detected candidates:")
            for cmd, source in candidates:
                self.logger.info(f"  {cmd}  (from {source})")
            best = candidates[0][0]
            self.logger.info(f'Pass --build-command="{best}" to enable auto-verify after edits')

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
        
        # Execution loop
        for iteration in range(1, self.config.max_iterations + 1):
            context.iteration = iteration
            self.logger.info(f"Iteration {iteration}/{self.config.max_iterations}")
            
            # Plan
            if not context.plan or context.review_feedback:
                context.plan = self._create_plan(context, skill)
                self.logger.info(f"Plan created:\n{context.plan}")
            
            # Execute
            success = self._execute_plan(context, skill)
            if not success:
                self.logger.warning(f"Execution failed on iteration {iteration}")
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
                break
            else:
                context.review_feedback = review_result
                if self.config.verbose:
                    self.logger.info(f"Review feedback:\n{review_result}")
                else:
                    self.logger.info(f"Review feedback: {review_result[:200]}...")
                if iteration >= self.config.max_iterations:
                    self.logger.warning("Max iterations reached without PASS")
                    break
        
        # Check for changes
        if not self._has_changes():
            self.logger.warning("No changes were made")
            return False
        
        # Get list of modified files before committing
        try:
            modified = self.repo.git.diff("--name-only").split('\n')
            self.modified_files = [f.strip() for f in modified if f.strip()]
            new_files = self.repo.untracked_files
            self.modified_files.extend(new_files)
        except GitCommandError:
            self.modified_files = []
        
        # Commit
        if self.config.auto_commit:
            self._commit_changes(task_id)
        
        # Update memory with modified files
        if self.modified_files and self.memory_manager:
            self.logger.info(f"Updating memory for {len(self.modified_files)} modified files")
            try:
                self.memory_manager.update_for_task(
                    self.modified_files,
                    task_description,
                    self.current_branch,
                    skill.name
                )
            except Exception as e:
                self.logger.warning(f"Failed to update memory: {e}")
        
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

        prompt = f"""{context.system_info}

## Project File Tree (REAL — use only these paths)
{file_tree}

{context_info}{skill.planning_prompt.format(task_description=context.task_description)}

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
- For npm/npx scripts, prefer: npx jest, npm test, npm run <script>
- Do NOT use shell pipes like | head, | tail, | grep — they are unreliable on Windows.
- Do NOT use cd /c/... syntax — that is Git Bash Unix-style and won't work.
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

            # --- Auto-verify after mutations ---
            if is_mutation and "Error" not in result and build_verify_enabled:
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
                        result += (
                            f"\n\n⚠ BUILD FAILED after this change — file has been auto-reverted.\n"
                            f"Build output:\n{verify_output}\n"
                            f"Revert: {revert_result}\n"
                            f"Fix the issue and try again."
                        )
                        write_count -= 1
                        self.logger.warning("Auto-reverted %s after build failure", written_path)

            if self.config.verbose:
                self.logger.info(f"Result:\n{result}")
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
            result = subprocess.run(
                self.config.build_command,
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return "", False

            output = (result.stdout + "\n" + result.stderr).strip()
            if len(output) > 2000:
                output = output[:2000] + "\n...(build output truncated)"

            # Check if the failure is the tool itself, not the code
            is_tool_broken = any(
                pattern in output for pattern in self._BUILD_TOOL_BROKEN_PATTERNS
            )
            return output, is_tool_broken

        except subprocess.TimeoutExpired:
            return "Build command timed out after 120 seconds", False
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
    
    def _commit_changes(self, task_id: str) -> None:
        """Commit the changes."""
        try:
            self.repo.git.add(A=True)
            self.repo.git.commit(m=f"Agent: {task_id.replace('.txt', '')}")
            self.logger.info("Changes committed successfully")
        except Exception as e:
            self.logger.error(f"Failed to commit: {e}")