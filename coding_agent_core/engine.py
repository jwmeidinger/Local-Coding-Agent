from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from .config import AgentConfig, SystemContext, TaskContext
from .deps import GitCommandError, VECTOR_MEMORY_AVAILABLE, VectorMemoryManager
from .llm import LLMManager
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
                self.memory_manager = VectorMemoryManager(repo_path)
                self.logger.info(f"Vector memory system initialized for {repo_path.name}")
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
                    "file_write. You MUST write code this iteration. Do NOT just read "
                    "files again. Use the information you already gathered to make the "
                    "changes immediately."
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
        """Create an execution plan."""
        # Build prompt with codebase context using vector search
        context_info = ""
        if self.memory_manager:
            try:
                # Search for relevant code files using vector + text hybrid search
                results = self.memory_manager.search_codebase(
                    context.task_description,
                    limit=10
                )
                if results:
                    context_info = "Relevant Code Files Found:\n"
                    for r in results[:5]:  # Top 5 most relevant
                        context_info += f"\n  📄 {r['file_path']} (score: {r['combined_score']:.2f})\n"
                        context_info += f"     {r.get('summary') or ''}\n"
                        kf = r.get('key_functions') or []
                        if kf:
                            context_info += f"     Functions: {', '.join(kf[:5])}\n"
                    context_info += "\n"
            except Exception as e:
                self.logger.warning(f"Vector search failed: {e}")
        
        prompt = f"""{context.system_info}

{context_info}{skill.planning_prompt.format(task_description=context.task_description)}

IMPORTANT: Keep your plan SHORT and ACTIONABLE. List only:
1. Files to read (to understand existing code)
2. Files to create or modify (with brief description of changes)
3. Commands to run (e.g. install dependencies, run tests)

Do NOT write code in the plan. Do NOT write essays. Maximum 30 lines."""
        if context.review_feedback:
            prompt += f"\n\nPrevious feedback to address:\n{context.review_feedback}"
        
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
        """Execute the plan step by step with context management.

        Context strategy:
        - The MOST RECENT tool result is kept in full so the LLM has complete
          information about the action it just took.
        - OLDER results are summarised (file structure + first/last lines) to
          stay within the model's context budget without losing key information.
        - Duplicate full-file reads are blocked; the agent is told to use
          start_line/end_line to revisit specific sections.
        """
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
            memory_context = f"""Most Relevant Files:
{chr(10).join([f"  - {f}" for f in relevant_files[:5]])}

"""

        plan_text = context.plan or ""
        if len(plan_text) > 3000:
            plan_text = plan_text[:3000] + "\n...(plan truncated)"

        tool_list_simple = self.tools.list_tools_simple()

        base_prompt = f"""{context.system_info}

{memory_context}You are a coding agent that executes tasks by calling tools. You MUST call tools to make changes.
DO NOT just describe what to do. DO NOT write essays. CALL THE TOOLS.

You are working in repository: {context.repo_path}
All file paths are relative to this directory.

Task: {context.task_description}

Plan (summary):
{plan_text}

## Available Tools

{tool_list_simple}

## Tool Calling Format

Use this format - ONE tool per response:

    file_read(path="src/main.py")
    file_read(path="src/main.py", start_line="50", end_line="100")
    file_write(path="src/auth.js", content="const token = process.env.TOKEN;")
    bash(command="npm install simple-oauth2 open")
    list_files(path="src")
    done(message="Task completed")

## CRITICAL RULES
1. BEFORE reading any file, use list_files to see what files exist
2. If a file_read fails with "not found", use list_files first
3. Call ONE tool per response only
4. For large files, use start_line/end_line to read specific sections
5. Do NOT re-read files you have already read — use start_line/end_line to revisit sections
6. When all work is done, call done(message="...")

Thought: I need to see what files exist first.
Action: list_files(path=".")
"""

        max_steps = 20
        summary_chars = getattr(self.config, "max_tool_result_chars", 1500)
        max_consecutive_errors = getattr(self.config, "max_consecutive_errors", 2)

        no_tool_count = 0
        error_count = 0
        write_count = 0
        files_already_read: set[str] = set()

        # Each entry: {"tool_call": str, "result_full": str, "result_summary": str}
        step_records: list[dict] = []
        MAX_HISTORY_WINDOW = 10

        # Step budget thresholds for escalating write nudges
        NUDGE_SOFT = int(max_steps * 0.5)    # 50% — gentle reminder
        NUDGE_HARD = int(max_steps * 0.75)   # 75% — strong warning
        NUDGE_FINAL = int(max_steps * 0.9)   # 90% — last chance

        for step in range(max_steps):
            # --- Build prompt: base + summarised older steps + full recent step ---
            window = step_records[-MAX_HISTORY_WINDOW:]
            history_parts = []

            if len(step_records) > MAX_HISTORY_WINDOW:
                history_parts.append(
                    f"({len(step_records) - MAX_HISTORY_WINDOW} earlier steps omitted)\n"
                )

            for idx, rec in enumerate(window):
                is_latest = (idx == len(window) - 1)
                result_text = rec["result_full"] if is_latest else rec["result_summary"]
                history_parts.append(
                    f"\nExecuted: {rec['tool_call']}\nResult:\n{result_text}\n"
                )

            if history_parts:
                history_parts.append("Call the next tool (or call done() if finished):")

            # Escalating nudges when the agent is only reading
            if write_count == 0 and step >= NUDGE_SOFT:
                remaining = max_steps - step
                if step >= NUDGE_FINAL:
                    history_parts.append(
                        f"\n** FINAL WARNING: You have {remaining} steps left and have "
                        f"NOT written any code yet. Call file_write NOW with the changes "
                        f"or the task will fail. Do NOT read any more files. **"
                    )
                elif step >= NUDGE_HARD:
                    history_parts.append(
                        f"\n** WARNING: You have used {step}/{max_steps} steps reading "
                        f"files but have NOT called file_write yet. You MUST start "
                        f"writing code NOW. You have enough information. **"
                    )
                else:
                    history_parts.append(
                        f"\nNote: You have used {step} of {max_steps} steps. "
                        f"Start making changes with file_write soon."
                    )

            prompt = base_prompt + "\n" + "\n".join(history_parts)

            response = self.llm.generate(prompt, skill.system_prompt)

            # Handle LLM failure (empty response = timeout or error)
            if not response or not response.strip():
                error_count += 1
                self.logger.warning(
                    "LLM returned empty response (%d/%d consecutive errors)",
                    error_count, max_consecutive_errors,
                )
                if error_count >= max_consecutive_errors:
                    self.logger.error(
                        "Aborting execution: %d consecutive LLM failures (likely timeout). "
                        "Try reducing task complexity or increasing the LLM timeout.",
                        error_count,
                    )
                    return False
                # Trim history aggressively on error to shrink context for retry
                step_records = step_records[-3:]
                continue

            error_count = 0

            if self.config.verbose:
                context.execution_log.append(f"Step {step + 1}:\n{response}")
            else:
                context.execution_log.append(f"Step {step + 1}: {response[:200]}...")

            tool_calls = self.llm.extract_tool_calls(response)

            if not tool_calls:
                if any(word in response.lower() for word in ["done", "complete", "finished", "all changes"]):
                    return True

                no_tool_count += 1
                if no_tool_count >= 3:
                    self.logger.warning("Model not producing tool calls after %d attempts", no_tool_count)
                    return True

                step_records.append({
                    "tool_call": "(no tool called)",
                    "result_full": "You MUST call a tool now. Do not explain. Output ONLY the tool call.\nExample: list_files(path=\".\")",
                    "result_summary": "You MUST call a tool now.",
                })
            else:
                no_tool_count = 0
                tool_call = tool_calls[0]
                self.logger.info(f"Executing: {tool_call}")

                if tool_call.startswith('done('):
                    self.logger.info("Task completed - done tool called")
                    return True

                # Deduplicate full-file reads (line-range re-reads are allowed)
                if tool_call.startswith('file_read('):
                    path_match = re.search(r'path\s*=\s*["\']([^"\']+)["\']', tool_call)
                    has_range = re.search(r'(start_line|end_line)\s*=', tool_call)
                    if path_match and not has_range:
                        read_path = path_match.group(1)
                        if read_path in files_already_read:
                            msg = (
                                f"You already read {read_path}. The content is in your earlier context. "
                                f"Use start_line/end_line to revisit specific sections."
                            )
                            step_records.append({
                                "tool_call": tool_call,
                                "result_full": msg,
                                "result_summary": msg,
                            })
                            self.logger.info(f"Skipped duplicate read of {read_path}")
                            continue
                        files_already_read.add(read_path)

                if tool_call.startswith('file_write('):
                    write_count += 1

                result = self.tools.execute(tool_call)

                if self.config.verbose:
                    self.logger.info(f"Result:\n{result}")
                else:
                    self.logger.info(f"Result: {result[:200]}...")

                if "not found" in result.lower():
                    result += "\n\nIMPORTANT: Use list_files to find the correct path before trying file_read again."

                step_records.append({
                    "tool_call": tool_call,
                    "result_full": result,
                    "result_summary": self._summarize_result(tool_call, result, summary_chars),
                })

        if write_count == 0:
            self.logger.warning(
                "Execution used all %d steps without calling file_write "
                "(read-loop detected: %d files read). The model investigated "
                "but never wrote code.",
                max_steps, len(files_already_read),
            )

        return True
    
    def _review_changes(self, context: TaskContext, skill: Skill) -> str:
        """Review the changes made."""
        try:
            diff = self.repo.git.diff()
            status = self.repo.git.status()
        except GitCommandError:
            diff = "Could not get diff"
            status = "Could not get status"
        
        max_diff = 4000 if not self.config.verbose else 12000
        diff_to_send = diff if len(diff) <= max_diff else (
            diff[:max_diff] + f"\n...(diff truncated, {len(diff) - max_diff} chars omitted)"
        )
        
        recent_log = context.execution_log[-3:]
        log_text = "\n".join(entry[:300] for entry in recent_log)
        
        prompt = f"""Review the changes made for this task:

Task: {context.task_description}

Git Status:
{status}

Changes (diff):
{diff_to_send}

Recent steps:
{log_text}

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
