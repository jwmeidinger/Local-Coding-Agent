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
                
                # If no files indexed, offer to index
                if "No files indexed" in summary:
                    self.logger.info("Codebase not indexed yet. Run with --index to index first.")
            except Exception as e:
                self.logger.warning(f"Could not get codebase summary: {e}")
        
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
            
            # Review
            review_result = self._review_changes(context, skill)
            
            if "PASS" in review_result.upper():
                self.logger.info("Task passed review")
                break
            else:
                context.review_feedback = review_result
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
    
    def _execute_plan(self, context: TaskContext, skill: Skill) -> bool:
        """Execute the plan step by step."""
        # Get relevant code context using vector search
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
        
        # Truncate plan if it's too long (model returned a huge plan with code)
        plan_text = context.plan or ""
        if len(plan_text) > 6000:
            plan_text = plan_text[:6000] + "\n...(plan truncated)"

        prompt = f"""{context.system_info}

{memory_context}You are a coding agent that executes tasks by calling tools. You MUST call tools to make changes.
DO NOT just describe what to do. DO NOT write essays. CALL THE TOOLS.

You are working in repository: {context.repo_path}
All file paths are relative to this directory.

Task: {context.task_description}

Plan (summary):
{plan_text}

## Available Tools

{self.tools.list_tools()}

## How to Call Tools

You MUST output tool calls in EXACTLY this format (one per response):

    file_read(path="src/main.py")

    file_write(path="src/auth.js", content="const token = process.env.TOKEN;")

    bash(command="npm install simple-oauth2 open")

    list_files(path="src")

## Rules
1. Call ONE tool per response
2. Start by reading existing files with file_read or list_files
3. Write files with file_write
4. Run commands with bash
5. When done, say DONE

## IMPORTANT
- Do NOT write long explanations. Just call the tool.
- Do NOT put tool calls inside markdown code blocks.
- Output the tool call on its own line.

Begin. First, list the files in the repository:
list_files(path=".")
"""
        
        max_steps = 20
        no_tool_count = 0
        for step in range(max_steps):
            # Guard against unbounded prompt growth
            if len(prompt) > 40000:
                prompt = prompt[:8000] + "\n\n...(earlier steps omitted)...\n\n" + prompt[-8000:]

            response = self.llm.generate(prompt, skill.system_prompt)
            context.execution_log.append(f"Step {step + 1}: {response[:200]}...")
            
            # Extract and execute tool calls
            tool_calls = self.llm.extract_tool_calls(response)
            
            if not tool_calls:
                # No tool calls - check if task is complete
                if any(word in response.lower() for word in ["done", "complete", "finished", "all changes"]):
                    return True

                no_tool_count += 1
                if no_tool_count >= 3:
                    self.logger.warning("Model not producing tool calls after %d attempts", no_tool_count)
                    return True

                # Nudge the model to call a tool
                prompt += f"\n\nYou MUST call a tool now. Do not explain. Output ONLY the tool call. Example:\nlist_files(path=\".\")\nCall a tool now:"
            else:
                no_tool_count = 0
                tool_call = tool_calls[0]
                self.logger.info(f"Executing: {tool_call}")
                result = self.tools.execute(tool_call)
                self.logger.info(f"Result: {result[:200]}...")
                prompt += f"\n\nExecuted: {tool_call}\nResult:\n{result[:2000]}\n\nCall the next tool (or say DONE if finished):"
        
        return True
    
    def _review_changes(self, context: TaskContext, skill: Skill) -> str:
        """Review the changes made."""
        # Get git diff
        try:
            diff = self.repo.git.diff()
            status = self.repo.git.status()
        except GitCommandError:
            diff = "Could not get diff"
            status = "Could not get status"
        
        prompt = f"""Review the changes made for this task:

Task: {context.task_description}

Git Status:
{status}

Changes (diff):
{diff[:4000]}

Execution log:
{"\n".join(context.execution_log[-5:])}

{skill.review_prompt}
"""
        
        return self.llm.generate(prompt, skill.system_prompt)
    
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
