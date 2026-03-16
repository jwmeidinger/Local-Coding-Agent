from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .config import AgentConfig
from .deps import GIT_AVAILABLE, Repo, VECTOR_MEMORY_AVAILABLE
from .engine import ExecutionEngine


class CodingAgent:
    """Main Coding Agent application with multi-repo support."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
        # Ensure required directories exist before logging is initialized.
        # Logging writes to workspace_dir/agent.log on startup.
        config.tasks_dir.mkdir(parents=True, exist_ok=True)
        config.skills_dir.mkdir(parents=True, exist_ok=True)
        config.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # Initialize git repos
        if not GIT_AVAILABLE:
            raise RuntimeError("GitPython is required. Install: pip install GitPython")
        
        self.repos: Dict[Path, Repo] = {}
        self.engines: Dict[Path, ExecutionEngine] = {}
        
        # Load all configured repos
        for repo_path in config.repo_paths:
            if self._should_ignore_repo(repo_path):
                self.logger.info(f"Ignoring repo: {repo_path}")
                continue
                
            try:
                repo = Repo(repo_path)
                engine = ExecutionEngine(config, repo, repo_path)
                # Only register after BOTH repo and engine succeed
                self.repos[repo_path] = repo
                self.engines[repo_path] = engine
                self.logger.info(f"Loaded repo: {repo_path}")
            except Exception as e:
                self.logger.warning(f"Could not load repo {repo_path}: {e}")
        
        if not self.repos:
            raise RuntimeError("No valid git repositories found!")
        
        # Directories are created at startup before logging initialization.
    
    def close(self):
        """Cleanup resources and database connections."""
        self.logger.info("Cleaning up resources...")
        for repo_path, engine in self.engines.items():
            if engine.memory_manager:
                try:
                    engine.memory_manager.close()
                    self.logger.debug(f"Closed database connection for {repo_path}")
                except Exception as e:
                    self.logger.warning(f"Error closing connection for {repo_path}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
    
    def _should_ignore_repo(self, repo_path: Path) -> bool:
        """Check if repo should be ignored based on patterns."""
        repo_name = repo_path.name
        for pattern in self.config.ignore_patterns:
            if re.match(pattern, repo_name):
                return True
        return False
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("coding-agent")
        logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)
        
        if not logger.handlers:
            # Force UTF-8 on stdout for Windows compatibility
            if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
                try:
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # File handler (mode='w' to start fresh each run)
            file_handler = logging.FileHandler(
                self.config.workspace_dir / "agent.log",
                mode="w",
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def discover_tasks(self) -> List[tuple[Path, Path]]:
        """Find all pending tasks across all repos.
        
        Returns list of (task_path, repo_path) tuples.
        """
        tasks = []
        
        # Check central tasks directory first
        if self.config.tasks_dir.exists():
            for task_file in self.config.tasks_dir.glob("*.txt"):
                # Determine which repo this task belongs to
                repo_path = self._determine_task_repo(task_file)
                if repo_path:
                    tasks.append((task_file, repo_path))
        
        # Also check each repo's tasks directory
        for repo_path in self.repos.keys():
            repo_tasks_dir = repo_path / "tasks"
            if repo_tasks_dir.exists():
                for task_file in repo_tasks_dir.glob("*.txt"):
                    tasks.append((task_file, repo_path))
        
        # Sort by modification time
        tasks.sort(key=lambda x: x[0].stat().st_mtime)
        return tasks
    
    def _determine_task_repo(self, task_path: Path) -> Optional[Path]:
        """Determine which repo a task belongs to.
        
        Checks for explicit repo specification in task or infers from content.
        """
        task_content = task_path.read_text(encoding="utf-8").strip()
        task_lower = task_content.lower()
        
        # Check for explicit repo specification: "REPO: reponame" at start
        repo_match = re.match(r'^\s*REPO:\s*(\S+)', task_content, re.IGNORECASE)
        if repo_match:
            specified_repo = repo_match.group(1)
            for repo_path in self.repos.keys():
                if repo_path.name == specified_repo:
                    return repo_path
        
        # Try to infer from task content by searching in vector memory (each engine is per-repo)
        if VECTOR_MEMORY_AVAILABLE:
            try:
                best_score = -1.0
                best_repo = None
                for repo_path in self.repos.keys():
                    engine = self.engines[repo_path]
                    if engine.memory_manager:
                        results = engine.memory_manager.search_codebase(task_content, limit=1)
                        if results and results[0]["combined_score"] > best_score:
                            best_score = results[0]["combined_score"]
                            best_repo = repo_path
                if best_repo is not None:
                    return best_repo
            except Exception:
                pass
        
        # Default to first repo if can't determine
        return list(self.repos.keys())[0] if self.repos else None
    
    def run(self) -> None:
        """Run the agent across all repos."""
        self.logger.info("=" * 60)
        self.logger.info("Coding Agent Starting")
        self.logger.info(f"Active repos: {len(self.repos)}")
        for repo_path in self.repos.keys():
            self.logger.info(f"  - {repo_path}")
        self.logger.info("=" * 60)
        
        tasks = self.discover_tasks()
        
        if not tasks:
            self.logger.info("No tasks found")
            return
        
        self.logger.info(f"Found {len(tasks)} tasks to process")
        
        for task_path, repo_path in tasks:
            task_id = task_path.name
            task_description = task_path.read_text(encoding="utf-8").strip()

            if not task_description:
                self.logger.warning(f"Skipping empty task file: {task_id}")
                continue

            self.logger.info(f"\nProcessing: {task_id} (repo: {repo_path.name})")
            
            try:
                engine = self.engines[repo_path]
                success = engine.execute_task(task_description, task_id, repo_path)
                
                if success:
                    self.logger.info(f"[OK] Task completed: {task_id}")
                    self._archive_task(task_path)
                else:
                    self.logger.error(f"[FAIL] Task failed: {task_id}")
                    
            except Exception as e:
                self.logger.error(f"[FAIL] Error processing {task_id}: {e}")
                if self.config.verbose:
                    import traceback
                    self.logger.error(traceback.format_exc())
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Coding Agent Complete")
        self.logger.info("=" * 60)
    
    def _archive_task(self, task_path: Path) -> None:
        """Move completed task to archive."""
        archive_dir = self.config.workspace_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        dest = archive_dir / task_path.name
        task_path.rename(dest)
        self.logger.info(f"Archived to: {dest}")
