from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import AgentConfig


class CheckpointManager:
    """Lightweight file checkpoint manager inspired by Hermes checkpointing."""

    def __init__(self, repo_root: Path, workspace_dir: Path, max_snapshots: int = 100):
        self.repo_root = repo_root.resolve()
        if workspace_dir.is_absolute():
            base_dir = workspace_dir
        else:
            base_dir = self.repo_root / workspace_dir
        self.checkpoints_dir = (base_dir / "checkpoints").resolve()
        self.max_snapshots = max_snapshots

    def snapshot_file(self, file_path: Path) -> Optional[Path]:
        """Create a timestamped snapshot of a file before mutation."""
        if not file_path.exists() or not file_path.is_file():
            return None

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        try:
            rel_path = file_path.resolve().relative_to(self.repo_root)
        except ValueError:
            # Fallback for paths outside repo root
            rel_path = Path(file_path.name)

        checkpoint_path = self.checkpoints_dir / timestamp / rel_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, checkpoint_path)
        self._prune()
        return checkpoint_path

    def _prune(self) -> None:
        """Keep checkpoint count bounded by deleting oldest snapshots."""
        if not self.checkpoints_dir.exists():
            return

        snapshots = sorted(
            [p for p in self.checkpoints_dir.iterdir() if p.is_dir()],
            key=lambda p: p.name
        )
        excess = len(snapshots) - self.max_snapshots
        if excess <= 0:
            return

        for old_snapshot in snapshots[:excess]:
            shutil.rmtree(old_snapshot, ignore_errors=True)


class FileReadTool:
    """Read file contents with line numbers and optional range selection."""
    name = "file_read"
    description = "Read contents of a file (supports optional start_line/end_line to read specific sections)"

    # Only truncate truly enormous files (generated code, minified bundles, etc.)
    HARD_MAX_LINES = 800

    schema = {
        "name": "file_read",
        "description": "Read the contents of a file. For large files, use start_line/end_line to read specific sections.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to read (1-based). Omit to start from beginning."
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to read (1-based, inclusive). Omit to read to end."
                }
            },
            "required": ["path"]
        }
    }
    
    def __init__(self, cwd: Path = None):
        self.cwd = cwd or Path(".")
    
    def execute(self, path: str, start_line: str = None, end_line: str = None) -> str:
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.cwd / file_path
            if not file_path.exists():
                return f"Error: File '{path}' not found. Use list_files to explore the directory structure first. The cwd is: {self.cwd}"
            
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            total_lines = len(lines)
            
            sl = int(start_line) if start_line else None
            el = int(end_line) if end_line else None
            
            # Line-range request — always return exactly what was asked for
            if sl or el:
                s = max((sl or 1) - 1, 0)
                e = min(el or total_lines, total_lines)
                selected = lines[s:e]
                numbered = [f"{s + i + 1}| {l}" for i, l in enumerate(selected)]
                return f"[{path} lines {s+1}-{e} of {total_lines}]\n" + "\n".join(numbered)
            
            # Full-file read — return complete content for normal-sized files
            if total_lines <= self.HARD_MAX_LINES:
                numbered = [f"{i + 1}| {l}" for i, l in enumerate(lines)]
                return f"[{path} — {total_lines} lines]\n" + "\n".join(numbered)
            
            # Truly huge files: show generous head + tail with clear guidance
            head_n = 300
            tail_n = 100
            head = lines[:head_n]
            tail = lines[-tail_n:]
            omitted = total_lines - head_n - tail_n
            
            head_numbered = [f"{i + 1}| {l}" for i, l in enumerate(head)]
            tail_start = total_lines - tail_n
            tail_numbered = [f"{tail_start + i + 1}| {l}" for i, l in enumerate(tail)]
            
            return (
                f"[{path} — {total_lines} lines, showing lines 1-{head_n} and {tail_start+1}-{total_lines}]\n"
                + "\n".join(head_numbered)
                + f"\n\n... ({omitted} lines omitted — use start_line/end_line to read lines {head_n+1}-{tail_start}) ...\n\n"
                + "\n".join(tail_numbered)
            )
        except Exception as e:
            return f"Error reading file: {e}"


class FileWriteTool:
    """Write content to a file."""
    name = "file_write"
    description = "Write content to a file"
    
    # JSON Schema for tool definition
    schema = {
        "name": "file_write",
        "description": "Write content to a file, creating it if it doesn't exist or overwriting if it does",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    }
    
    def __init__(self, cwd: Path = None, checkpoint_manager: Optional[CheckpointManager] = None):
        self.cwd = cwd or Path(".")
        self.checkpoint_manager = checkpoint_manager
    
    def execute(self, path: str, content: str) -> str:
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.cwd / file_path

            checkpoint_path = None
            if self.checkpoint_manager and file_path.exists():
                checkpoint_path = self.checkpoint_manager.snapshot_file(file_path)

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            if checkpoint_path:
                return f"Successfully wrote to {path} (checkpoint: {checkpoint_path})"
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"


class BashTool:
    """Execute bash commands."""
    name = "bash"
    description = "Execute bash commands in the repository"
    
    # JSON Schema for tool definition
    schema = {
        "name": "bash",
        "description": "Execute a bash command in the repository directory",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                }
            },
            "required": ["command"]
        }
    }
    
    DANGEROUS_PATTERNS = [
        r'^\s*sudo\s+apt\s+install',
        r'^\s*sudo\s+apt\s+upgrade',
        r'^\s*sudo\s+apt\s+dist-upgrade',
        r'^\s*sudo\s+yum\s+install',
        r'^\s*sudo\s+yum\s+update',
        r'^\s*sudo\s+dnf\s+install',
        r'^\s*sudo\s+dnf\s+update',
        r'^\s*sudo\s+pacman\s+-S',
        r'^\s*sudo\s+apk\s+add',
        r'^\s*brew\s+install\s+python',
        r'^\s*brew\s+upgrade\s+python',
        r'^\s*pip\s+install\s+--upgrade\s+python',
        r'^\s*python\s+-m\s+pip\s+install\s+--upgrade',
        r'^\s*nvm\s+install',
        r'^\s*source\s+nvm',
        r'^\s*curl.*\|.*bash',
        r'^\s*wget.*\|.*bash',
        r'^\s*rm\s+-rf\s+/',
        r'^\s*rm\s+-rf\s+/usr',
        r'^\s*rm\s+-rf\s+/bin',
        r'^\s*dd\s+if=',
        r'^\s*>:',
        r'^\s*>\s*/dev/',
    ]
    
    DANGEROUS_KEYWORDS = [
        'chmod 777',
        'chown',
        'systemctl restart',
        'systemctl stop',
        'service restart',
        'kill -9',
        'killall',
        'reboot',
        'shutdown',
        'init 0',
        'init 6',
    ]
    
    def __init__(self, cwd: Path):
        self.cwd = cwd
    
    def execute(self, command: str) -> str:
        # Second layer of defense: check command for dangerous patterns
        is_safe, reason = self._check_dangerous_command(command)
        if not is_safe:
            return f"Error: Command blocked by safety guard - {reason}\n\nThis command attempts to modify system components. The agent is not allowed to:\n- Install or upgrade system packages\n- Modify system-wide Python/Java/Node\n- Execute potentially destructive commands\n\nIf you need to install project dependencies, use:\n- pip install -r requirements.txt\n- npm install\n- Just regular commands without sudo/brew upgrade"
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=60
            )
            output = result.stdout if result.stdout else "(no output)"
            if result.stderr:
                output += f"\nStderr: {result.stderr}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            return output
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 60 seconds"
        except Exception as e:
            return f"Error executing command: {e}"
    
    def _check_dangerous_command(self, command: str) -> tuple[bool, str]:
        """Check if command is dangerous."""
        import re
        
        # Check patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE | re.MULTILINE):
                return False, f"matches dangerous pattern: {pattern}"
        
        # Check keywords
        for keyword in self.DANGEROUS_KEYWORDS:
            if keyword.lower() in command.lower():
                return False, f"contains dangerous keyword: {keyword}"
        
        return True, ""


class ListFilesTool:
    """List files in a directory."""
    name = "list_files"
    description = "List files in a directory"
    
    # JSON Schema for tool definition
    schema = {
        "name": "list_files",
        "description": "List files in a directory matching a glob pattern",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)"
                },
                "pattern": {
                    "type": "string", 
                    "description": "Glob pattern to match files (default: * for all files)"
                }
            }
        }
    }
    
    def __init__(self, cwd: Path = None):
        self.cwd = cwd or Path(".")
    
    def execute(self, path: str = ".", pattern: str = "*") -> str:
        try:
            dir_path = Path(path)
            if not dir_path.is_absolute():
                dir_path = self.cwd / dir_path
            if not dir_path.exists():
                return f"Error: Directory '{path}' not found. Available directories in {self.cwd}:\n" + "\n".join([str(f.relative_to(self.cwd)) for f in self.cwd.iterdir()])[:500]
            files = list(dir_path.glob(pattern))
            return "\n".join([str(f.relative_to(dir_path)) for f in files])
        except Exception as e:
            return f"Error listing files: {e}"


class GrepTool:
    """Search for text patterns in files."""
    name = "grep"
    description = "Search for text patterns in files using regex"
    
    # JSON Schema for tool definition
    schema = {
        "name": "grep",
        "description": "Search for text patterns in files using regular expressions. Similar to the ripgrep (rg) command.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path to search in (default: current directory)"
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern for files to include (e.g., '*.js', '*.py')"
                }
            },
            "required": ["pattern"]
        }
    }
    
    def __init__(self, cwd: Path = None):
        self.cwd = cwd or Path(".")
    
    def execute(self, pattern: str, path: str = ".", include: str = None) -> str:
        import re
        try:
            search_path = Path(path)
            if not search_path.is_absolute():
                search_path = self.cwd / search_path
            
            if not search_path.exists():
                return f"Error: Path '{path}' not found"
            
            results = []
            
            # Determine which files to search
            if search_path.is_file():
                files_to_search = [search_path]
            else:
                files_to_search = []
                if include:
                    files_to_search = list(search_path.glob(include))
                else:
                    # Search all text files
                    for ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.cpp', '.c', '.h', '.rb', '.php', '.json', '.yaml', '.yml', '.md', '.txt']:
                        files_to_search.extend(search_path.rglob(f'*{ext}'))
            
            # Compile regex
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return f"Error: Invalid regex pattern: {e}"
            
            # Search files
            for file_path in files_to_search[:100]:  # Limit to 100 files
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    for line_num, line in enumerate(content.split('\n'), 1):
                        if regex.search(line):
                            results.append(f"{file_path.relative_to(self.cwd)}:{line_num}: {line.rstrip()}")
                except Exception:
                    continue
            
            if not results:
                return f"No matches found for pattern: {pattern}"
            
            return "\n".join(results[:50])  # Limit results
        
        except Exception as e:
            return f"Error searching: {e}"


class GitStatusTool:
    """Check git status."""
    name = "git_status"
    description = "Check current git status"
    
    # JSON Schema for tool definition
    schema = {
        "name": "git_status",
        "description": "Check the current git status of the repository",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
    
    def __init__(self, repo):
        self.repo = repo
    
    def execute(self) -> str:
        try:
            return self.repo.git.status()
        except Exception as e:
            return f"Error checking git status: {e}"


class DoneTool:
    """Signal that the task is complete."""
    name = "done"
    description = "Signal that all work is done. Call this when you have completed the task."
    
    # JSON Schema for tool definition
    schema = {
        "name": "done",
        "description": "Signal that the task is complete. Call this when all work has been finished.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Optional message describing what was completed"
                }
            }
        }
    }
    
    def execute(self, message: str = "") -> str:
        return f"DONE: {message if message else 'Task completed successfully'}"


class SearchGuard:
    """Validates web search queries to ensure they are general questions, not code."""
    
    CODE_PATTERNS = [
        r'def\s+\w+',           # function definitions
        r'class\s+\w+',         # class definitions
        r'import\s+\w+',       # imports
        r'from\s+\w+\s+import', # from imports
        r'function\s+\w+\s*\(', # JS functions
        r'const\s+\w+\s*=',     # JS const
        r'let\s+\w+\s*=',       # JS let
        r'var\s+\w+\s*=',       # JS var
        r'=\s*\{',              # object literals
        r'\[.*\]\s*=',          # array assignments
        r'<\w+>',               # HTML/JSX tags
        r'#include',            # C/C++ includes
        r'package\s+\w+',       # Go/Java packages
        r'pub\s+fn',            # Rust functions
        r'func\s+\w+',          # Go functions
        r'@\w+',                # decorators
        r'\$\w+',               # PHP/jQuery
        r'select\s+.*from',    # SQL SELECT
        r'insert\s+into',      # SQL INSERT
        r'create\s+table',     # SQL CREATE
        r'update\s+\w+\s+set', # SQL UPDATE
        r'delete\s+from',      # SQL DELETE
        r'where\s+\w+',         # SQL WHERE
    ]
    
    CODE_EXTENSIONS = [
        r'\.py\b', r'\.js\b', r'\.ts\b', r'\.jsx\b', r'\.tsx\b',
        r'\.java\b', r'\.go\b', r'\.rs\b', r'\.c\b', r'\.cpp\b',
        r'\.h\b', r'\.cs\b', r'\.rb\b', r'\.php\b', r'\.sql\b',
    ]
    
    @classmethod
    def is_safe_query(cls, query: str) -> tuple[bool, str]:
        """
        Validate if a search query is safe (general question, not code).
        Returns (is_safe, reason_if_unsafe)
        """
        if not query or not query.strip():
            return False, "Query is empty"
        
        query_lower = query.lower().strip()
        
        # Check for code patterns
        for pattern in cls.CODE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Query contains code pattern: {pattern}"
        
        # Check for file extensions (likely trying to search for code)
        for ext in cls.CODE_EXTENSIONS:
            if re.search(ext, query, re.IGNORECASE):
                return False, f"Query appears to reference a code file: {ext}"
        
        # Check for common code keywords that shouldn't be searched on the web
        code_keywords = [
            r'^\s*def\s', r'^\s*class\s', r'^\s*function\s',
            r'\{[\s\S]*\}', r'\[[\s\S]*\]',  # braces/brackets
            r'==\s*\w+', r'!=\s*\w+',  # comparisons
            r'&&\s', r'\|\|\s',  # logical operators
            r';\s*$',  # semicolons at end
        ]
        
        for keyword in code_keywords:
            if re.search(keyword, query):
                return False, "Query appears to contain code syntax"
        
        # Query should be a natural language question (at least 3 words for context)
        words = query.split()
        if len(words) < 2:
            return False, "Query too short - provide a general question"
        
        return True, ""


class SystemUpgradeGuard:
    """Detects and blocks attempts to upgrade system-level packages/dependencies."""
    
    SYSTEM_PACKAGE_PATTERNS = [
        r'upgrade\s+(python|java|node|ruby|go|rust|php|perl|r)',
        r'update\s+(python|java|node|ruby|go|rust|php|perl|r)\s+version',
        r'python\s+3\.\d+\s*->\s*3\.\d+',
        r'python\s+3\.\d+\s+to\s+3\.\d+',
        r'from\s+python\s+3\.\d+\s+to\s+3\.\d+',
        r'install\s+python\s+3\.\d+',
        r'use\s+python\s+3\.\d+',
        r'switch\s+to\s+python\s+3\.\d+',
        r'change\s+python\s+version',
        r'set\s+python\s+version',
        r'upgrade\s+node\.js',
        r'upgrade\s+nodejs',
        r'update\s+node\.js',
        r'upgrade\s+java',
        r'update\s+java',
        r'install\s+java\s+\d+',
        r'switch\s+java\s+version',
        r'change\s+java\s+version',
        r'upgrade\s+system',
        r'upgrade\s+os',
        r'distro\s+upgrade',
        r'apt\s+upgrade',
        r'yum\s+upgrade',
        r'dnf\s+upgrade',
        r'brew\s+upgrade',
        r'pip\s+install\s+--upgrade\s+pip',
        r'pip\s+install\s+--upgrade\s+python',
    ]
    
    # Languages/runtimes that shouldn't be upgraded via task
    PROTECTED_RUNTIMES = [
        'python', 'java', 'node', 'nodejs', 'ruby', 'go', 'rust', 'php', 'perl', 'r'
    ]
    
    @classmethod
    def is_safe_task(cls, task_description: str) -> tuple[bool, str, Optional[str]]:
        """
        Check if task is safe to execute.
        Returns (is_safe, reason_if_unsafe, report_content_if_abort)
        """
        task_lower = task_description.lower()
        
        # Check for system upgrade patterns
        for pattern in cls.SYSTEM_PACKAGE_PATTERNS:
            if re.search(pattern, task_lower):
                report = cls._create_abort_report(task_description, pattern)
                return False, f"Task involves system package upgrade: {pattern}", report
        
        # Check for version jump patterns (e.g., Python 3.8 -> 3.12)
        version_jump = re.findall(r'3\.(\d+)', task_lower)
        if len(version_jump) >= 2:
            versions = [int(v) for v in version_jump]
            if max(versions) - min(versions) >= 2:
                report = cls._create_abort_report(task_description, "major version jump detected")
                return False, "Task involves major version upgrade", report
        
        return True, "", None
    
    @classmethod
    def _create_abort_report(cls, task_description: str, reason: str) -> str:
        """Create an abort report explaining why the task can't be performed."""
        return f"""# Task Abort Report

## Task Description
{task_description}

## Reason for Abort
This task was blocked because it attempts to modify system-level components: {reason}

## Why This Is Blocked
The coding agent is designed to work within your existing development environment. 
Upgrading system-level packages (Python, Java, Node.js, etc.) can:
- Break system dependencies
- Cause compatibility issues with other applications
- Require root/admin privileges
- Potentially damage the operating system

## Recommended Actions
1. **Manual upgrade required**: System package upgrades should be done manually by you
2. **Use virtual environments**: For Python, use venv or conda to manage project-specific dependencies
3. **Containerized environments**: Consider Docker for isolated development environments
4. **Check documentation**: Refer to your OS/package manager documentation for proper upgrade procedures

## What The Agent Can Do Instead
- Update project-specific dependencies (requirements.txt, package.json, etc.)
- Refactor code to work with current versions
- Add compatibility layers
- Update build scripts for current environment

---
This report was generated automatically by the coding agent.
"""


class WebSearchTool:
    """Search the web for general information (with safeguards against code searches)."""
    name = "web_search"
    description = "Search the web for general information (NOT for code)"
    
    # JSON Schema for tool definition
    schema = {
        "name": "web_search",
        "description": "Search the web for general information. Use for questions about concepts, documentation, tutorials. NOT for searching code.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query - use natural language questions, NOT code"
                }
            },
            "required": ["query"]
        }
    }
    
    def __init__(self):
        self.guard = SearchGuard()
    
    def execute(self, query: str) -> str:
        """Execute a web search with validation."""
        # Validate query first
        is_safe, reason = self.guard.is_safe_query(query)
        if not is_safe:
            return f"Error: Search rejected - {reason}\n\nThis tool is for general questions only. Do not search for code. Ask general questions like 'how does React useEffect work' instead of 'def useEffect()'."

        # Free/default backend first (Hermes-style fallback pattern):
        # use DuckDuckGo via ddgs when premium API backends are not configured.
        ddgs_result = self._search_with_ddgs(query)
        if ddgs_result is not None:
            return ddgs_result

        # Optional premium backend (only if EXA_API_KEY is configured).
        exa_result = self._search_with_exa(query)
        if exa_result is not None:
            return exa_result

        return (
            "Error: No web search backend available.\n"
            "Install free backend: pip install ddgs\n"
            "Optional premium backend: pip install exa-py and set EXA_API_KEY"
        )

    def _search_with_ddgs(self, query: str) -> Optional[str]:
        """Search with ddgs (free) and return formatted results or None."""
        try:
            from ddgs import DDGS
        except ImportError:
            return None

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))

            if not results:
                return "No results found"

            formatted = ["Web Search Results (DuckDuckGo):\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title") or "No title"
                url = r.get("href") or r.get("url") or "No URL"
                snippet = r.get("body") or r.get("snippet") or ""
                formatted.append(f"{i}. {title}")
                formatted.append(f"   {url}")
                if snippet:
                    formatted.append(f"   {snippet[:200]}...")
                formatted.append("")
            return "\n".join(formatted)
        except Exception:
            # Keep fallback behavior graceful; next backend may still work.
            return None

    def _search_with_exa(self, query: str) -> Optional[str]:
        """Search with Exa (optional paid backend) and return results or None."""
        if not os.getenv("EXA_API_KEY"):
            return None

        try:
            from exa import Exa
        except ImportError:
            return None

        try:
            exa = Exa()
            results = exa.search(query, num_results=5)
            if results and results.results:
                formatted = ["Web Search Results (Exa):\n"]
                for i, r in enumerate(results.results, 1):
                    formatted.append(f"{i}. {r.title or 'No title'}")
                    formatted.append(f"   {r.url}")
                    if r.description:
                        formatted.append(f"   {r.description[:200]}...")
                    formatted.append("")
                return "\n".join(formatted)
            return "No results found"
        except Exception:
            return None


class ToolRegistry:
    """Registry of available tools."""
    
    def __init__(self, config: AgentConfig, repo):
        self.tools = {}
        self._register_default_tools(config, repo)
    
    def _register_default_tools(self, config: AgentConfig, repo):
        cwd = Path(repo.working_dir or ".") if repo else Path(".")
        checkpoint_manager = CheckpointManager(cwd, config.workspace_dir)
        self.tools["file_read"] = FileReadTool(cwd)
        self.tools["file_write"] = FileWriteTool(cwd, checkpoint_manager=checkpoint_manager)
        self.tools["bash"] = BashTool(cwd)
        self.tools["list_files"] = ListFilesTool(cwd)
        self.tools["grep"] = GrepTool(cwd)
        self.tools["web_search"] = WebSearchTool()
        self.tools["done"] = DoneTool()
        if repo:
            self.tools["git_status"] = GitStatusTool(repo)
    
    def register(self, tool):
        self.tools[tool.name] = tool
    
    def get(self, name: str):
        return self.tools.get(name)
    
    def list_tools(self) -> str:
        """List tools with JSON schemas for better LLM understanding."""
        import json
        tool_schemas = []
        
        for name, tool in self.tools.items():
            if hasattr(tool, 'schema'):
                tool_schemas.append(tool.schema)
            else:
                # Fallback for tools without schema
                tool_schemas.append({
                    "name": name,
                    "description": getattr(tool, 'description', ''),
                    "parameters": {"type": "object", "properties": {}}
                })
        
        # Return as formatted JSON for the prompt
        return "## Tools (JSON Schema)\n\n" + json.dumps(tool_schemas, indent=2)
    
    def list_tools_simple(self) -> str:
        """Simple list of tools with descriptions (fallback)."""
        return "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])
    
    def execute(self, tool_call: str) -> str:
        """Execute a tool from a tool call string like 'file_read(path="foo.txt")'"""
        try:
            # Parse tool call
            match = re.match(r'(\w+)\((.*)\)', tool_call.strip())
            if not match:
                return f"Error: Invalid tool call format: {tool_call}"
            
            tool_name = match.group(1)
            args_str = match.group(2)
            
            tool = self.get(tool_name)
            if not tool:
                return f"Error: Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}"
            
            # Parse arguments
            kwargs = {}
            if args_str.strip():
                # Parse key=value pairs, handling escaped quotes in values
                i = 0
                while i < len(args_str):
                    # Skip whitespace and commas
                    while i < len(args_str) and args_str[i] in ' ,\t\n':
                        i += 1
                    if i >= len(args_str):
                        break

                    # Find key
                    key_match = re.match(r'(\w+)\s*=\s*', args_str[i:])
                    if not key_match:
                        break
                    key = key_match.group(1)
                    i += key_match.end()

                    # Find value
                    if i < len(args_str) and args_str[i] in ('"', "'"):
                        quote = args_str[i]
                        i += 1
                        value_chars = []
                        while i < len(args_str):
                            if args_str[i] == '\\' and i + 1 < len(args_str):
                                # Escaped character
                                next_ch = args_str[i + 1]
                                if next_ch == 'n':
                                    value_chars.append('\n')
                                elif next_ch == 't':
                                    value_chars.append('\t')
                                else:
                                    value_chars.append(next_ch)
                                i += 2
                            elif args_str[i] == quote:
                                i += 1
                                break
                            else:
                                value_chars.append(args_str[i])
                                i += 1
                        kwargs[key] = ''.join(value_chars)
                    else:
                        # Unquoted value
                        val_match = re.match(r'([^,\s]*)', args_str[i:])
                        if val_match:
                            kwargs[key] = val_match.group(1)
                            i += val_match.end()
            
            return tool.execute(**kwargs)
            
        except Exception as e:
            return f"Error executing tool: {e}"
