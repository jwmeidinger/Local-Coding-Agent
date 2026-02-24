#!/usr/bin/env python3
"""
Coding Agent - Lightweight coding automation framework inspired by OpenClaw.

A focused, coding-only agent that:
- Reads tasks and executes them using a planning → execution → review loop
- Uses a skills system for different types of coding tasks
- Creates isolated git branches for each task
- Never publishes branches (local only)

Inspired by OpenClaw but simplified for coding only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional dependencies with graceful degradation
try:
    from langchain_community.llms import Ollama
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from git import Repo, InvalidGitRepositoryError, GitCommandError
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

# Import vector memory system
try:
    from vector_memory import VectorMemoryManager, get_db_url
    VECTOR_MEMORY_AVAILABLE = True
except ImportError:
    VECTOR_MEMORY_AVAILABLE = False

# For source IP binding (bypass VPNs)
SOURCE_IP_AVAILABLE = False
_source_ip_adapter = None

try:
    import socket
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.poolmanager import PoolManager
    SOURCE_IP_AVAILABLE = True
    
    class _SourceIpAdapter(HTTPAdapter):
        """Custom HTTP adapter that binds to a specific source IP address."""
        
        def __init__(self, source_ip: str, *args, **kwargs):
            self.source_ip = source_ip
            super().__init__(*args, **kwargs)
        
        def init_poolmanager(self, *args, **kwargs):
            def socket_factory(*args, **kwargs):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.source_ip, 0))
                return sock
            
            kwargs['socket_options'] = None
            self.poolmanager = PoolManager(*args, socket_factory=socket_factory, **kwargs)
    
    _source_ip_adapter = _SourceIpAdapter
except ImportError:
    pass


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class AgentConfig:
    """Configuration for the Coding Agent."""
    # Multi-repo configuration
    repo_paths: List[Path] = field(default_factory=lambda: [Path(".")])
    ignore_patterns: List[str] = field(default_factory=list)
    workspace_dir: Path = field(default_factory=lambda: Path(".coding-agent"))
    
    # Task discovery
    tasks_dir: Path = field(default_factory=lambda: Path("tasks"))
    skills_dir: Path = field(default_factory=lambda: Path("skills"))
    
    # Git settings
    base_branch: str = "main"
    branch_prefix: str = "agent/"
    
    # LLM settings
    llm_url: str = "http://localhost:11434"
    model: str = "codellama"
    temperature: float = 0.2
    num_predict: int = 4096
    source_ip: str = ""  # Bind to specific local IP to bypass VPNs
    
    # Execution settings
    max_iterations: int = 5
    max_retries: int = 2
    auto_commit: bool = True
    verbose: bool = False


@dataclass
class SystemContext:
    """System context information passed to the LLM."""
    os_name: str = ""
    os_version: str = ""
    hostname: str = ""
    python_version: str = ""
    java_version: str = ""
    node_version: str = ""
    shell: str = ""
    user: str = ""
    cwd: str = ""
    
    @classmethod
    def detect(cls, repo_path: Path = None) -> "SystemContext":
        """Detect system information."""
        import platform
        import os
        import subprocess
        
        context = cls()
        
        import sys
        
        # OS info
        context.os_name = platform.system()  # Linux, Darwin, Windows
        context.os_version = platform.release()
        context.hostname = platform.node()
        
        # Python version
        context.python_version = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Java version
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Java outputs version to stderr
            if result.stderr:
                first_line = result.stderr.split('\n')[0]
                # Extract version like "21.0.2" or "11.0.20"
                import re
                match = re.search(r'(\d+\.\d+\.\d+)', first_line)
                if match:
                    context.java_version = f"Java {match.group(1)}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Node.js version
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout:
                context.node_version = f"Node.js {result.stdout.strip()}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Shell
        context.shell = os.environ.get("SHELL", "unknown")
        
        # User
        context.user = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
        
        # Current working directory
        context.cwd = str(repo_path) if repo_path else os.getcwd()
        
        return context
    
    def to_prompt(self) -> str:
        """Convert to prompt section for LLM."""
        lines = [
            "# System Context",
            "",
            f"- **OS**: {self.os_name} {self.os_version}",
            f"- **Hostname**: {self.hostname}",
            f"- **Python**: {self.python_version}",
        ]
        
        if self.java_version:
            lines.append(f"- **{self.java_version}**")
        
        if self.node_version:
            lines.append(f"- **{self.node_version}**")
        
        lines.extend([
            f"- **Shell**: {self.shell}",
            f"- **User**: {self.user}",
            f"- **Working Directory**: {self.cwd}",
        ])
        
        return "\n".join(lines)


@dataclass
class TaskContext:
    """Context for executing a task."""
    task_id: str
    task_description: str
    branch_name: str
    repo_path: Path  # Which repo this task belongs to
    system_info: str = ""  # System context (OS, Python, Java, Node, etc.)
    iteration: int = 0
    plan: Optional[str] = None
    execution_log: list = field(default_factory=list)
    files_modified: list = field(default_factory=list)
    review_feedback: Optional[str] = None


# ============================================================================
# Tool System
# ============================================================================

class FileReadTool:
    """Read file contents."""
    name = "file_read"
    description = "Read contents of a file"
    
    def execute(self, path: str) -> str:
        try:
            file_path = Path(path)
            if not file_path.exists():
                return f"Error: File {path} not found"
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"


class FileWriteTool:
    """Write content to a file."""
    name = "file_write"
    description = "Write content to a file"
    
    def execute(self, path: str, content: str) -> str:
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"


class BashTool:
    """Execute bash commands."""
    name = "bash"
    description = "Execute bash commands in the repository"
    
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
    
    def execute(self, path: str = ".", pattern: str = "*") -> str:
        try:
            dir_path = Path(path)
            if not dir_path.exists():
                return f"Error: Directory {path} not found"
            files = list(dir_path.glob(pattern))
            return "\n".join([str(f.relative_to(dir_path)) for f in files])
        except Exception as e:
            return f"Error listing files: {e}"


class GitStatusTool:
    """Check git status."""
    name = "git_status"
    description = "Check current git status"
    
    def __init__(self, repo):
        self.repo = repo
    
    def execute(self) -> str:
        try:
            return self.repo.git.status()
        except Exception as e:
            return f"Error checking git status: {e}"


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
    
    def __init__(self):
        self.guard = SearchGuard()
    
    def execute(self, query: str) -> str:
        """Execute a web search with validation."""
        # Validate query first
        is_safe, reason = self.guard.is_safe_query(query)
        if not is_safe:
            return f"Error: Search rejected - {reason}\n\nThis tool is for general questions only. Do not search for code. Ask general questions like 'how does React useEffect work' instead of 'def useEffect()'."
        
        # Use exa-py for web search
        try:
            from exa import Exa
            exa = Exa()
            results = exa.search(query, num_results=5)
            if results and results.results:
                formatted = ["Web Search Results:\n"]
                for i, r in enumerate(results.results, 1):
                    formatted.append(f"{i}. {r.title or 'No title'}")
                    formatted.append(f"   {r.url}")
                    if r.description:
                        formatted.append(f"   {r.description[:200]}...")
                    formatted.append("")
                return "\n".join(formatted)
            return "No results found"
        except ImportError:
            return "Error: Web search not available. Install exa-py: pip install exa-py"
        except Exception as e:
            return f"Search error: {e}"


class ToolRegistry:
    """Registry of available tools."""
    
    def __init__(self, config: AgentConfig, repo):
        self.tools = {}
        self._register_default_tools(config, repo)
    
    def _register_default_tools(self, config: AgentConfig, repo):
        cwd = Path(repo.working_dir or ".") if repo else Path(".")
        self.tools["file_read"] = FileReadTool()
        self.tools["file_write"] = FileWriteTool()
        self.tools["bash"] = BashTool(cwd)
        self.tools["list_files"] = ListFilesTool()
        self.tools["web_search"] = WebSearchTool()
        if repo:
            self.tools["git_status"] = GitStatusTool(repo)
    
    def register(self, tool):
        self.tools[tool.name] = tool
    
    def get(self, name: str):
        return self.tools.get(name)
    
    def list_tools(self) -> str:
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
                # Arg parsing: key="value" or key=value
                # arg[1] is the quoted group, arg[2] is the unquoted group;
                # re.findall returns "" (not None) for non-matching groups.
                for arg in re.findall(r'(\w+)=(?:"([^"]*)"|([^,\s]*))', args_str):
                    key = arg[0]
                    value = arg[1] if arg[1] != "" else arg[2]
                    kwargs[key] = value
            
            return tool.execute(**kwargs)
            
        except Exception as e:
            return f"Error executing tool: {e}"


# ============================================================================
# Skill System
# ============================================================================

@dataclass
class Skill:
    """A skill defines how to handle a specific type of task."""
    name: str
    description: str
    system_prompt: str
    planning_prompt: str
    review_prompt: str


class SkillRegistry:
    """Registry of available skills."""
    
    def __init__(self, skills_dir: Path):
        self.skills = {}
        self.skills_dir = skills_dir
        self._register_default_skills()
        self._load_custom_skills()
    
    def _register_default_skills(self):
        # Refactoring skill
        self.register(Skill(
            name="refactor",
            description="Refactor existing code to improve structure, readability, or performance",
            system_prompt="""You are an expert code refactoring specialist. Your role is to:
1. Analyze existing code for issues (complexity, duplication, poor naming)
2. Apply best practices (SOLID principles, clean code, design patterns)
3. Ensure all tests still pass after refactoring
4. Preserve external behavior while improving internal structure

When refactoring:
- Start by reading the target files
- Understand the current behavior before changing
- Make incremental changes
- Run any available tests after each change
- Explain the rationale for each refactoring decision""",
            planning_prompt="""Analyze the refactoring task and create a step-by-step plan:
1. Identify the files that need to be refactored
2. List specific issues to address (complexity, duplication, etc.)
3. Plan the refactoring steps in order
4. Identify any tests that should be run

Task: {task_description}

Create a detailed execution plan:""",
            review_prompt="""Review the refactoring changes:
1. Did the refactoring improve code quality?
2. Were any bugs introduced?
3. Is the code more readable now?
4. Are there any remaining issues?

Respond with:
- STATUS: [PASS/NEEDS_WORK]
- FEEDBACK: Detailed feedback on what was done well and what needs improvement
- SUGGESTIONS: Any additional refactoring that could be done"""
        ))
        
        # Feature implementation skill
        self.register(Skill(
            name="feature",
            description="Implement new features or functionality",
            system_prompt="""You are a senior software engineer implementing new features. Your role is to:
1. Understand the requirements thoroughly
2. Design simple, maintainable solutions
3. Write clean, well-documented code
4. Follow existing code patterns and conventions
5. Add appropriate tests

When implementing:
- Check existing code for patterns to follow
- Keep changes focused on the requirement
- Do not over-engineer - simple is better
- Consider edge cases and error handling""",
            planning_prompt="""Analyze the feature request and create an implementation plan:
1. What files need to be created or modified?
2. What is the minimal implementation to satisfy the requirement?
3. Are there existing patterns to follow?
4. What tests should be added?

Task: {task_description}

Create a detailed implementation plan:""",
            review_prompt="""Review the feature implementation:
1. Does it satisfy the requirements?
2. Is the code clean and maintainable?
3. Are there any bugs or edge cases missed?
4. Is it consistent with existing code patterns?

Respond with:
- STATUS: [PASS/NEEDS_WORK]
- FEEDBACK: What was done well and what needs work
- SUGGESTIONS: Improvements or missing pieces"""
        ))
        
        # Bug fix skill
        self.register(Skill(
            name="bugfix",
            description="Fix bugs and issues in code",
            system_prompt="""You are a debugging specialist. Your role is to:
1. Understand the reported issue thoroughly
2. Locate the root cause (not just symptoms)
3. Create minimal fixes that solve the problem
4. Verify the fix works
5. Check for similar issues elsewhere

When fixing bugs:
- Read relevant code to understand context
- Reproduce the issue if possible
- Fix the root cause, not symptoms
- Test your fix
- Consider edge cases""",
            planning_prompt="""Analyze the bug report and create a debugging plan:
1. What files are likely involved?
2. How can we reproduce or understand the issue?
3. What debugging steps should we take?
4. What is the plan for testing the fix?

Bug Report: {task_description}

Create a detailed debugging plan:""",
            review_prompt="""Review the bug fix:
1. Does it fix the root cause?
2. Are there any side effects?
3. Is the fix minimal and focused?
4. Are there tests to prevent regression?

Respond with:
- STATUS: [PASS/NEEDS_WORK]
- FEEDBACK: Analysis of the fix
- SUGGESTIONS: Any additional considerations"""
        ))
        
        # Documentation skill
        self.register(Skill(
            name="docs",
            description="Add or improve documentation",
            system_prompt="""You are a technical documentation specialist. Your role is to:
1. Write clear, helpful documentation
2. Add docstrings and comments where needed
3. Update README and guides
4. Ensure accuracy and completeness

When documenting:
- Focus on clarity over completeness
- Use examples where helpful
- Keep documentation close to code
- Update existing docs when code changes""",
            planning_prompt="""Analyze the documentation task:
1. What needs to be documented?
2. What format should be used?
3. Where should documentation be added?
4. Are there existing docs to update?

Task: {task_description}

Create a documentation plan:""",
            review_prompt="""Review the documentation:
1. Is it clear and accurate?
2. Are examples helpful?
3. Is it properly formatted?
4. Does it cover what is needed?

Respond with:
- STATUS: [PASS/NEEDS_WORK]
- FEEDBACK: Quality assessment
- SUGGESTIONS: Improvements"""
        ))
        
        # Unit test skill
        self.register(Skill(
            name="test",
            description="Create comprehensive unit tests for code",
            system_prompt="""You are a test-driven development specialist. Your role is to:
1. Write comprehensive unit tests that cover happy paths and edge cases
2. Follow existing testing patterns and conventions in the codebase
3. Use appropriate mocking for external dependencies
4. Name tests descriptively (what_input_expected_behavior)
5. Ensure tests are isolated and repeatable

When writing tests:
- First examine existing test files to understand the testing framework and patterns
- Read the target code thoroughly to understand what needs testing
- Test both success and failure scenarios
- Test edge cases (null, empty, boundary values)
- Use descriptive test names that explain what's being tested
- Add setup/teardown if needed
- Group related tests in describe/context blocks if the framework supports it""",
            planning_prompt="""Analyze the testing task and create a test plan:
1. What code needs to be tested? (identify target files/functions/classes)
2. What testing framework is being used? (pytest, jest, unittest, etc.)
3. Where should tests be placed? (test file naming conventions)
4. What are the main scenarios to test?
   - Happy paths (normal operation)
   - Error cases (invalid inputs, exceptions)
   - Edge cases (boundaries, empty values, nulls)
5. Are there existing tests to use as reference?

Task: {task_description}

Create a detailed test plan:""",
            review_prompt="""Review the unit tests:
1. Do tests cover the main functionality?
2. Are edge cases and error scenarios tested?
3. Do test names clearly describe what they test?
4. Are tests properly isolated (no dependencies between tests)?
5. Do tests follow the existing patterns in the codebase?
6. Is the testing framework used correctly?
7. Are assertions clear and meaningful?

Respond with:
- STATUS: [PASS/NEEDS_WORK]
- FEEDBACK: What tests are good and what's missing
- SUGGESTIONS: Additional test cases or improvements
- COVERAGE: Estimate of code coverage (high/medium/low)"""
        ))
    
    def _load_custom_skills(self):
        """Load custom skills from skills directory."""
        if not self.skills_dir.exists():
            return
        
        for skill_file in self.skills_dir.glob("*.json"):
            try:
                data = json.loads(skill_file.read_text())
                skill = Skill(**data)
                self.register(skill)
            except Exception as e:
                logging.warning(f"Failed to load skill {skill_file}: {e}")
    
    def register(self, skill: Skill):
        self.skills[skill.name] = skill
    
    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)
    
    def detect_skill(self, task_description: str) -> Skill:
        """Auto-detect the best skill for a task."""
        task_lower = task_description.lower()
        
        # Simple keyword matching - order matters (more specific first)
        if any(word in task_lower for word in ["test", "tests", "testing", "unittest", "pytest", "spec"]):
            skill = self.skills.get("test")
            if skill:
                return skill
        elif any(word in task_lower for word in ["bug", "fix", "error", "crash", "broken"]):
            skill = self.skills.get("bugfix")
            if skill:
                return skill
        elif any(word in task_lower for word in ["refactor", "clean", "restructure", "improve", "simplify"]):
            skill = self.skills.get("refactor")
            if skill:
                return skill
        elif any(word in task_lower for word in ["doc", "comment", "readme", "guide", "explain"]):
            skill = self.skills.get("docs")
            if skill:
                return skill
        
        # Default to feature skill (always registered by _register_default_skills)
        skill = self.skills.get("feature")
        if skill:
            return skill
        # Fallback to any available skill
        for skill in self.skills.values():
            return skill
        raise ValueError("No skills registered")
    
    def list_skills(self) -> str:
        return "\n".join([f"- {name}: {skill.description}" for name, skill in self.skills.items()])


# ============================================================================
# Codebase Memory System
# ============================================================================

@dataclass
class FileEntry:
    """Entry for a file in the codebase memory."""
    path: str
    last_modified: float
    size: int
    file_type: str
    summary: str = ""
    key_functions: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)


@dataclass
class CodebaseMemory:
    """Persistent memory of the codebase structure and key information."""
    indexed_at: str
    repo_path: Path
    files: dict[str, FileEntry] = field(default_factory=dict)
    test_files: list = field(default_factory=list)
    config_files: list = field(default_factory=list)
    entry_points: list = field(default_factory=list)
    dependencies: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "indexed_at": self.indexed_at,
            "repo_path": str(self.repo_path),
            "files": {
                path: {
                    "path": entry.path,
                    "last_modified": entry.last_modified,
                    "size": entry.size,
                    "file_type": entry.file_type,
                    "summary": entry.summary,
                    "key_functions": entry.key_functions,
                    "dependencies": entry.dependencies
                }
                for path, entry in self.files.items()
            },
            "test_files": self.test_files,
            "config_files": self.config_files,
            "entry_points": self.entry_points,
            "dependencies": self.dependencies
        }
    
    @staticmethod
    def from_dict(data: dict) -> CodebaseMemory:
        """Create from dictionary."""
        memory = CodebaseMemory(
            indexed_at=data.get("indexed_at", ""),
            repo_path=Path(data.get("repo_path", ".")),
            test_files=data.get("test_files", []),
            config_files=data.get("config_files", []),
            entry_points=data.get("entry_points", []),
            dependencies=data.get("dependencies", {})
        )
        
        for path, entry_data in data.get("files", {}).items():
            memory.files[path] = FileEntry(
                path=entry_data["path"],
                last_modified=entry_data["last_modified"],
                size=entry_data["size"],
                file_type=entry_data["file_type"],
                summary=entry_data.get("summary", ""),
                key_functions=entry_data.get("key_functions", []),
                dependencies=entry_data.get("dependencies", [])
            )
        
        return memory
    
    def get_summary(self, max_files: int = 50) -> str:
        """Get a summary of the codebase for context."""
        lines = [
            f"Codebase Index (last updated: {self.indexed_at})",
            f"Total files indexed: {len(self.files)}",
            f"Test files: {len(self.test_files)}",
            f"Entry points: {', '.join(self.entry_points) if self.entry_points else 'None identified'}",
            "",
            "Key Files:"
        ]
        
        # Show most important files first
        important_files = []
        
        # Config files
        for f in self.config_files[:5]:
            if f in self.files:
                important_files.append(("CONFIG", self.files[f]))
        
        # Entry points
        for f in self.entry_points[:5]:
            if f in self.files and f not in [x[1].path for x in important_files]:
                important_files.append(("ENTRY", self.files[f]))
        
        # Largest files (likely main modules)
        sorted_by_size = sorted(self.files.values(), key=lambda x: x.size, reverse=True)
        for entry in sorted_by_size[:10]:
            if entry.path not in [x[1].path for x in important_files]:
                important_files.append(("MODULE", entry))
        
        # Add to output
        for file_type, entry in important_files[:max_files]:
            summary = entry.summary[:100] + "..." if len(entry.summary) > 100 else entry.summary
            lines.append(f"  [{file_type}] {entry.path} - {summary}")
        
        # Test file patterns
        if self.test_files:
            lines.extend(["", "Test Files:"])
            for test_file in self.test_files[:10]:
                lines.append(f"  - {test_file}")
        
        return "\n".join(lines)


class CodebaseIndexer:
    """Indexes the codebase and maintains memory."""
    
    def __init__(self, repo_path: Path, memory_path: Path, llm_manager: Optional[LLMManager] = None):
        self.repo_path = repo_path
        self.memory_path = memory_path
        self.llm = llm_manager
        self.memory: Optional[CodebaseMemory] = None
        
    def load_or_create_memory(self) -> CodebaseMemory:
        """Load existing memory or create new if outdated."""
        if self.memory_path.exists():
            try:
                data = json.loads(self.memory_path.read_text())
                self.memory = CodebaseMemory.from_dict(data)
                
                # Check if memory is outdated (older than 24 hours or files changed)
                if self._is_memory_stale():
                    logging.info("Codebase memory is stale, re-indexing...")
                    return self.index_codebase()
                
                return self.memory
            except Exception as e:
                logging.warning(f"Failed to load memory: {e}, creating new...")
        
        return self.index_codebase()
    
    def _is_memory_stale(self) -> bool:
        """Check if memory needs refresh."""
        if not self.memory:
            return True
        
        # Check if memory is older than 24 hours
        try:
            indexed_time = datetime.fromisoformat(self.memory.indexed_at)
            if (datetime.now() - indexed_time).total_seconds() > 24 * 3600:
                return True
        except (ValueError, AttributeError):
            return True
        
        # Check if any tracked files have been modified
        for path, entry in self.memory.files.items():
            file_path = self.repo_path / path
            if file_path.exists():
                current_mtime = file_path.stat().st_mtime
                if current_mtime > entry.last_modified:
                    return True
        
        return False
    
    def index_codebase(self) -> CodebaseMemory:
        """Index the entire codebase."""
        logging.info("Indexing codebase...")
        
        memory = CodebaseMemory(
            indexed_at=datetime.now().isoformat(),
            repo_path=self.repo_path
        )
        
        # Find all source files
        source_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.cpp', '.c', '.h'}
        config_patterns = ['package.json', 'requirements.txt', 'Cargo.toml', 'pom.xml', 'setup.py', 'pyproject.toml', 'Dockerfile', 'docker-compose.yml']
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build', '.coding-agent']]
            
            for file in files:
                file_path = Path(root) / file
                relative_path = str(file_path.relative_to(self.repo_path))
                
                # Skip binary and large files
                try:
                    stat = file_path.stat()
                    if stat.st_size > 500000:  # Skip files > 500KB
                        continue
                except OSError:
                    continue
                
                # Determine file type
                ext = file_path.suffix.lower()
                is_source = ext in source_extensions
                is_test = 'test' in file.lower() or '__test__' in str(file_path) or 'spec' in file.lower()
                is_config = any(pattern in file for pattern in config_patterns)
                
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                except OSError:
                    continue
                
                # Create file entry
                entry = FileEntry(
                    path=relative_path,
                    last_modified=stat.st_mtime,
                    size=stat.st_size,
                    file_type=self._get_file_type(file_path, is_source, is_test, is_config)
                )
                
                # Extract key information
                if is_source:
                    entry.key_functions = self._extract_functions(content, ext)
                    entry.dependencies = self._extract_imports(content, ext)
                    
                    # Generate summary if LLM available
                    if self.llm and stat.st_size < 50000:  # Only for reasonably sized files
                        entry.summary = self._generate_summary(content, relative_path)
                    else:
                        entry.summary = self._simple_summary(content)
                
                memory.files[relative_path] = entry
                
                if is_test:
                    memory.test_files.append(relative_path)
                
                if is_config:
                    memory.config_files.append(relative_path)
                
                # Identify entry points
                if self._is_entry_point(file, content, ext):
                    memory.entry_points.append(relative_path)
        
        # Detect dependencies from config files
        memory.dependencies = self._detect_dependencies(memory.config_files)
        
        # Save memory
        self.memory = memory
        self._save_memory()
        
        logging.info(f"Indexed {len(memory.files)} files")
        return memory
    
    def _get_file_type(self, file_path: Path, is_source: bool, is_test: bool, is_config: bool) -> str:
        """Determine file type category."""
        if is_test:
            return "test"
        elif is_config:
            return "config"
        elif is_source:
            return "source"
        elif file_path.suffix in ['.md', '.txt', '.rst']:
            return "documentation"
        else:
            return "other"
    
    def _extract_functions(self, content: str, ext: str) -> list:
        """Extract function/class names from source code."""
        functions = []
        
        if ext == '.py':
            # Python: def function_name and class ClassName
            for match in re.finditer(r'^(?:async\s+)?def\s+(\w+)|^class\s+(\w+)', content, re.MULTILINE):
                func = match.group(1) or match.group(2)
                if func and not func.startswith('_'):
                    functions.append(func)
        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
            # JavaScript/TypeScript
            for match in re.finditer(r'(?:function|const|let|var)\s+(\w+)|(\w+)\s*[=:]\s*(?:async\s*)?\(|class\s+(\w+)', content):
                func = match.group(1) or match.group(2) or match.group(3)
                if func:
                    functions.append(func)
        
        return functions[:20]  # Limit to 20 functions
    
    def _extract_imports(self, content: str, ext: str) -> list:
        """Extract import statements."""
        imports = []
        
        if ext == '.py':
            for match in re.finditer(r'^(?:from|import)\s+(\S+)', content, re.MULTILINE):
                imports.append(match.group(1))
        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
            for match in re.finditer(r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]|require\(['\"]([^'\"]+)['\"]\)", content):
                imp = match.group(1) or match.group(2)
                if imp:
                    imports.append(imp)
        
        return list(set(imports))[:10]  # Limit and dedupe
    
    def _generate_summary(self, content: str, path: str) -> str:
        """Generate a summary using LLM."""
        if not self.llm:
            return self._simple_summary(content)
        
        try:
            prompt = f"""Provide a one-sentence summary of what this code file does:

File: {path}

Content (first 2000 chars):
{content[:2000]}

Summary:"""
            
            summary = self.llm.generate(prompt, "").strip()
            return summary[:200]  # Limit length
        except Exception as e:
            logging.warning(f"LLM summary generation failed: {e}")
            return self._simple_summary(content)
    
    def _simple_summary(self, content: str) -> str:
        """Generate a simple summary without LLM."""
        lines = content.split('\n')
        
        # Look for docstring or comments at start
        for line in lines[:10]:
            line = line.strip()
            if line and not line.startswith('import') and not line.startswith('from'):
                # Clean up common comment markers
                for prefix in ['#', '//', '/*', '*', '"""', "'''"]:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                if len(line) > 10:
                    return line[:150]
        
        return "Source code file"
    
    def _is_entry_point(self, filename: str, content: str, ext: str) -> bool:
        """Check if file is an entry point."""
        # Common entry point patterns
        entry_patterns = ['main.py', 'index.js', 'app.py', 'server.js', 'main.go', 'main.rs']
        if filename in entry_patterns:
            return True
        
        # Check content for entry point patterns
        if ext == '.py':
            if '__main__' in content or 'if __name__' in content:
                return True
        elif ext in ['.js', '.ts']:
            if 'module.exports' in content or 'export default' in content:
                return True
        
        return False
    
    def _detect_dependencies(self, config_files: list) -> dict:
        """Detect project dependencies from config files."""
        deps = {}
        
        for config_file in config_files:
            config_path = self.repo_path / config_file
            if not config_path.exists():
                continue
            
            try:
                content = config_path.read_text()
                
                if 'requirements.txt' in config_file:
                    deps['python'] = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')][:20]
                elif 'package.json' in config_file:
                    import json
                    data = json.loads(content)
                    deps['javascript'] = list(data.get('dependencies', {}).keys())[:20]
                    deps['devDependencies'] = list(data.get('devDependencies', {}).keys())[:10]
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                pass
        
        return deps
    
    def _save_memory(self):
        """Save memory to disk."""
        if self.memory:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_path.write_text(json.dumps(self.memory.to_dict(), indent=2))
    
    def update_for_task(self, modified_files: list, llm_manager: LLMManager):
        """Update memory after a task modifies files."""
        if not self.memory:
            return
        
        logging.info(f"Updating memory for {len(modified_files)} modified files")
        
        for file_path in modified_files:
            # modified_files from git are relative to repo root
            full_path = self.repo_path / file_path
            relative_path = str(Path(file_path).as_posix())
            
            if not full_path.exists():
                # File was deleted
                if relative_path in self.memory.files:
                    del self.memory.files[relative_path]
                continue
            
            # Re-index the file
            try:
                stat = full_path.stat()
                content = full_path.read_text(encoding='utf-8', errors='ignore')
                ext = full_path.suffix.lower()
                
                entry = FileEntry(
                    path=relative_path,
                    last_modified=stat.st_mtime,
                    size=stat.st_size,
                    file_type=self._get_file_type(full_path, ext in {'.py', '.js', '.ts'}, 'test' in relative_path, False)
                )
                
                entry.key_functions = self._extract_functions(content, ext)
                entry.dependencies = self._extract_imports(content, ext)
                entry.summary = self._generate_summary(content, relative_path) if llm_manager else self._simple_summary(content)
                
                self.memory.files[relative_path] = entry
            except Exception as e:
                logging.warning(f"Failed to update memory for {file_path}: {e}")
        
        # Update timestamp
        self.memory.indexed_at = datetime.now().isoformat()
        self._save_memory()


# ============================================================================
# LLM Manager
# ============================================================================

class LLMManager:
    """Manages LLM interactions."""
    
    def __init__(self, config: AgentConfig):
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain is required. Install: pip install langchain-community")
        
        self.config = config
        self.llm = Ollama(
            base_url=config.llm_url,
            model=config.model,
            temperature=config.temperature,
            num_predict=config.num_predict
        )
        
        # Setup source IP binding if configured
        self.session = None
        if config.source_ip and SOURCE_IP_AVAILABLE and _source_ip_adapter:
            try:
                self.session = requests.Session()
                adapter = _source_ip_adapter(config.source_ip)
                self.session.mount("http://", adapter)
                self.session.mount("https://", adapter)
            except Exception as e:
                import logging
                logging.warning(f"Failed to setup source IP binding: {e}")
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text using LLM."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        return self.llm.invoke(full_prompt)
    
    def extract_tool_calls(self, text: str) -> list:
        """Extract tool calls from LLM output."""
        tool_calls = []
        for line in text.split('\n'):
            line = line.strip()
            if re.match(r'^(file_read|file_write|bash|list_files|git_status|web_search)\(', line):
                tool_calls.append(line)
        return tool_calls


# ============================================================================
# Core Execution Engine
# ============================================================================

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
                self.logger.error(f"Failed to initialize vector memory: {e}")
                self.logger.warning("Falling back to no memory")
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

{context_info}{skill.planning_prompt.format(task_description=context.task_description)}"""
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
        
        prompt = f"""{context.system_info}

{memory_context}Execute this plan step by step:

Task: {context.task_description}

Plan:
{context.plan}

Available tools:
{self.tools.list_tools()}

IMPORTANT: The web_search tool is for GENERAL QUESTIONS ONLY. It will REJECT searches that contain code.
- GOOD: "how does React useEffect work", "what is Docker container", "explain REST APIs"
- BAD:  "def useEffect()", "function foo() {{}}", "import os" (will be rejected)

IMPORTANT: The bash tool has SAFETY GUARDS that block dangerous commands:
- Blocked: sudo apt install, brew upgrade python, pip install --upgrade python, nvm install, curl | bash, rm -rf /
- Allowed: pip install -r requirements.txt, npm install, git commands, regular project commands

Execute the plan by calling tools one at a time. For each step:
1. Explain what you are doing
2. Call the appropriate tool using the format: tool_name(arg1="value1", arg2="value2")
3. Wait for the result (will be provided)

Start executing:
"""
        
        max_steps = 20
        for step in range(max_steps):
            response = self.llm.generate(prompt, skill.system_prompt)
            context.execution_log.append(f"Step {step + 1}: {response[:200]}...")
            
            # Extract and execute tool calls
            tool_calls = self.llm.extract_tool_calls(response)
            
            if not tool_calls:
                # No tool calls - check if task is complete
                if "complete" in response.lower() or "done" in response.lower():
                    return True
                prompt += f"\n\nResponse: {response}\n\nContinue executing or indicate completion:"
            else:
                # Execute first tool call
                tool_call = tool_calls[0]
                self.logger.info(f"Executing: {tool_call}")
                result = self.tools.execute(tool_call)
                prompt += f"\n\nExecuted: {tool_call}\nResult: {result}\n\nContinue with next step:"
        
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


# ============================================================================
# Main Application
# ============================================================================

class CodingAgent:
    """Main Coding Agent application with multi-repo support."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
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
                self.repos[repo_path] = repo
                self.engines[repo_path] = ExecutionEngine(config, repo, repo_path)
                self.logger.info(f"Loaded repo: {repo_path}")
            except Exception as e:
                self.logger.warning(f"Could not load repo {repo_path}: {e}")
        
        if not self.repos:
            raise RuntimeError("No valid git repositories found!")
        
        # Ensure directories exist
        config.tasks_dir.mkdir(parents=True, exist_ok=True)
        config.skills_dir.mkdir(parents=True, exist_ok=True)
        config.workspace_dir.mkdir(parents=True, exist_ok=True)
    
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
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # File handler
            file_handler = logging.FileHandler(
                self.config.workspace_dir / "agent.log"
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
                    self.logger.info(f"✓ Task completed: {task_id}")
                    self._archive_task(task_path)
                else:
                    self.logger.error(f"✗ Task failed: {task_id}")
                    
            except Exception as e:
                self.logger.error(f"✗ Error processing {task_id}: {e}")
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


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> AgentConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Coding Agent - Autonomous coding assistant inspired by OpenClaw",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (current directory)
  python coding_agent.py
  
  # Multiple repos with auto-discovery
  python coding_agent.py --repos ~/Code --auto-discover --ignore "archive/*" "old-*"
  
  # Explicitly specify repos
  python coding_agent.py --repo ~/Code/project1 --repo ~/Code/project2
  
  # Specify repos via file
  python coding_agent.py --repo-list repos.txt
  
  # Index all repos
  python coding_agent.py --repos ~/Code --auto-discover --index
  
  # List available skills
  python coding_agent.py --list-skills
        """
    )
    
    # Multi-repo options
    parser.add_argument("--repo", type=Path, action="append", dest="repo_paths",
                        help="Path to git repository (can specify multiple)")
    parser.add_argument("--repos", type=Path,
                        help="Parent directory containing multiple repos")
    parser.add_argument("--auto-discover", action="store_true",
                        help="Auto-discover git repos in --repos directory")
    parser.add_argument("--ignore", type=str, action="append", dest="ignore_patterns",
                        help="Patterns to ignore (e.g., 'archive/*', 'temp-*')")
    parser.add_argument("--repo-list", type=Path,
                        help="File containing list of repo paths (one per line)")
    
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"),
                        help="Directory containing task files")
    parser.add_argument("--skills-dir", type=Path, default=Path("skills"),
                        help="Directory containing custom skills")
    parser.add_argument("--base-branch", default="main",
                        help="Base git branch")
    parser.add_argument("--branch-prefix", default="agent/",
                        help="Prefix for agent branches")
    
    parser.add_argument("--llm-url", default="http://localhost:11434",
                        help="LLM API server URL")
    parser.add_argument("--model", default="codellama",
                        help="LLM model to use")
    parser.add_argument("--source-ip", default="",
                        help="Bind to specific local IP (e.g., 10.152.50.103) to bypass VPNs")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="LLM temperature")
    
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Max execution iterations per task")
    parser.add_argument("--no-commit", action="store_true",
                        help="Don't auto-commit changes")
    
    parser.add_argument("--list-skills", action="store_true",
                        help="List available skills and exit")
    parser.add_argument("--index", action="store_true",
                        help="Index the codebase and exit (builds vector memory)")
    parser.add_argument("--search", type=str, metavar="QUERY",
                        help="Search codebase and exit (for testing vector memory)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Collect repo paths
    repo_paths = []
    
    # From explicit --repo args
    if args.repo_paths:
        repo_paths.extend(args.repo_paths)
    
    # From repo list file
    if args.repo_list and args.repo_list.exists():
        with open(args.repo_list) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    repo_paths.append(Path(line))
    
    # From --repos directory with auto-discovery
    if args.repos:
        if args.auto_discover:
            # Find all git repos in the directory
            for item in args.repos.iterdir():
                if item.is_dir() and (item / '.git').exists():
                    repo_paths.append(item)
        else:
            # Just use the directory itself as a single repo
            repo_paths.append(args.repos)
    
    # Default to current directory if nothing specified
    if not repo_paths:
        repo_paths = [Path(".")]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_repo_paths = []
    for path in repo_paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_repo_paths.append(path)
    repo_paths = unique_repo_paths
    
    if args.list_skills:
        skills = SkillRegistry(args.skills_dir)
        print("Available Skills:")
        print(skills.list_skills())
        sys.exit(0)
    
    if args.index:
        if not VECTOR_MEMORY_AVAILABLE:
            print("Error: Vector memory not available. Install psycopg2-binary", file=sys.stderr)
            sys.exit(1)
        
        for repo_path in repo_paths:
            print(f"\nIndexing codebase: {repo_path}")
            try:
                memory_manager = VectorMemoryManager(repo_path)
                stats = memory_manager.index_codebase()
                print(f"  Files indexed: {stats['indexed']}")
                print(f"  Files updated: {stats['updated']}")
                print(f"  Files skipped: {stats['skipped']}")
                print(f"  Errors: {stats['errors']}")
                
                summary = memory_manager.get_codebase_summary()
                print(f"  {summary}")
                memory_manager.close()
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
        
        print("\nIndexing complete!")
        sys.exit(0)
    
    if args.search:
        if not VECTOR_MEMORY_AVAILABLE:
            print("Error: Vector memory not available. Install psycopg2-binary", file=sys.stderr)
            sys.exit(1)
        
        print(f"Searching for: {args.search}")
        print(f"In repos: {[str(p) for p in repo_paths]}")
        
        for repo_path in repo_paths:
            print(f"\n--- Results from {repo_path.name} ---")
            try:
                memory_manager = VectorMemoryManager(repo_path)
                results = memory_manager.search_codebase(args.search, limit=10)
                if results:
                    for i, r in enumerate(results, 1):
                        print(f"{i}. {r['file_path']} (score: {r['combined_score']:.3f})")
                        print(f"   Summary: {r.get('summary') or ''}")
                        kf = r.get('key_functions') or []
                        if kf:
                            print(f"   Functions: {', '.join(kf[:5])}")
                else:
                    print("  No results")
                memory_manager.close()
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
        
        sys.exit(0)
    
    return AgentConfig(
        repo_paths=repo_paths,
        ignore_patterns=args.ignore_patterns or [],
        tasks_dir=args.tasks_dir,
        skills_dir=args.skills_dir,
        base_branch=args.base_branch,
        branch_prefix=args.branch_prefix,
        llm_url=args.llm_url,
        model=args.model,
        source_ip=args.source_ip,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        auto_commit=not args.no_commit,
        verbose=args.verbose,
    )


def main() -> int:
    """Main entry point."""
    agent = None
    try:
        config = parse_args()
        agent = CodingAgent(config)
        agent.run()
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        # Ensure cleanup happens even on errors
        if agent:
            agent.close()


if __name__ == "__main__":
    sys.exit(main())
