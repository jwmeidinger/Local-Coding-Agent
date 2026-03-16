from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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
