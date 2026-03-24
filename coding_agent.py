#!/usr/bin/env python3
"""
Coding Agent - Lightweight coding automation framework inspired by OpenClaw.

This file is now a thin entrypoint and compatibility shim.
Core implementation lives under `coding_agent_core/`.
"""

from __future__ import annotations

import sys

from coding_agent_core.app import CodingAgent
from coding_agent_core.cli import parse_args
from coding_agent_core.config import AgentConfig, SystemContext, TaskContext
from coding_agent_core.deps import (
    GIT_AVAILABLE,
    VECTOR_MEMORY_AVAILABLE,
    GitCommandError,
    InvalidGitRepositoryError,
    Repo,
    VectorMemoryManager,
    get_db_url,
)
from coding_agent_core.engine import ExecutionEngine
from coding_agent_core.indexer import CodebaseIndexer, CodebaseMemory, FileEntry
from coding_agent_core.llm import LLMManager
from coding_agent_core.network import apply_source_ip_binding
from coding_agent_core.skills import Skill, SkillRegistry
from coding_agent_core.tools import (
    BashTool,
    CheckpointManager,
    FileEditTool,
    FileReadTool,
    FileTreeTool,
    FileWriteTool,
    GitDiffTool,
    GitStatusTool,
    ListFilesTool,
    RevertFileTool,
    SearchGuard,
    SystemUpgradeGuard,
    ToolRegistry,
    WebSearchTool,
)


__all__ = [
    "AgentConfig",
    "BashTool",
    "CheckpointManager",
    "CodebaseIndexer",
    "CodebaseMemory",
    "CodingAgent",
    "ExecutionEngine",
    "FileEditTool",
    "FileEntry",
    "FileReadTool",
    "FileTreeTool",
    "FileWriteTool",
    "GIT_AVAILABLE",
    "GitCommandError",
    "GitDiffTool",
    "GitStatusTool",
    "InvalidGitRepositoryError",
    "LLMManager",
    "ListFilesTool",
    "Repo",
    "RevertFileTool",
    "SearchGuard",
    "Skill",
    "SkillRegistry",
    "SystemContext",
    "SystemUpgradeGuard",
    "TaskContext",
    "ToolRegistry",
    "VECTOR_MEMORY_AVAILABLE",
    "VectorMemoryManager",
    "WebSearchTool",
    "apply_source_ip_binding",
    "get_db_url",
    "main",
    "parse_args",
]


def main() -> int:
    """Main entry point."""
    agent = None
    try:
        config = parse_args()

        # Apply global source-IP binding before HTTP clients are created.
        if config.source_ip:
            apply_source_ip_binding(config.source_ip)

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
        if agent:
            agent.close()


if __name__ == "__main__":
    sys.exit(main())