from __future__ import annotations

try:
    from git import Repo, InvalidGitRepositoryError, GitCommandError
    GIT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Repo = None
    InvalidGitRepositoryError = Exception
    GitCommandError = Exception
    GIT_AVAILABLE = False


try:
    from .vector_memory import VectorMemoryManager, get_db_url
    VECTOR_MEMORY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    VectorMemoryManager = None
    get_db_url = None
    VECTOR_MEMORY_AVAILABLE = False