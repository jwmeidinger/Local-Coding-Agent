from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


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
