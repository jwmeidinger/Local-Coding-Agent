"""
Vector Memory System for Coding Agent using Postgres + pgvector.

Similar to OpenClaw's memory system but using Postgres instead of SQLite.
Features:
- Vector embeddings for semantic code search
- Hybrid search (vector + full-text)
- HNSW indexes for fast nearest neighbor queries
- Incremental updates
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Try to import psycopg2
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


@dataclass
class CodeFile:
    """Represents a code file in the memory."""
    path: str
    file_type: str
    content_hash: str
    content_preview: str
    summary: str
    key_functions: List[str]
    dependencies: List[str]
    embedding: Optional[List[float]] = None
    last_modified: datetime = field(default_factory=datetime.now)


class VectorMemoryManager:
    """Manages vector-based memory of the codebase using Postgres + pgvector."""
    
    def __init__(self, repo_path: Path, db_config: Optional[Dict[str, Any]] = None):
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError(
                "psycopg2 is required for vector memory. "
                "Install with: pip install psycopg2-binary"
            )
        
        self.repo_path = repo_path
        self.logger = logging.getLogger("coding-agent.memory")
        
        # Database configuration
        self.db_config = db_config or {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'coding_agent'),
            'user': os.getenv('DB_USER', 'agent'),
            'password': os.getenv('DB_PASSWORD', 'agent_password')
        }
        
        self.conn = None
        self._connect()
        self._ensure_schema()
    
    def _connect(self):
        """Connect to Postgres database."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.logger.info(f"Connected to Postgres database: {self.db_config['database']}")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _ensure_schema(self):
        """Ensure database schema exists."""
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create code_embeddings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS code_embeddings (
                    id SERIAL PRIMARY KEY,
                    repo_path TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT,
                    content_hash TEXT,
                    content_preview TEXT,
                    summary TEXT,
                    key_functions TEXT[],
                    dependencies TEXT[],
                    embedding VECTOR(1536),
                    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(repo_path, file_path)
                );
            """)
            
            # Create HNSW index for vector similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS code_embeddings_hnsw_idx 
                ON code_embeddings USING hnsw (embedding vector_cosine_ops);
            """)
            
            # Create indexes for filtering
            cur.execute("""
                CREATE INDEX IF NOT EXISTS code_embeddings_type_idx 
                ON code_embeddings(file_type);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS code_embeddings_repo_idx 
                ON code_embeddings(repo_path);
            """)
            
            # Create task_memory table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS task_memory (
                    id SERIAL PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    repo_path TEXT NOT NULL,
                    task_description TEXT,
                    branch_name TEXT,
                    skill_used TEXT,
                    files_modified TEXT[],
                    execution_log TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding VECTOR(1536),
                    UNIQUE(task_id)
                );
            """)
            
            # Create index for task similarity
            cur.execute("""
                CREATE INDEX IF NOT EXISTS task_memory_hnsw_idx 
                ON task_memory USING hnsw (embedding vector_cosine_ops);
            """)
            
            self.conn.commit()
            self.logger.info("Database schema initialized")
    
    def _compute_hash(self, content: str) -> str:
        """Compute MD5 hash of content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _generate_embedding(self, text: str, embedding_model: Any = None) -> List[float]:
        """Generate embedding vector for text.
        
        If embedding_model is provided (e.g., OpenAI client), use it.
        Otherwise, use a simple fallback (not for production).
        """
        if embedding_model:
            try:
                # Try to use provided embedding model
                response = embedding_model.embeddings.create(
                    model="text-embedding-3-small",
                    input=text[:8000]  # Limit text length
                )
                return response.data[0].embedding
            except Exception as e:
                self.logger.warning(f"Failed to generate embedding with model: {e}")
        
        # Fallback: Create a simple hash-based embedding (not semantic!)
        # This is just for testing when no embedding service is available
        self.logger.warning("Using fallback embedding (not semantic)")
        
        # Simple bag-of-words style embedding
        words = re.findall(r'\w+', text.lower())
        embedding = [0.0] * 1536
        
        for word in words[:100]:  # Limit to first 100 words
            # Hash word to position in embedding
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            pos = hash_val % 1536
            embedding[pos] = min(1.0, embedding[pos] + 0.1)
        
        return embedding
    
    def index_codebase(self, embedding_model: Any = None, ignore_patterns: List[str] = None) -> Dict[str, Any]:
        """Index the entire codebase with optional ignore patterns."""
        self.logger.info("Starting codebase indexing...")
        
        stats = {'indexed': 0, 'updated': 0, 'errors': 0, 'skipped': 0}
        repo_str = str(self.repo_path)
        ignore_patterns = ignore_patterns or []
        
        # Get list of files currently in database for this repo
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT file_path, content_hash FROM code_embeddings WHERE repo_path = %s",
                (repo_str,)
            )
            existing_files = {row[0]: row[1] for row in cur.fetchall()}
        
        # Find all source files
        source_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.cpp', '.c', '.h', '.rb', '.php'}
        current_files = set()
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in [
                '.git', 'node_modules', '__pycache__', '.venv', 'venv', 
                'dist', 'build', '.coding-agent', 'target', '.idea', '.vscode'
            ]]
            
            for file in files:
                file_path = Path(root) / file
                relative_path = str(file_path.relative_to(self.repo_path))
                current_files.add(relative_path)
                
                # Check if file needs indexing
                ext = file_path.suffix.lower()
                if ext not in source_extensions:
                    continue
                
                try:
                    stat = file_path.stat()
                    if stat.st_size > 500000:  # Skip files > 500KB
                        stats['skipped'] += 1
                        continue
                    
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    content_hash = self._compute_hash(content)
                    
                    # Check if file already exists and hasn't changed
                    if relative_path in existing_files:
                        if existing_files[relative_path] == content_hash:
                            continue  # Skip unchanged files
                    
                    # Determine file type
                    is_test = 'test' in file.lower() or 'spec' in file.lower()
                    file_type = 'test' if is_test else 'source'
                    
                    # Extract metadata
                    key_functions = self._extract_functions(content, ext)
                    dependencies = self._extract_imports(content, ext)
                    summary = self._generate_summary(content, relative_path)
                    
                    # Generate embedding
                    embedding_text = f"{relative_path}\n{summary}\n{' '.join(key_functions)}"
                    embedding = self._generate_embedding(embedding_text, embedding_model)
                    
                    # Store in database (batch for performance)
                    with self.conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO code_embeddings 
                            (repo_path, file_path, file_type, content_hash, content_preview, 
                             summary, key_functions, dependencies, embedding, last_modified, indexed_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                            ON CONFLICT (repo_path, file_path) 
                            DO UPDATE SET
                                file_type = EXCLUDED.file_type,
                                content_hash = EXCLUDED.content_hash,
                                content_preview = EXCLUDED.content_preview,
                                summary = EXCLUDED.summary,
                                key_functions = EXCLUDED.key_functions,
                                dependencies = EXCLUDED.dependencies,
                                embedding = EXCLUDED.embedding,
                                last_modified = EXCLUDED.last_modified,
                                indexed_at = CURRENT_TIMESTAMP
                        """, (
                            repo_str, relative_path, file_type, content_hash,
                            content[:1000], summary, key_functions, dependencies,
                            embedding, datetime.fromtimestamp(stat.st_mtime)
                        ))
                    
                    if relative_path in existing_files:
                        stats['updated'] += 1
                    else:
                        stats['indexed'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error indexing {relative_path}: {e}")
                    stats['errors'] += 1
        
        # Commit all file insertions/updates in one transaction
        try:
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to commit indexing changes: {e}")
            self.conn.rollback()
            raise

        # Remove files that no longer exist
        deleted_files = set(existing_files.keys()) - current_files
        if deleted_files:
            with self.conn.cursor() as cur:
                for file_path in deleted_files:
                    cur.execute(
                        "DELETE FROM code_embeddings WHERE repo_path = %s AND file_path = %s",
                        (repo_str, file_path)
                    )
            self.conn.commit()
            self.logger.info(f"Removed {len(deleted_files)} deleted files from index")
        
        # Update statistics after bulk operation
        try:
            with self.conn.cursor() as cur:
                cur.execute("ANALYZE code_embeddings")
            self.conn.commit()
        except Exception as e:
            self.logger.warning(f"Could not update statistics: {e}")
        
        self.logger.info(f"Indexing complete: {stats}")
        return stats
    
    def search_codebase(
        self, 
        query: str, 
        embedding_model: Any = None,
        file_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search codebase using hybrid (vector + text) search."""
        
        # Generate embedding for query
        query_embedding = self._generate_embedding(query, embedding_model)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Use hybrid search combining vector similarity and text search
            cur.execute("""
                WITH vector_scores AS (
                    SELECT 
                        file_path,
                        summary,
                        file_type,
                        key_functions,
                        dependencies,
                        1 - (embedding <=> %s::vector) AS vector_score
                    FROM code_embeddings
                    WHERE repo_path = %s
                    AND embedding IS NOT NULL
                    AND (%s IS NULL OR file_type = %s)
                ),
                text_scores AS (
                    SELECT 
                        file_path,
                        ts_rank(
                            to_tsvector('english', COALESCE(summary, '') || ' ' || COALESCE(content_preview, '')),
                            plainto_tsquery('english', %s)
                        ) AS text_score
                    FROM code_embeddings
                    WHERE repo_path = %s
                    AND (%s IS NULL OR file_type = %s)
                )
                SELECT 
                    v.file_path,
                    v.summary,
                    v.file_type,
                    v.key_functions,
                    v.dependencies,
                    v.vector_score,
                    COALESCE(t.text_score, 0) AS text_score,
                    (0.7 * v.vector_score + 0.3 * COALESCE(t.text_score, 0)) AS combined_score
                FROM vector_scores v
                LEFT JOIN text_scores t ON v.file_path = t.file_path
                ORDER BY combined_score DESC
                LIMIT %s
            """, (
                query_embedding, str(self.repo_path), file_type, file_type,
                query, str(self.repo_path), file_type, file_type,
                limit
            ))
            
            results = cur.fetchall()
            return [dict(row) for row in results]
    
    def get_file_context(self, file_paths: List[str]) -> str:
        """Get context for specific files."""
        if not file_paths:
            return ""
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT file_path, summary, key_functions, dependencies
                FROM code_embeddings
                WHERE repo_path = %s
                AND file_path = ANY(%s)
            """, (str(self.repo_path), file_paths))
            
            rows = cur.fetchall()
            
            if not rows:
                return ""
            
            lines = ["Relevant Code Files:"]
            for row in rows:
                lines.append(f"\n  📄 {row['file_path']}")
                lines.append(f"     Summary: {row['summary']}")
                if row['key_functions']:
                    lines.append(f"     Functions: {', '.join(row['key_functions'][:5])}")
                if row['dependencies']:
                    lines.append(f"     Dependencies: {', '.join(row['dependencies'][:5])}")
            
            return "\n".join(lines)
    
    def get_codebase_summary(self) -> str:
        """Get a summary of the indexed codebase."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE file_type = 'source') as source,
                    COUNT(*) FILTER (WHERE file_type = 'test') as test,
                    MAX(indexed_at) as last_indexed
                FROM code_embeddings
                WHERE repo_path = %s
            """, (str(self.repo_path),))
            
            row = cur.fetchone()
            if not row or row[0] == 0:
                return "No files indexed yet. Run indexing first."
            
            return f"""Codebase Index Summary:
- Total files: {row[0]}
- Source files: {row[1]}
- Test files: {row[2]}
- Last indexed: {row[3] or 'Unknown'}"""
    
    def update_for_task(
        self, 
        modified_files: List[str], 
        task_description: str,
        branch_name: str,
        skill_used: str,
        embedding_model: Any = None
    ):
        """Update memory after a task modifies files."""
        self.logger.info(f"Updating memory for {len(modified_files)} modified files")
        
        repo_str = str(self.repo_path)
        
        # Re-index modified files
        for file_path in modified_files:
            # Handle both relative and absolute paths safely
            # file_path from git is relative to repo root
            file_path_obj = Path(file_path)
            if file_path_obj.is_absolute():
                # If somehow absolute, check it's under repo_path
                try:
                    relative_path = file_path_obj.relative_to(self.repo_path)
                    full_path = file_path_obj
                except ValueError:
                    # Path not under repo, skip
                    self.logger.warning(f"Skipping file outside repo: {file_path}")
                    continue
            else:
                # Normal case: relative path from git
                full_path = self.repo_path / file_path_obj
            
            # Security: ensure path doesn't escape repo
            try:
                full_path.resolve().relative_to(self.repo_path.resolve())
            except ValueError:
                self.logger.warning(f"Skipping path traversal attempt: {file_path}")
                continue
            
            if not full_path.exists():
                # File was deleted
                with self.conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM code_embeddings WHERE repo_path = %s AND file_path = %s",
                        (repo_str, file_path)
                    )
                continue
            
            try:
                content = full_path.read_text(encoding='utf-8', errors='ignore')
                content_hash = self._compute_hash(content)
                ext = full_path.suffix.lower()
                
                is_test = 'test' in file_path.lower() or 'spec' in file_path.lower()
                file_type = 'test' if is_test else 'source'
                
                key_functions = self._extract_functions(content, ext)
                dependencies = self._extract_imports(content, ext)
                summary = self._generate_summary(content, file_path)
                
                # Generate new embedding
                embedding_text = f"{file_path}\n{summary}\n{' '.join(key_functions)}"
                embedding = self._generate_embedding(embedding_text, embedding_model)
                
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO code_embeddings 
                        (repo_path, file_path, file_type, content_hash, content_preview, 
                         summary, key_functions, dependencies, embedding, last_modified, indexed_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT (repo_path, file_path) 
                        DO UPDATE SET
                            file_type = EXCLUDED.file_type,
                            content_hash = EXCLUDED.content_hash,
                            content_preview = EXCLUDED.content_preview,
                            summary = EXCLUDED.summary,
                            key_functions = EXCLUDED.key_functions,
                            dependencies = EXCLUDED.dependencies,
                            embedding = EXCLUDED.embedding,
                            last_modified = CURRENT_TIMESTAMP,
                            indexed_at = CURRENT_TIMESTAMP
                    """, (
                        repo_str, file_path, file_type, content_hash,
                        content[:1000], summary, key_functions, dependencies,
                        embedding
                    ))
            except Exception as e:
                self.logger.warning(f"Failed to update {file_path}: {e}")
        
        # Store task memory
        if task_description:
            try:
                task_embedding = self._generate_embedding(task_description, embedding_model)
                task_id = f"{branch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO task_memory 
                        (task_id, repo_path, task_description, branch_name, skill_used, files_modified, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        task_id, repo_str, task_description, branch_name, 
                        skill_used, modified_files, task_embedding
                    ))
            except Exception as e:
                self.logger.warning(f"Failed to store task memory: {e}")
        
        self.conn.commit()
        self.logger.info("Memory update complete")
    
    def find_similar_tasks(self, query: str, embedding_model: Any = None, limit: int = 5) -> List[Dict]:
        """Find similar past tasks."""
        query_embedding = self._generate_embedding(query, embedding_model)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    task_id,
                    task_description,
                    branch_name,
                    skill_used,
                    files_modified,
                    created_at,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM task_memory
                WHERE repo_path = %s
                AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, str(self.repo_path), query_embedding, limit))
            
            return [dict(row) for row in cur.fetchall()]
    
    def _extract_functions(self, content: str, ext: str) -> List[str]:
        """Extract function/class names from source code."""
        functions = []
        
        if ext == '.py':
            for match in re.finditer(r'^(?:async\s+)?def\s+(\w+)|^class\s+(\w+)', content, re.MULTILINE):
                func = match.group(1) or match.group(2)
                if func and not func.startswith('_'):
                    functions.append(func)
        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
            for match in re.finditer(r'(?:function|const|let|var)\s+(\w+)|(\w+)\s*[=:]\s*(?:async\s*)?\(|class\s+(\w+)', content):
                func = match.group(1) or match.group(2) or match.group(3)
                if func:
                    functions.append(func)
        
        return functions[:20]
    
    def _extract_imports(self, content: str, ext: str) -> List[str]:
        """Extract import statements."""
        imports = []
        
        if ext == '.py':
            for match in re.finditer(r'^(?:from|import)\s+(\S+)', content, re.MULTILINE):
                imports.append(match.group(1))
        elif ext in ['.js', '.ts']:
            for match in re.finditer(r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]|require\(['\"]([^'\"]+)['\"]\)", content):
                imp = match.group(1) or match.group(2)
                if imp:
                    imports.append(imp)
        
        return list(set(imports))[:10]
    
    def _generate_summary(self, content: str, path: str) -> str:
        """Generate a simple summary without LLM."""
        lines = content.split('\n')
        
        for line in lines[:10]:
            line = line.strip()
            if line and not line.startswith('import') and not line.startswith('from'):
                for prefix in ['#', '//', '/*', '*', '"""', "'''"]:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                if len(line) > 10:
                    return line[:150]
        
        return f"Source file: {path}"
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")


def get_db_url() -> str:
    """Get database URL from environment or defaults."""
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    database = os.getenv('DB_NAME', 'coding_agent')
    user = os.getenv('DB_USER', 'agent')
    password = os.getenv('DB_PASSWORD', 'agent_password')
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"
