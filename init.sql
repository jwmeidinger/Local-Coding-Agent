-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table for code file embeddings
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
    embedding VECTOR(1536),  -- OpenAI text-embedding-3-small dimensions
    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(repo_path, file_path)
);

-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS code_embeddings_hnsw_idx 
ON code_embeddings USING hnsw (embedding vector_cosine_ops);

-- Create index for file type filtering
CREATE INDEX IF NOT EXISTS code_embeddings_type_idx ON code_embeddings(file_type);

-- Create index for repo filtering
CREATE INDEX IF NOT EXISTS code_embeddings_repo_idx ON code_embeddings(repo_path);

-- Create table for conversation/task memory
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

-- Create index for task similarity search
CREATE INDEX IF NOT EXISTS task_memory_hnsw_idx 
ON task_memory USING hnsw (embedding vector_cosine_ops);

-- Create full-text search index for hybrid search
CREATE INDEX IF NOT EXISTS code_embeddings_fts_idx 
ON code_embeddings USING gin(to_tsvector('english', COALESCE(summary, '') || ' ' || COALESCE(content_preview, '')));

-- Create function for hybrid search (vector + keyword)
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding VECTOR(1536),
    query_text TEXT,
    repo_path_filter TEXT,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    file_path TEXT,
    summary TEXT,
    file_type TEXT,
    key_functions TEXT[],
    vector_similarity FLOAT,
    text_rank FLOAT,
    combined_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            ce.file_path,
            ce.summary,
            ce.file_type,
            ce.key_functions,
            1 - (ce.embedding <=> query_embedding) AS vector_similarity,
            ts_rank(
                to_tsvector('english', COALESCE(ce.summary, '') || ' ' || COALESCE(ce.content_preview, '')),
                plainto_tsquery('english', query_text)
            ) AS text_rank
        FROM code_embeddings ce
        WHERE ce.repo_path = repo_path_filter
        AND ce.embedding IS NOT NULL
    ),
    ranked AS (
        SELECT 
            vr.*,
            -- Combine scores: 70% vector similarity, 30% text rank (normalized)
            (0.7 * vr.vector_similarity + 0.3 * COALESCE(vr.text_rank / NULLIF((SELECT MAX(text_rank) FROM vector_results), 0), 0)) AS combined_score
        FROM vector_results vr
    )
    SELECT 
        ranked.file_path,
        ranked.summary,
        ranked.file_type,
        ranked.key_functions,
        ranked.vector_similarity,
        ranked.text_rank,
        ranked.combined_score
    FROM ranked
    ORDER BY ranked.combined_score DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get codebase stats
CREATE OR REPLACE FUNCTION get_codebase_stats(repo_path_filter TEXT)
RETURNS TABLE (
    total_files BIGINT,
    source_files BIGINT,
    test_files BIGINT,
    config_files BIGINT,
    last_indexed TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_files,
        COUNT(*) FILTER (WHERE file_type = 'source')::BIGINT as source_files,
        COUNT(*) FILTER (WHERE file_type = 'test')::BIGINT as test_files,
        COUNT(*) FILTER (WHERE file_type = 'config')::BIGINT as config_files,
        MAX(indexed_at) as last_indexed
    FROM code_embeddings
    WHERE repo_path = repo_path_filter;
END;
$$ LANGUAGE plpgsql;