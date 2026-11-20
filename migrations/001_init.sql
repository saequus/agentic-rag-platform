-- Day 2 schema: documents + chunks in Postgres with a pgvector HNSW index.
-- The migrate runner replaces __EMBEDDING_DIM__ with the model dimension
-- (768 for intfloat/multilingual-e5-base, 1024 for BAAI/bge-m3, 1536 for
-- text-embedding-3-small).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    tenant_id   TEXT        NOT NULL,
    doc_id      TEXT        NOT NULL,
    title       TEXT,
    source_uri  TEXT,
    page_count  INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, doc_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    tenant_id        TEXT        NOT NULL,
    doc_id           TEXT        NOT NULL,
    chunk_no         INTEGER     NOT NULL,
    page             INTEGER,
    section          TEXT,
    content          TEXT        NOT NULL,
    content_hash     TEXT        NOT NULL,
    embedding_model  TEXT        NOT NULL,
    embedding        vector(__EMBEDDING_DIM__) NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, doc_id, chunk_no),
    FOREIGN KEY (tenant_id, doc_id)
        REFERENCES documents (tenant_id, doc_id)
        ON DELETE CASCADE
);

-- Graph index: cosine distance, m=16, ef_construction=64 (Day 2 spec).
-- Query-time ef_search is set per session (default 40 in pgvector).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_tenant_model_idx
    ON chunks (tenant_id, embedding_model);
