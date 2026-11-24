-- Day 3: lexical leg of hybrid search on Postgres.
-- 'simple' = lowercase + split, no stemming and no stopwords. The corpus mixes
-- English (ECB) and Polish (treasury bonds) and Postgres has no Polish
-- dictionary, so a language-specific config would only help half the corpus.
-- Stopwords are removed from the *query* in Python (retrieval/sparse.py).
-- Adding a STORED generated column rewrites the table once (seconds at 10k rows).

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin
    ON chunks USING gin (content_tsv);
