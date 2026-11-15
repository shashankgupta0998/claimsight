CREATE EXTENSION IF NOT EXISTS vector;

-- Policy corpus -------------------------------------------------------------

CREATE TABLE policy_docs (
    doc_id      text PRIMARY KEY,
    title       text NOT NULL,
    source_url  text,
    ingested_at timestamptz NOT NULL DEFAULT now()
);

-- Text lane: one row per clause. Dense (bge-m3, 1024-d) + sparse (tsvector) for hybrid search.
CREATE TABLE policy_chunks (
    clause_id text PRIMARY KEY,
    doc_id    text NOT NULL REFERENCES policy_docs,
    page      int,
    text      text NOT NULL,
    metadata  jsonb NOT NULL DEFAULT '{}',
    embedding vector(1024),
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX policy_chunks_embedding_idx ON policy_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX policy_chunks_tsv_idx ON policy_chunks USING gin (tsv);

-- Visual lane: ColQwen2 multi-vector page embeddings, one row per page patch (128-d).
CREATE TABLE policy_page_patches (
    doc_id    text NOT NULL REFERENCES policy_docs,
    page      int  NOT NULL,
    patch     int  NOT NULL,
    embedding vector(128) NOT NULL,
    PRIMARY KEY (doc_id, page, patch)
);
CREATE INDEX policy_page_patches_embedding_idx
    ON policy_page_patches USING hnsw (embedding vector_cosine_ops);

-- Policies and claims -------------------------------------------------------

CREATE TABLE policies (
    policy_id   text PRIMARY KEY,
    coverages   text[] NOT NULL,
    deductibles jsonb NOT NULL,
    vin         text
);

CREATE TABLE claims (
    claim_id   text PRIMARY KEY,
    policy_id  text NOT NULL REFERENCES policies,
    payload    jsonb NOT NULL,
    status     text NOT NULL DEFAULT 'received',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE decision_packets (
    claim_id       text PRIMARY KEY REFERENCES claims,
    packet         jsonb NOT NULL,
    decided_by     text NOT NULL,
    model_versions jsonb NOT NULL DEFAULT '{}',
    created_at     timestamptz NOT NULL DEFAULT now()
);

-- Fraud: perceptual hash for exact-ish reuse, DINOv2 embedding for near-duplicates.
CREATE TABLE photo_hashes (
    claim_id       text NOT NULL,
    photo_id       text NOT NULL,
    phash          bit(64) NOT NULL,
    dino_embedding vector(768),
    PRIMARY KEY (claim_id, photo_id)
);
CREATE INDEX photo_hashes_dino_idx ON photo_hashes USING hnsw (dino_embedding vector_cosine_ops);
