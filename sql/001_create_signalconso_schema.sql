-- Schéma PostgreSQL pour le projet SignalConso
-- Table de staging

CREATE TABLE IF NOT EXISTS signalconso_raw_staging (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT,
    payload JSONB NOT NULL,
    source_file TEXT,
    ingested_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Table de données brutes

CREATE TABLE IF NOT EXISTS raw_complaints (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT UNIQUE,
    submitted_at TIMESTAMP,
    theme TEXT,
    sub_theme TEXT,
    company_name TEXT,
    region TEXT,
    department TEXT,
    channel TEXT,
    status TEXT,
    complaint_text TEXT,
    raw_payload JSONB NOT NULL,
    ingested_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Table de données nettoyées

CREATE TABLE IF NOT EXISTS clean_complaints (
    id BIGSERIAL PRIMARY KEY,
    raw_complaint_id BIGINT NOT NULL REFERENCES raw_complaints(id) ON DELETE CASCADE,
    clean_text TEXT NOT NULL,
    language VARCHAR(20),
    token_count INTEGER,
    is_valid BOOLEAN NOT NULL DEFAULT TRUE,
    cleaned_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_clean_complaints_raw UNIQUE (raw_complaint_id)
);

-- Tables de sortie ML
-- Table de prédictions

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    clean_complaint_id BIGINT NOT NULL REFERENCES clean_complaints(id) ON DELETE CASCADE,
    predicted_category VARCHAR(255) NOT NULL,
    predicted_priority VARCHAR(50) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    predicted_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGSERIAL PRIMARY KEY,
    input_text TEXT NOT NULL,
    clean_text TEXT NOT NULL,
    predicted_category VARCHAR(255) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);
-- Table de suivi des exécutions de modèles

CREATE TABLE IF NOT EXISTS model_runs (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    dataset_version VARCHAR(50),
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifact_path VARCHAR(500),
    trained_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);