-- PostgreSQL setup for the Heart Disease Prediction project.
-- Run once in psql or pgAdmin:
--     CREATE DATABASE heart_disease;
-- then connect to heart_disease and run the rest of this file.
--
-- Note: the Python code creates these tables automatically as well
-- (train.py --to-db and the Streamlit app), so this file is mainly for reference.

-- Raw UCI dataset (filled by: python train.py --to-db)
CREATE TABLE IF NOT EXISTS heart_disease_raw (
    id        INTEGER PRIMARY KEY,
    age       INTEGER,
    sex       TEXT,
    dataset   TEXT,       -- source hospital
    cp        TEXT,       -- chest pain type
    trestbps  DOUBLE PRECISION,
    chol      DOUBLE PRECISION,
    fbs       TEXT,       -- 'TRUE' / 'FALSE'
    restecg   TEXT,
    thalch    DOUBLE PRECISION,
    exang     TEXT,       -- 'TRUE' / 'FALSE'
    oldpeak   DOUBLE PRECISION,
    slope     TEXT,
    ca        DOUBLE PRECISION,
    thal      TEXT,
    num       INTEGER     -- 0 = no disease, 1-4 = disease
);

-- Every prediction made in the Streamlit app
CREATE TABLE IF NOT EXISTS predictions (
    id           SERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    patient_ref  VARCHAR(100),
    age          INTEGER,
    sex          VARCHAR(10),
    cp           VARCHAR(30),
    trestbps     DOUBLE PRECISION,
    chol         DOUBLE PRECISION,
    fbs          VARCHAR(5),
    restecg      VARCHAR(30),
    thalch       DOUBLE PRECISION,
    exang        VARCHAR(5),
    oldpeak      DOUBLE PRECISION,
    slope        VARCHAR(20),
    model_name   VARCHAR(40) NOT NULL,
    probability  DOUBLE PRECISION NOT NULL,
    prediction   INTEGER NOT NULL
);

-- Handy queries
-- SELECT dataset, COUNT(*), AVG((num > 0)::int) AS disease_rate FROM heart_disease_raw GROUP BY dataset;
-- SELECT model_name, COUNT(*), AVG(probability) FROM predictions GROUP BY model_name;
