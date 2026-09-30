"""
Database layer (PostgreSQL through SQLAlchemy).

Two tables are used:
- heart_disease_raw : the original UCI dataset, loaded once by train.py --to-db
- predictions       : every prediction made in the Streamlit app

Connection:
    Set an environment variable called DATABASE_URL, for example
        postgresql+psycopg2://postgres:yourpassword@localhost:5432/heart_disease
    If DATABASE_URL is not set, the project falls back to a local SQLite file
    (heart_app.db) so that everything still runs without PostgreSQL installed.
"""
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (
    Column, DateTime, Float, Integer, MetaData, String, Table, create_engine,
    inspect, select,
)

from . import config

SQLITE_FALLBACK_URL = f"sqlite:///{(config.ROOT_DIR / 'heart_app.db').as_posix()}"
RAW_TABLE = "heart_disease_raw"

metadata = MetaData()

predictions_table = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("patient_ref", String(100)),
    Column("age", Integer),
    Column("sex", String(10)),
    Column("cp", String(30)),
    Column("trestbps", Float),
    Column("chol", Float),
    Column("fbs", String(5)),
    Column("restecg", String(30)),
    Column("thalch", Float),
    Column("exang", String(5)),
    Column("oldpeak", Float),
    Column("slope", String(20)),
    Column("model_name", String(40), nullable=False),
    Column("probability", Float, nullable=False),
    Column("prediction", Integer, nullable=False),
)

_engine = None


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", SQLITE_FALLBACK_URL)


def get_engine():
    """Create the database connection once and reuse it."""
    global _engine
    if _engine is None:
        _engine = create_engine(get_database_url(), pool_pre_ping=True, future=True)
    return _engine


def backend_name() -> str:
    return "PostgreSQL" if get_engine().dialect.name == "postgresql" else "SQLite (fallback)"


def init_db():
    """Create the predictions table if it does not exist yet."""
    metadata.create_all(get_engine())


def load_dataset_to_db(raw_df: pd.DataFrame, table_name: str = RAW_TABLE) -> int:
    """Store the raw UCI dataset in the database (replaces the table if it exists)."""
    df = raw_df.copy()
    # Store True/False columns as text so every database handles them the same way.
    for col in ["fbs", "exang"]:
        df[col] = df[col].map({True: "TRUE", False: "FALSE"}).where(df[col].notna(), None)
    df.to_sql(table_name, get_engine(), if_exists="replace", index=False)
    return len(df)


def read_dataset_from_db(table_name: str = RAW_TABLE) -> pd.DataFrame:
    """Read the raw dataset back from the database."""
    df = pd.read_sql_table(table_name, get_engine())
    for col in ["fbs", "exang"]:
        df[col] = df[col].map({"TRUE": True, "FALSE": False})
    return df


def raw_table_exists(table_name: str = RAW_TABLE) -> bool:
    return inspect(get_engine()).has_table(table_name)


def save_prediction(patient: dict, model_name: str, probability: float,
                    prediction: int, patient_ref: str = "") -> None:
    """Insert one prediction made in the app."""
    row = {k: patient.get(k) for k in config.ALL_FEATURES}
    row.update({
        "created_at": datetime.now(timezone.utc),
        "patient_ref": patient_ref or None,
        "model_name": model_name,
        "probability": float(probability),
        "prediction": int(prediction),
    })
    with get_engine().begin() as conn:
        conn.execute(predictions_table.insert().values(**row))


def fetch_predictions(limit: int = 200) -> pd.DataFrame:
    """Most recent predictions first."""
    query = (select(predictions_table)
             .order_by(predictions_table.c.created_at.desc())
             .limit(limit))
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)
