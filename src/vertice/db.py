"""Camada de persistência em SQLite (SQLAlchemy 2.x).

Cada base (EUA, Brasil) tem o próprio arquivo SQLite.

Tabelas de dados (substituídas a cada execução da pipeline):
    raw_loans, clean_loans, test_predictions, feature_iv, feature_psi,
    quality_checks, cleaning_log, model_metrics, feature_importance

Tabelas de metadados (acumulativas, histórico):
    pipeline_runs, step_timings, model_registry, api_prediction_log
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from . import context
from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id         TEXT PRIMARY KEY,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    status         TEXT NOT NULL,
    duration_s     REAL,
    dataset_sha256 TEXT,
    rows_raw       INTEGER,
    rows_clean     INTEGER,
    quality_raw    REAL,
    quality_score  REAL,
    champion_model TEXT,
    champion_auc   REAL,
    error          TEXT
);
CREATE TABLE IF NOT EXISTS step_timings (
    run_id     TEXT NOT NULL,
    step       TEXT NOT NULL,
    seconds    REAL NOT NULL,
    rows_out   INTEGER,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);
CREATE TABLE IF NOT EXISTS model_registry (
    model_id    TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    name        TEXT NOT NULL,
    version     INTEGER NOT NULL,
    path        TEXT NOT NULL,
    metrics     TEXT NOT NULL,
    params      TEXT NOT NULL,
    is_champion INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_prediction_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    model_id    TEXT,
    payload     TEXT NOT NULL,
    probability REAL NOT NULL,
    decision    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_step_timings_run ON step_timings(run_id);
"""


def get_engine() -> Engine:
    """Engine do SQLite da base ativa (um arquivo por base: data/vertice_<id>.db)."""
    return _engine(str(settings.db_path(context.current_id())))


@lru_cache(maxsize=8)
def _engine(path: str) -> Engine:
    engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - infraestrutura
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")  # leitura concorrente (API + dashboard)
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    with engine.begin() as conn:
        for stmt in filter(None, (s.strip() for s in SCHEMA.split(";"))):
            conn.execute(text(stmt))
    return engine


@contextmanager
def connection() -> Iterator[Any]:
    with get_engine().begin() as conn:
        yield conn


def write_df(df: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    df.to_sql(table, get_engine(), if_exists=if_exists, index=False, chunksize=5000)


def read_sql(query: str, **params: Any) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def table_exists(table: str) -> bool:
    df = read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name=:t", t=table)
    return not df.empty


def execute(query: str, **params: Any) -> None:
    with connection() as conn:
        conn.execute(text(query), params)


def to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=float)
