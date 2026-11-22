"""Camada de serviços: consultas analíticas e inferência.

Usada tanto pela API (FastAPI) quanto pelo dashboard (Streamlit), garantindo
uma única fonte de verdade. Tudo opera sobre a base ativa (`vertice.context`).
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import context, db
from .pipeline.modeling import MODEL_LABELS, evaluate
from .pipeline.monitoring import psi_table

ALLOWED_TABLES = {"raw_loans", "clean_loans", "quality_checks", "quality_dimensions", "cleaning_log", "feature_iv",
                  "feature_woe", "feature_psi", "feature_importance", "model_metrics", "test_predictions"}


# ----------------------------------------------------------------- leitura
def last_run() -> dict | None:
    if not db.table_exists("pipeline_runs"):
        return None
    df = db.read_sql("SELECT * FROM pipeline_runs WHERE status = 'sucesso' ORDER BY started_at DESC LIMIT 1")
    return None if df.empty else df.iloc[0].replace({np.nan: None}).to_dict()


def runs(limit: int = 50) -> pd.DataFrame:
    return db.read_sql("SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT :n", n=limit)


def step_timings(run_id: str | None = None) -> pd.DataFrame:
    if run_id:
        return db.read_sql("SELECT * FROM step_timings WHERE run_id = :r", r=run_id)
    return db.read_sql("SELECT * FROM step_timings")


def table(name: str) -> pd.DataFrame:
    if name not in ALLOWED_TABLES:
        raise ValueError(f"Tabela não permitida: {name}")
    return db.read_sql(f"SELECT * FROM {name}")  # noqa: S608 - nome validado na allowlist


def filter_grades(df: pd.DataFrame, grades: Iterable[str] | None, col: str | None = None) -> pd.DataFrame:
    """Filtra por nota/rating. Lista vazia ou None = todas. Tolera texto sujo do bruto ('b ', 'aa')."""
    grades = list(grades or [])
    if not grades:
        return df
    col = col or context.spec().grade
    return df[df[col].astype(str).str.strip().str.upper().isin(grades)]


def metrics_for(preds: pd.DataFrame) -> pd.DataFrame:
    """Métricas de teste de cada candidato, recalculadas para o recorte recebido."""
    rows = []
    y = preds["y_true"].to_numpy()
    for col in [c for c in preds.columns if c.startswith("p_")]:
        name = col[2:]
        rows.append({"model": name, "label": MODEL_LABELS.get(name, name), **evaluate(y, preds[col].to_numpy())})
    return pd.DataFrame(rows)


def overview(grades: Iterable[str] | None = None) -> dict:
    spec = context.spec()
    clean = filter_grades(table("clean_loans"), grades)
    preds = filter_grades(table("test_predictions"), grades)
    metrics = table("model_metrics")
    champ = metrics.query("is_champion == 1").iloc[0]
    m = metrics_for(preds).set_index("model").loc[champ["model"]] if len(preds) else champ
    return {
        "dataset": spec.id,
        "run": last_run(),
        "rows_raw": int(len(filter_grades(table("raw_loans"), grades))),
        "rows_clean": int(len(clean)),
        "default_rate": float(clean[spec.target].mean()) if len(clean) else 0.0,
        "total_volume": float(clean[spec.amount].sum()),
        "avg_rate": float(clean[spec.rate].mean()) if len(clean) else 0.0,
        "avg_income": float(clean[spec.income].mean()) if len(clean) else 0.0,
        "champion": {"model": champ["model"], "label": champ["label"],
                     **{k: (None if pd.isna(m[k]) else float(m[k])) for k in ("roc_auc", "gini", "ks", "pr_auc", "brier")}},
    }


def loans(page: int = 1, size: int = 50, grade: str | None = None, purpose: str | None = None,
          status: int | None = None) -> dict:
    spec = context.spec()
    where, params = [], {}
    if grade:
        where.append(f"{spec.grade} = :grade")
        params["grade"] = grade
    if purpose:
        where.append(f"{spec.purpose} = :purpose")
        params["purpose"] = purpose
    if status is not None:
        where.append(f"{spec.target} = :status")
        params["status"] = status
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    total = int(db.read_sql(f"SELECT COUNT(*) n FROM clean_loans {clause}", **params)["n"].iloc[0])
    params |= {"lim": size, "off": (page - 1) * size}
    rows = db.read_sql(f"SELECT * FROM clean_loans {clause} ORDER BY record_id LIMIT :lim OFFSET :off", **params)
    return {"total": total, "page": page, "size": size, "items": rows.to_dict(orient="records")}


def segment(dimension: str) -> pd.DataFrame:
    spec = context.spec()
    if dimension not in spec.segment_dims:
        raise ValueError(f"Dimensão inválida. Use uma de: {list(spec.segment_dims)}")
    df = db.read_sql(
        f"SELECT {dimension} AS segment, COUNT(*) AS n, AVG({spec.target}) AS default_rate, "
        f"AVG({spec.rate}) AS avg_rate, AVG({spec.amount}) AS avg_amount FROM clean_loans GROUP BY {dimension}")
    order = spec.dim_orders.get(dimension)
    if order:
        df["_o"] = df["segment"].map({v: i for i, v in enumerate(order)})
        df = df.sort_values("_o").drop(columns="_o")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------- política
def policy_curve(lgd: float = 0.6, model: str | None = None, preds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Simula "aprovar se PD < corte" no conjunto de teste (ou no recorte recebido)."""
    spec = context.spec()
    preds = table("test_predictions") if preds is None else preds
    model = model or champion_name()
    p = preds[f"p_{model}"].to_numpy()
    y = preds["y_true"].to_numpy()
    amt = preds[spec.amount].to_numpy()
    revenue, loss = spec.revenue_and_loss(preds, lgd)
    n_bad, n_good = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    rows = []
    for cut in np.round(np.arange(0.02, 1.0001, 0.01), 2):
        approved = p < cut
        n_app = approved.sum()
        good_app, bad_app = approved & (y == 0), approved & (y == 1)
        rows.append({
            "cutoff": float(cut),
            "approval_rate": float(n_app / max(len(p), 1)),
            "bad_rate_approved": float(bad_app.sum() / n_app) if n_app else 0.0,
            "bads_rejected_pct": float(((~approved) & (y == 1)).sum() / n_bad),
            "goods_rejected_pct": float(((~approved) & (y == 0)).sum() / n_good),
            "profit": float(revenue[good_app].sum() - loss[bad_app].sum()),
            "volume_approved": float(amt[approved].sum()),
        })
    return pd.DataFrame(rows)


def psi_for(grades: Iterable[str] | None = None) -> pd.DataFrame:
    spec = context.spec()
    clean = filter_grades(table("clean_loans"), grades)
    test_ids = set(table("test_predictions")["record_id"])
    is_test = clean["record_id"].isin(test_ids)
    if is_test.sum() < 50 or (~is_test).sum() < 50:
        return pd.DataFrame(columns=["feature", "psi", "status"])
    return psi_table(clean[~is_test], clean[is_test], [*spec.numeric_features, *spec.categorical_features])


# ----------------------------------------------------------------- inferência
def champion_record() -> dict:
    df = db.read_sql("SELECT * FROM model_registry WHERE is_champion = 1 ORDER BY version DESC LIMIT 1")
    if df.empty:
        raise RuntimeError("Nenhum modelo campeão registrado. Execute a pipeline.")
    rec = df.iloc[0].to_dict()
    rec["metrics"] = json.loads(rec["metrics"])
    return rec


def champion_name() -> str:
    return str(champion_record()["name"])


@lru_cache(maxsize=8)
def _load_model(path: str) -> dict:
    return joblib.load(Path(path))


def risk_band(pd_value: float) -> str:
    if pd_value < 0.10:
        return "Baixo"
    if pd_value < 0.30:
        return "Moderado"
    if pd_value < 0.50:
        return "Alto"
    return "Muito alto"


def predict(payload: dict, cutoff: float | None = None, log: bool = True) -> dict:
    spec = context.spec()
    cutoff = spec.default_cutoff if cutoff is None else cutoff
    rec = champion_record()
    bundle = _load_model(rec["path"])
    X = spec.prepare_payload(payload)[bundle["features"]]
    proba = float(bundle["pipeline"].predict_proba(X)[:, 1][0])
    decision = "Aprovar" if proba < cutoff else "Recusar"
    result = {"probability_default": round(proba, 4), "risk_band": risk_band(proba), "decision": decision,
              "cutoff": cutoff, "model_id": rec["model_id"], "model": MODEL_LABELS.get(rec["name"], rec["name"]),
              "dataset": spec.id}
    if log:
        db.execute("INSERT INTO api_prediction_log (created_at, model_id, payload, probability, decision) "
                   "VALUES (:t, :m, :p, :pr, :d)", t=datetime.now(UTC).isoformat(timespec="seconds"),
                   m=rec["model_id"], p=db.to_json(payload), pr=proba, d=decision)
    return result


def prediction_log(limit: int = 100) -> pd.DataFrame:
    return db.read_sql("SELECT * FROM api_prediction_log ORDER BY id DESC LIMIT :n", n=limit)
