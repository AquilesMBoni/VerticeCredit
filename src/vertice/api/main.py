"""API REST do Vértice Credit Analytics.

    uvicorn vertice.api.main:app --reload --port 8000

Documentação interativa: http://127.0.0.1:8000/docs
Todas as rotas de leitura aceitam `?dataset=us` (padrão) ou `?dataset=br`.
"""
from __future__ import annotations

import logging
import threading
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__, context, services
from ..datasets import DATASETS
from ..pipeline import runner
from .schemas import Applicant, Health, Prediction, PropostaBR, RunResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

Dataset = Literal["us", "br"]
DS = Query("us", description="Base: us (Kaggle, EUA) ou br (Brasil, regras 2026)")

app = FastAPI(
    title="Vértice Credit Analytics API",
    version=__version__,
    description="Pipeline de análise de risco de crédito em duas bases (EUA e Brasil): qualidade de dados, limpeza, "
                "atributos, modelo de Probabilidade de Default e simulação de política.",
    openapi_tags=[
        {"name": "Sistema", "description": "Saúde e metadados"},
        {"name": "Pipeline", "description": "Execução e histórico"},
        {"name": "Dados", "description": "Consultas na base limpa"},
        {"name": "Qualidade", "description": "Contrato de dados e log de limpeza"},
        {"name": "Modelos", "description": "Registro, métricas e predição"},
    ],
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_run_lock = threading.Lock()


def _records(df):
    return df.replace({np.nan: None}).to_dict(orient="records")


def _require_pipeline() -> None:
    if not runner.has_successful_run():
        raise HTTPException(status_code=409, detail="Nenhuma execução concluída para esta base. Chame POST /pipeline/run.")


@app.get("/health", response_model=Health, tags=["Sistema"])
def health() -> Health:
    info = {}
    for ds in DATASETS:
        with context.use(ds):
            run = services.last_run()
            info[ds] = {"has_model": runner.has_successful_run(), "last_run": run["run_id"] if run else None,
                        "name": DATASETS[ds].name}
    return Health(status="ok", version=__version__, datasets=info)


@app.post("/pipeline/run", response_model=RunResult, tags=["Pipeline"])
def run_pipeline(dataset: Dataset = DS) -> RunResult:
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A pipeline já está em execução.")
    try:
        services._load_model.cache_clear()
        return RunResult(**runner.run(dataset=dataset))
    finally:
        _run_lock.release()


@app.get("/pipeline/runs", tags=["Pipeline"])
def list_runs(limit: int = Query(20, ge=1, le=200), dataset: Dataset = DS):
    with context.use(dataset):
        return _records(services.runs(limit))


@app.get("/pipeline/runs/{run_id}/steps", tags=["Pipeline"])
def run_steps(run_id: str, dataset: Dataset = DS):
    with context.use(dataset):
        return _records(services.step_timings(run_id))


@app.get("/overview", tags=["Dados"])
def overview(dataset: Dataset = DS, grades: list[str] | None = Query(None, description="Filtrar por notas")):
    with context.use(dataset):
        _require_pipeline()
        return services.overview(grades)


@app.get("/data/loans", tags=["Dados"])
def loans(page: int = Query(1, ge=1), size: int = Query(50, ge=1, le=500), grade: str | None = None,
          purpose: str | None = None, status: int | None = Query(None, ge=0, le=1), dataset: Dataset = DS):
    with context.use(dataset):
        _require_pipeline()
        out = services.loans(page, size, grade, purpose, status)
        out["items"] = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for r in out["items"]]
        return out


@app.get("/data/segments/{dimension}", tags=["Dados"])
def segments(dimension: str, dataset: Dataset = DS):
    with context.use(dataset):
        _require_pipeline()
        try:
            return _records(services.segment(dimension))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/quality/checks", tags=["Qualidade"])
def quality_checks(stage: str | None = Query(None, pattern="^(bruto|limpo)$"), dataset: Dataset = DS):
    with context.use(dataset):
        _require_pipeline()
        df = services.table("quality_checks")
        return _records(df[df["stage"] == stage] if stage else df)


@app.get("/quality/cleaning-log", tags=["Qualidade"])
def cleaning_log(dataset: Dataset = DS):
    with context.use(dataset):
        _require_pipeline()
        return _records(services.table("cleaning_log"))


@app.get("/features/iv", tags=["Modelos"])
def information_value(dataset: Dataset = DS):
    with context.use(dataset):
        _require_pipeline()
        return _records(services.table("feature_iv"))


@app.get("/models", tags=["Modelos"])
def models(dataset: Dataset = DS):
    with context.use(dataset):
        _require_pipeline()
        return _records(services.table("model_metrics"))


@app.get("/models/champion", tags=["Modelos"])
def champion(dataset: Dataset = DS):
    with context.use(dataset):
        _require_pipeline()
        rec = services.champion_record()
        rec.pop("params", None)
        return rec


@app.get("/models/policy", tags=["Modelos"])
def policy(lgd: float = Query(0.6, ge=0, le=1), dataset: Dataset = DS):
    with context.use(dataset):
        _require_pipeline()
        return _records(services.policy_curve(lgd=lgd))


@app.post("/predict", response_model=Prediction, tags=["Modelos"], summary="Pontuar pedido (base EUA)")
def predict(applicant: Applicant, cutoff: float | None = Query(None, gt=0, lt=1)) -> Prediction:
    with context.use("us"):
        _require_pipeline()
        return Prediction(**services.predict(applicant.model_dump(), cutoff=cutoff))


@app.post("/br/predict", response_model=Prediction, tags=["Modelos"], summary="Pontuar proposta (base Brasil)")
def predict_br(proposta: PropostaBR, cutoff: float | None = Query(None, gt=0, lt=1)) -> Prediction:
    """Rejeita com 422 propostas que violam regras brasileiras (teto INSS, margem consignável, elegibilidade, prazo)."""
    with context.use("br"):
        _require_pipeline()
        return Prediction(**services.predict(proposta.model_dump(), cutoff=cutoff))


@app.get("/predict/log", tags=["Modelos"])
def predict_log(limit: int = Query(50, ge=1, le=500), dataset: Dataset = DS):
    with context.use(dataset):
        return _records(services.prediction_log(limit))
