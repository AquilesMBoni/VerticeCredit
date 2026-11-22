"""Acesso a dados do dashboard: cache do Streamlit, recortes por nota e cliente da API.

Todo cache recebe a base e o id da última execução como argumento: trocar de
base ou reexecutar a pipeline invalida os resultados automaticamente.
"""
from __future__ import annotations

import time

import httpx
import pandas as pd
import streamlit as st

from vertice import context, services
from vertice.config import settings
from vertice.pipeline import cleaning, features, runner, validation


def ds() -> str:
    return context.current_id()


def run_key() -> str:
    run = services.last_run()
    return f"{ds()}:{run['run_id'] if run else 'none'}"


def _g(grades) -> tuple:
    return tuple(sorted(grades or ()))


@st.cache_data(show_spinner=False, max_entries=64)
def _table(name: str, key: str) -> pd.DataFrame:
    return services.table(name)


def t(name: str) -> pd.DataFrame:
    return _table(name, run_key())


def tg(name: str, grades, col: str | None = None) -> pd.DataFrame:
    """Tabela já filtrada pelas notas escolhidas."""
    return services.filter_grades(t(name), grades, col)


@st.cache_data(show_spinner=False, max_entries=32)
def _overview(grades: tuple, key: str) -> dict:
    return services.overview(grades)


def overview(grades=()) -> dict:
    return _overview(_g(grades), run_key())


@st.cache_data(show_spinner=False, max_entries=64)
def _policy(lgd: float, grades: tuple, key: str) -> pd.DataFrame:
    return services.policy_curve(lgd=lgd, preds=services.filter_grades(services.table("test_predictions"), grades))


def policy(lgd: float, grades=()) -> pd.DataFrame:
    return _policy(lgd, _g(grades), run_key())


@st.cache_data(show_spinner=False, max_entries=32)
def _quality(grades: tuple, key: str) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Contrato de qualidade reexecutado no recorte (bruto e limpo)."""
    raw = services.filter_grades(services.table("raw_loans").drop(columns="record_id"), grades)
    clean = services.filter_grades(services.table("clean_loans"), grades)
    rep_raw, s_raw = validation.run_quality_checks(raw, "bruto")
    rep_clean, s_clean = validation.run_quality_checks(clean, "limpo")
    dims = pd.concat([validation.dimension_scores(rep_raw, len(raw)), validation.dimension_scores(rep_clean, len(clean))])
    return pd.concat([rep_raw, rep_clean], ignore_index=True), dims, s_raw, s_clean


def quality(grades=()):
    if not grades:
        run = services.last_run()
        return t("quality_checks"), t("quality_dimensions"), float(run["quality_raw"]), float(run["quality_score"])
    return _quality(_g(grades), run_key())


@st.cache_data(show_spinner=False, max_entries=32)
def _cleaning_log(grades: tuple, key: str) -> pd.DataFrame:
    raw = services.filter_grades(services.table("raw_loans"), grades)
    return cleaning.clean(raw)[1]


def cleaning_log(grades=()) -> pd.DataFrame:
    return t("cleaning_log") if not grades else _cleaning_log(_g(grades), run_key())


@st.cache_data(show_spinner=False, max_entries=32)
def _iv(grades: tuple, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return features.woe_iv(services.filter_grades(services.table("clean_loans"), grades))


def iv(grades=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (t("feature_woe"), t("feature_iv")) if not grades else _iv(_g(grades), run_key())


@st.cache_data(show_spinner=False, max_entries=32)
def _psi(grades: tuple, key: str) -> pd.DataFrame:
    return services.psi_for(grades)


def psi(grades=()) -> pd.DataFrame:
    return t("feature_psi") if not grades else _psi(_g(grades), run_key())


# ------------------------------------------------------------------ API
@st.cache_data(ttl=10, show_spinner=False)
def api_health() -> dict | None:
    try:
        r = httpx.get(f"{settings.api_url}/health", timeout=1.2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def predict(payload: dict, cutoff: float) -> tuple[dict, str, float]:
    """Tenta a API; se estiver offline ou sem esta base, usa o mesmo serviço localmente."""
    t0 = time.perf_counter()
    health = api_health()
    if health and health.get("datasets", {}).get(ds(), {}).get("has_model"):
        try:
            r = httpx.post(f"{settings.api_url}{context.spec().predict_route}", json=payload,
                           params={"cutoff": cutoff}, timeout=5)
            if r.status_code == 422:
                detail = r.json().get("detail")
                msg = detail[0]["msg"] if isinstance(detail, list) else str(detail)
                return {"error": msg.replace("Value error, ", "")}, "API REST", (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            return r.json(), "API REST", (time.perf_counter() - t0) * 1000
        except Exception:
            pass
    if context.current_id() == "br":  # mesmas regras da API, sem depender dela
        from pydantic import ValidationError

        from vertice.api.schemas import PropostaBR
        try:
            payload = PropostaBR(**payload).model_dump()
        except ValidationError as exc:
            return {"error": exc.errors()[0]["msg"].replace("Value error, ", "")}, "serviço local", 0.0
    return services.predict(payload, cutoff=cutoff), "serviço local", (time.perf_counter() - t0) * 1000


def ensure_pipeline() -> None:
    if runner.has_successful_run():
        return
    with st.container(key="pipeline_progress"):
        with st.status(f"Primeira execução da base {context.spec().short}: rodando a pipeline completa…",
                       expanded=True) as status:
            bar = st.progress(0.0)

            def progress(step: str, i: int, n: int) -> None:
                bar.progress(i / n, text=f"{step} ({i}/{n})")

            runner.run(progress)
            status.update(label="Pipeline concluída", state="complete", expanded=False)
    st.cache_data.clear()
