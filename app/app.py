"""Vértice Credit Analytics — dashboard Streamlit.

    streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Vértice Credit Analytics", page_icon=":material/insights:",
                   layout="wide", initial_sidebar_state="expanded")

import data  # noqa: E402
import ui  # noqa: E402
from vertice import context, services  # noqa: E402
from vertice.datasets import DATASETS  # noqa: E402
from vertice.pipeline import runner  # noqa: E402
from views import (  # noqa: E402
    api_view,
    cleaning_view,
    eda_view,
    features_view,
    ingestion_view,
    modeling_view,
    monitoring_view,
    overview_view,
    quality_view,
    results_view,
)

# ordem = trilho da sidebar (ícones correspondentes em ui.NAV_ICONS)
PAGES = {
    "Visão geral": overview_view,
    "Ingestão": ingestion_view,
    "Qualidade": quality_view,
    "Limpeza": cleaning_view,
    "Atributos": features_view,
    "Exploração": eda_view,
    "Modelagem": modeling_view,
    "Resultados": results_view,
    "API e serviço": api_view,
    "Monitoramento": monitoring_view,
}
DATASET_LABELS = {"us": "EUA", "br": "Brasil"}

# ---------------------------------------------------------------- estado inicial (URL → sessão)
qp = st.query_params
if "theme_ctl" not in st.session_state:
    st.session_state.theme_ctl = qp.get("theme") if qp.get("theme") in ui.THEME_OPTIONS else "auto"
if st.session_state.theme_ctl is None:  # clicar no botão ativo desmarca: mantém o anterior
    st.session_state.theme_ctl = st.session_state.get("theme_last", "auto")
st.session_state.theme_last = theme = st.session_state.theme_ctl

if "dataset_ctl" not in st.session_state:
    st.session_state.dataset_ctl = qp.get("dataset") if qp.get("dataset") in DATASETS else "us"
if st.session_state.dataset_ctl is None:
    st.session_state.dataset_ctl = st.session_state.get("dataset_last", "us")
st.session_state.dataset_last = dataset = st.session_state.dataset_ctl
context.set_dataset(dataset)

if "page" not in st.session_state:
    st.session_state.page = qp.get("page") if qp.get("page") in PAGES else "Visão geral"

ui.inject_css(theme)

with st.sidebar:
    st.markdown("""<div class="vx-brand"><div class="mark"><i class="fa-solid fa-chart-line"></i></div>
<div><div class="name">Vértice Credit</div><div class="sub">Análise de risco de crédito</div></div></div>""",
                unsafe_allow_html=True)

    st.markdown('<div class="vx-side-label">Base de dados</div>', unsafe_allow_html=True)
    with st.container(key="dataset"):
        st.segmented_control("Base", list(DATASETS), format_func=DATASET_LABELS.get, key="dataset_ctl",
                             label_visibility="collapsed", width="stretch")
    spec = context.spec()
    st.markdown(f'<div class="vx-note" style="margin-top:.35rem;font-size:.78rem">{ui.esc(spec.name)}</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="vx-side-label">Pipeline</div>', unsafe_allow_html=True)
    with st.container(key="nav"):
        page = st.radio("Fase da pipeline", list(PAGES), key="page", label_visibility="collapsed")

    st.markdown('<div class="vx-side-label">Tema</div>', unsafe_allow_html=True)
    with st.container(key="theme"):
        st.segmented_control("Tema", list(ui.THEME_OPTIONS), format_func=ui.THEME_OPTIONS.get,
                             key="theme_ctl", label_visibility="collapsed", width="stretch")

    st.markdown('<div class="vx-side-label">Status</div>', unsafe_allow_html=True)
    health = data.api_health()
    api_pill = ui.pill("good", "API online", "fa-plug-circle-check") if health else \
        ui.pill("warn", "API offline · modo local", "fa-plug-circle-xmark")
    run = services.last_run()
    run_txt = (f'<div class="vx-note" style="margin-top:.5rem"><i class="fa-regular fa-clock"></i> '
               f'Última execução ({spec.short})<br><b>{ui.esc(run["run_id"])}</b> · {ui.fmt_num(run["duration_s"] or 0, 1)} s</div>'
               if run else "")
    st.markdown(f"{api_pill}{run_txt}", unsafe_allow_html=True)
    if st.button("Reexecutar pipeline", icon=":material/replay:", width="stretch"):
        with st.spinner(f"Executando as 8 etapas na base {spec.short}…"):
            services._load_model.cache_clear()
            runner.run()
        st.cache_data.clear()
        st.toast("Pipeline executada. Painéis atualizados.", icon=":material/check_circle:")
        st.rerun()

st.query_params.update({"page": page, "theme": theme, "dataset": dataset})

data.ensure_pipeline()
mode = ui.resolve_mode(theme)
PAGES[page].render(mode)
