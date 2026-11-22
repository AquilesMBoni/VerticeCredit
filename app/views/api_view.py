from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import data
import ui
from vertice import context, services
from vertice.config import settings

ENDPOINTS = [
    ("GET", "/health", "Saúde da API, banco e modelo"),
    ("POST", "/pipeline/run", "Executa a pipeline completa"),
    ("GET", "/pipeline/runs", "Histórico de execuções"),
    ("GET", "/pipeline/runs/{run_id}/steps", "Tempo de cada etapa"),
    ("GET", "/overview", "Indicadores gerais da base limpa"),
    ("GET", "/data/loans", "Pedidos paginados com filtros (grade, intent, status)"),
    ("GET", "/data/segments/{dimension}", "Default por segmento"),
    ("GET", "/quality/checks", "Resultado do contrato de dados"),
    ("GET", "/quality/cleaning-log", "Log auditável da limpeza"),
    ("GET", "/features/iv", "Information Value por variável"),
    ("GET", "/models", "Métricas de todos os candidatos"),
    ("GET", "/models/champion", "Modelo campeão no registro"),
    ("GET", "/models/policy", "Curva da política de crédito (parâmetro lgd)"),
    ("POST", "/predict", "Pontua um pedido da base EUA"),
    ("POST", "/br/predict", "Pontua uma proposta brasileira (valida as regras do BCB/INSS/CLT)"),
    ("GET", "/predict/log", "Últimas pontuações"),
]



def render(mode: str) -> None:
    spec = context.spec()
    health = data.api_health()
    ds_info = (health or {}).get("datasets", {}).get(spec.id, {})
    ui.page_header("fa-plug", "Fase 8 de 8", "API e serviço",
                   "O modelo campeão de cada base é servido por uma API <b>FastAPI</b> com contratos <b>Pydantic</b>. "
                   "Dashboard e API usam a mesma camada de serviços e o mesmo SQLite, então os números são sempre iguais. "
                   "Rotas de leitura recebem <code>?dataset=us</code> ou <code>?dataset=br</code>.")

    status = ui.pill("good", "online", "fa-circle-check") if health else ui.pill("warn", "offline", "fa-circle-pause")
    ui.kpi_row([
        ui.kpi("fa-server", "Status da API", status, settings.api_url),
        ui.kpi("fa-code-branch", "Versão", health["version"] if health else "—", "vertice.api.main:app"),
        ui.kpi("fa-cube", f"Modelo da base {spec.short}", "pronto" if ds_info.get("has_model") else "—", "lido do registro no SQLite"),
        ui.kpi("fa-book-open", "Documentação", f'<a href="{settings.api_url}/docs" target="_blank" style="font-size:1.1rem">/docs</a>',
               "Swagger UI interativo"),
    ])
    if not health:
        st.write("")
        st.info("A API não está rodando. Inicie em outro terminal com "
                "`python -m uvicorn vertice.api.main:app --port 8000`. Enquanto isso, o dashboard pontua localmente.")

    st.write("")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        with ui.glass("endpoints"):
            ui.section("fa-route", "Endpoints")
            df = pd.DataFrame(ENDPOINTS, columns=["método", "rota", "descrição"])
            df["método"] = df["método"].map(lambda m: ui.pill("good" if m == "GET" else "warn", m, "fa-arrow-right-arrow-left"))
            df["rota"] = df["rota"].map(lambda r: f"<code>{ui.esc(r)}</code>")
            ui.html_table(df, raw_html=["método", "rota"], max_height=560)
    with c2:
        with ui.glass("curl"):
            ui.section("fa-terminal", "Exemplo de chamada")
            st.code(f"curl -X POST {settings.api_url}{spec.predict_route}?cutoff={spec.default_cutoff} \\\n"
                    f"  -H 'Content-Type: application/json' \\\n  -d '{json.dumps(spec.example_payload, ensure_ascii=False)}'",
                    language="bash", wrap_lines=True)
            st.code("import httpx\n\n"
                    f"r = httpx.post('{settings.api_url}{spec.predict_route}', json=pedido)\n"
                    "r.json()  # {'probability_default': ..., 'decision': 'Aprovar', ...}\n\n"
                    f"httpx.get('{settings.api_url}/models', params={{'dataset': '{spec.id}'}}).json()",
                    language="python", wrap_lines=True)
        st.write("")
        with ui.glass("log"):
            ui.section("fa-clock-rotate-left", "Últimas pontuações registradas", f"base {spec.short}")
            log = services.prediction_log(12)
            if log.empty:
                ui.note("Nenhuma pontuação ainda. Use o simulador em <b>Resultados</b> ou chame <code>POST /predict</code>.")
            else:
                log["decisão"] = log["decision"].map(lambda d: ui.pill("good" if d == "Aprovar" else "bad", d, "fa-gavel"))
                ui.html_table(log[["created_at", "probability", "decisão", "model_id"]]
                              .rename(columns={"created_at": "quando (UTC)", "probability": "PD", "model_id": "modelo"}),
                              formats={"PD": lambda v: ui.fmt_pct(v)}, raw_html=["decisão"], max_height=300)
