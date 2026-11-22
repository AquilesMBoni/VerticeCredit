from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui
from vertice import context, db, services

LOCAL_TZ = "America/Sao_Paulo"


def _when(iso: str) -> str:
    """ISO UTC → horário local legível (dd/mm/aaaa hh:mm)."""
    try:
        return pd.Timestamp(iso).tz_convert(LOCAL_TZ).strftime("%d/%m/%Y %H:%M")
    except Exception:
        return str(iso)


def render(mode: str) -> None:
    cols = charts.colors(mode)
    runs = services.runs(50)
    timings = services.step_timings()
    spec = context.spec()
    registry = db.read_sql("SELECT model_id, name, version, is_champion, created_at, metrics FROM model_registry ORDER BY created_at DESC")

    ui.page_header("fa-wave-square", "Operação", "Monitoramento",
                   "Histórico de execuções, tempo por etapa, registro de modelos e estabilidade da população "
                   "(PSI). Em produção, o PSI compara a base de treino com os pedidos que chegam a cada mês.")
    grades = ui.grade_filter(spec)
    psi = data.psi(grades)
    if psi.empty:
        st.info("Poucos registros nas notas escolhidas para calcular o PSI. Inclua mais notas.")
        return

    ok = runs[runs["status"] == "sucesso"]
    ui.kpi_row([
        ui.kpi("fa-repeat", "Execuções", str(len(runs)), f"{len(ok)} com sucesso"),
        ui.kpi("fa-stopwatch", "Duração média", f"{ui.fmt_num(ok['duration_s'].mean(), 1)} s" if len(ok) else "—", "pipeline completa"),
        ui.kpi("fa-cubes", "Modelos registrados", str(len(registry)), f"versão atual v{int(registry['version'].max())}" if len(registry) else ""),
        ui.kpi("fa-wave-square", "Maior PSI", ui.fmt_num(psi["psi"].max(), 4), psi.iloc[0]["status"] + " · treino vs. teste"),
    ])
    st.write("")

    c1, c2 = st.columns(2)
    last = runs.iloc[0]["run_id"] if len(runs) else None
    with c1:
        with ui.glass("steps"):
            ui.section("fa-stopwatch", "Tempo por etapa", "última execução · não depende do filtro")
            d = timings[timings["run_id"] == last]
            fig = go.Figure(go.Bar(y=d["step"], x=d["seconds"], orientation="h", marker=dict(color=cols[0], cornerradius=4),
                                   text=[f"{ui.fmt_num(s, 2)} s" for s in d["seconds"]], textposition="outside", cliponaxis=False,
                                   customdata=d["rows_out"], hovertemplate="%{y}<br>%{x:.2f} s<br>Saída: %{customdata:,} linhas<extra></extra>"))
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(showgrid=True, title="segundos", range=[0, float(d["seconds"].max()) * 1.18])
            fig.update_traces(textfont=dict(color=charts.INK[mode][1], size=11))
            st.plotly_chart(charts.style(fig, mode, 330, legend=False), config=charts.config(), width="stretch")
    with c2:
        with ui.glass("psi"):
            ui.section("fa-scale-balanced", "PSI treino × teste", f"< 0,10 estável{ui.filter_note(grades)}")
            d = psi.sort_values("psi")
            fig = go.Figure(go.Bar(y=d["feature"], x=d["psi"], orientation="h", marker=dict(color=cols[2], cornerradius=4),
                                   customdata=d["status"], hovertemplate="%{y}<br>PSI %{x:.4f}<br>%{customdata}<extra></extra>"))
            fig.add_vline(x=0.10, line=dict(color="#fab219", dash="dot", width=1.5), annotation_text="atenção 0,10", annotation_position="bottom left",
                          annotation_font=dict(size=10, color=charts.INK[mode][2]))
            fig.update_xaxes(range=[0, max(0.12, float(d["psi"].max()) * 1.2)], showgrid=True, title="PSI")
            st.plotly_chart(charts.style(fig, mode, 330, legend=False), config=charts.config(), width="stretch")

    st.write("")
    with ui.glass("runs"):
        ui.section("fa-clock-rotate-left", "Histórico de execuções", "tabela pipeline_runs")
        t = runs.copy()
        t["status"] = t["status"].map(lambda s: ui.pill("good", s) if s == "sucesso" else ui.pill("bad", s) if s == "falha" else ui.pill("warn", s))
        t["início"] = t["started_at"].map(_when)
        t["execução"] = t["run_id"].map(lambda r: f"<code>{ui.esc(r)}</code>")
        out = t[["execução", "início", "status", "duration_s", "rows_raw", "rows_clean", "quality_raw", "quality_score", "champion_auc"]]
        out.columns = ["execução", "início", "status", "duração (s)", "linhas brutas", "linhas limpas", "qualidade bruto",
                       "qualidade limpo", "AUC"]
        ui.html_table(out, raw_html=["status", "execução"], formats={"duração (s)": lambda v: ui.fmt_num(v, 2), "linhas brutas": ui.fmt_int,
                                                          "linhas limpas": ui.fmt_int, "qualidade bruto": lambda v: ui.fmt_num(v, 2),
                                                          "qualidade limpo": lambda v: ui.fmt_num(v, 2), "AUC": lambda v: ui.fmt_num(v, 4)})
        if len(ok) > 1:
            fig = go.Figure(go.Scatter(x=ok["started_at"], y=ok["duration_s"], mode="lines+markers", line=dict(color=cols[0], width=2),
                                       marker=dict(size=8), hovertemplate="%{x}<br>%{y:.2f} s<extra></extra>"))
            fig.update_yaxes(title="duração (s)")
            st.plotly_chart(charts.style(fig, mode, 220, legend=False), config=charts.config(), width="stretch")

    st.write("")
    with ui.glass("registry"):
        ui.section("fa-cubes", "Registro de modelos", "artefatos .joblib versionados em models/")
        import json
        r = registry.copy()
        r["AUC"] = r["metrics"].map(lambda m: json.loads(m)["roc_auc"])
        r["CV AUC"] = r["metrics"].map(lambda m: json.loads(m)["cv_auc_mean"])
        r["campeão"] = r["is_champion"].map(lambda v: ui.pill("good", "campeão", "fa-trophy") if v else "")
        r["criado em"] = r["created_at"].map(_when)
        ui.html_table(r[["model_id", "name", "version", "criado em", "AUC", "CV AUC", "campeão"]]
                      .rename(columns={"model_id": "id", "name": "algoritmo", "version": "versão"}),
                      formats={"AUC": lambda v: ui.fmt_num(v, 4), "CV AUC": lambda v: ui.fmt_num(v, 4)}, raw_html=["campeão"],
                      max_height=300)
