from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui
from vertice import context, services

FLOW = [("fa-file-import", "Ingestão"), ("fa-clipboard-check", "Validação (bruto)"), ("fa-broom", "Limpeza"),
        ("fa-shield-halved", "Validação (limpo)"), ("fa-wand-magic-sparkles", "Atributos e IV"),
        ("fa-brain", "Modelagem"), ("fa-wave-square", "Monitoramento (PSI)"), ("fa-box-archive", "Exportação")]


def render(mode: str) -> None:
    spec = context.spec()
    ui.page_header("fa-compass", f"Vértice Credit Analytics · base {spec.short}",
                   "Risco de crédito, do arquivo bruto à decisão de aprovar",
                   f"Pipeline completa sobre <b>{spec.name}</b>: contrato de qualidade, limpeza auditável, atributos, "
                   "modelo de Probabilidade de Default, simulação de política e API de pontuação. "
                   "Troque a base na barra lateral.")
    grades = ui.grade_filter(spec)

    ov = data.overview(grades)
    clean = data.tg("clean_loans", grades)
    run = ov["run"]
    timings = services.step_timings(run["run_id"] if run else None)
    if clean.empty:
        st.info("Nenhum registro nas notas escolhidas. Clique em “Todas as notas” para limpar o filtro.")
        return

    champ = ov["champion"]
    auc = "—" if champ["roc_auc"] is None else ui.fmt_num(champ["roc_auc"], 3)
    gini_ks = "" if champ["gini"] is None else f'Gini {ui.fmt_num(champ["gini"], 3)} · KS {ui.fmt_num(champ["ks"], 3)}'
    ui.kpi_row([
        ui.kpi("fa-database", "Registros após limpeza", ui.fmt_int(ov["rows_clean"]),
               f'{ui.fmt_int(ov["rows_raw"] - ov["rows_clean"])} removidos de {ui.fmt_int(ov["rows_raw"])}'),
        ui.kpi("fa-triangle-exclamation", "Taxa de inadimplência", ui.fmt_pct(ov["default_rate"]),
               f"proporção de {spec.target} = 1{ui.filter_note(grades)}"),
        ui.kpi("fa-shield-halved", "Score de qualidade", ui.fmt_num(run["quality_score"], 1),
               f'<span class="up"><i class="fa-solid fa-arrow-up"></i> {ui.fmt_num(run["quality_score"] - run["quality_raw"], 1)} pts</span> sobre o bruto'),
        ui.kpi("fa-brain", f'ROC-AUC · {champ["label"]}', auc, gini_ks or "recorte com uma só classe"),
    ])
    st.write("")

    with ui.glass("flow"):
        ui.section("fa-diagram-project", "Etapas da última execução", f'run {run["run_id"]}' if run else "")
        tmap = dict(zip(timings["step"], timings["seconds"]))
        nodes = []
        for i, (icon, name) in enumerate(FLOW):
            sec = tmap.get(name)
            nodes.append(f'<div class="node"><i class="fa-solid {icon}"></i>{name}'
                         f'<small>{ui.fmt_num(sec, 2) + " s" if sec is not None else ""}</small></div>')
            if i < len(FLOW) - 1:
                nodes.append('<span class="sep"><i class="fa-solid fa-chevron-right"></i></span>')
        st.markdown(f'<div class="vx-flow">{"".join(nodes)}</div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        with ui.glass("insights"):
            ui.section("fa-lightbulb", "Principais achados", ui.filter_note(grades).strip(" ·"))
            pol = data.policy(0.6, grades)
            best = pol.loc[pol["profit"].idxmax()]
            items = spec.insights(clean, pol) + [(
                "fa-sack-dollar", f"No conjunto de teste, o corte de PD que maximiza o resultado é <b>{ui.fmt_pct(best['cutoff'], 0)}</b>, "
                                  f"aprovando <b>{ui.fmt_pct(best['approval_rate'])}</b> dos pedidos com inadimplência de "
                                  f"<b>{ui.fmt_pct(best['bad_rate_approved'])}</b> entre os aprovados.")]
            st.markdown("".join(
                f'<div style="display:flex;gap:.75rem;padding:.55rem 0;border-bottom:1px solid var(--hairline)">'
                f'<i class="fa-solid {ic}" style="color:var(--accent);margin-top:.2rem;width:18px"></i>'
                f'<div class="vx-note" style="margin:0">{txt}</div></div>' for ic, txt in items), unsafe_allow_html=True)
    with right:
        with ui.glass("grade"):
            ui.section("fa-ranking-star", f"Inadimplência por {'nota' if spec.id == 'us' else 'rating'}")
            g = clean.groupby(spec.grade, as_index=False).agg(rate=(spec.target, "mean"), n=(spec.target, "size"))
            g[spec.grade] = g[spec.grade].astype("category").cat.set_categories(list(spec.grade_order))
            g = g.sort_values(spec.grade)
            g[spec.grade] = g[spec.grade].astype(str)
            fig = charts.bar_rate(g, spec.grade, "rate", mode, height=320)
            fig.update_traces(customdata=g["n"], hovertemplate="%{x}<br>Inadimplência: %{text}<br>Pedidos: %{customdata:,}<extra></extra>")
            st.plotly_chart(fig, config=charts.config(), width="stretch")

    st.write("")
    with ui.glass("volume"):
        ui.section("fa-sack-dollar", f"Volume por {spec.purpose_label.lower()} e situação", f"{spec.currency}, base limpa")
        v = clean.groupby([spec.purpose, spec.target], as_index=False)[spec.amount].sum()
        order = v.groupby(spec.purpose)[spec.amount].sum().sort_values(ascending=False).index
        fig = go.Figure()
        for k, label in ((0, "Adimplente"), (1, "Inadimplente")):
            s = v[v[spec.target] == k].set_index(spec.purpose).reindex(order)
            fig.add_trace(go.Bar(x=list(order), y=s[spec.amount], name=label,
                                 marker=dict(color=charts.colors(mode)[k], cornerradius=4),
                                 hovertemplate=f"%{{x}}<br>{label}: {spec.currency} %{{y:,.0f}}<extra></extra>"))
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickprefix=f"{spec.currency} ", tickformat="~s")
        st.plotly_chart(charts.style(fig, mode, 320), config=charts.config(), width="stretch")
