from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui
from vertice import context

SEV = {"alta": ("bad", "fa-angles-up"), "média": ("warn", "fa-angle-up"), "baixa": ("neutral", "fa-angle-down")}


def _box(values, title: str, mode: str, ref: float | None = None, ref_label: str = ""):
    """Boxplot horizontal com hover só nos pontos e resumo dos quartis abaixo."""
    v = values.dropna()
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    fig = go.Figure(go.Box(
        x=v, name="", orientation="h", boxpoints="outliers", hoveron="points",
        marker=dict(color=charts.colors(mode)[1], size=7, opacity=.75),
        line=dict(color=charts.colors(mode)[0], width=1.5), fillcolor="rgba(42,120,214,.15)",
        hovertemplate=f"{title}: %{{x}}<extra></extra>"))
    if ref is not None:
        fig.add_vline(x=ref, line=dict(color="#d03b3b", dash="dot", width=1.5), annotation_text=ref_label,
                      annotation_position="top", annotation_font=dict(size=10.5, color=charts.INK[mode][1]))
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(title=title, showgrid=True)
    fig.update_layout(hovermode="closest")
    stats = (f"mediana {ui.fmt_num(med, 1)} · Q1 {ui.fmt_num(q1, 1)} · Q3 {ui.fmt_num(q3, 1)} · "
             f"mín {ui.fmt_num(v.min(), 1)} · máx {ui.fmt_num(v.max(), 1)}")
    return charts.style(fig, mode, 190, legend=False), stats


def render(mode: str) -> None:
    spec = context.spec()
    ui.page_header("fa-clipboard-check", "Fase 2 de 8", "Qualidade de dados",
                   "Um contrato escrito com <b>Pandera</b> define como os dados deveriam ser. Ele roda duas vezes: "
                   "no bruto, para diagnosticar, e no limpo, como garantia antes de qualquer análise. Cada regra "
                   "pertence a uma dimensão de qualidade."
                   + (" A base brasileira tem também <b>formato</b> (moeda e data em padrão brasileiro) e "
                      "<b>conformidade regulatória</b> (teto do INSS, margem consignável, elegibilidade)." if spec.id == "br" else ""))
    grades = ui.grade_filter(spec)

    checks, dims, s_raw, s_clean = data.quality(grades)
    raw = data.tg("raw_loans", grades).drop(columns="record_id")
    if raw.empty:
        st.info("Nenhum registro nas notas escolhidas.")
        return

    raw_c = checks[checks["stage"] == "bruto"]
    clean_c = checks[checks["stage"] == "limpo"]
    ui.kpi_row([
        ui.kpi("fa-gauge-high", "Score no bruto", ui.fmt_num(s_raw, 2), "% de linhas sem nenhuma violação"),
        ui.kpi("fa-shield-halved", "Score após limpeza", ui.fmt_num(s_clean, 2),
               f'<span class="up"><i class="fa-solid fa-arrow-up"></i> {ui.fmt_num(s_clean - s_raw, 2)} pts</span>'),
        ui.kpi("fa-list-check", "Regras avaliadas", str(raw_c["rule"].nunique()), f"{dims['dimension'].nunique()} dimensões"),
        ui.kpi("fa-circle-xmark", "Regras reprovadas",
               f'{int((raw_c.status == "reprovado").sum())} → {int((clean_c.status == "reprovado").sum())}', "bruto → limpo"),
    ])
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        with ui.glass("dims"):
            ui.section("fa-layer-group", "Score por dimensão", "100 = nenhuma linha afetada")
            fig = go.Figure()
            for i, (stage, label) in enumerate((("bruto", "Bruto"), ("limpo", "Limpo"))):
                d = dims[dims["stage"] == stage].sort_values("dimension")
                fig.add_trace(go.Bar(y=d["dimension"], x=d["score"], name=label, orientation="h",
                                     marker=dict(color=charts.colors(mode)[i], cornerradius=4), customdata=d["failed_rows"],
                                     hovertemplate=f"{label} · %{{y}}<br>Score: %{{x:.2f}}<br>Linhas afetadas: %{{customdata:,}}<extra></extra>"))
            fig.update_layout(barmode="group")
            lo = max(0, float(dims["score"].min()) - 3)
            fig.update_xaxes(range=[lo, 100.5], showgrid=True, title="Score")
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(charts.style(fig, mode, 330), config=charts.config(), width="stretch")
    with c2:
        with ui.glass("missing"):
            ui.section("fa-circle-half-stroke", "Valores ausentes no bruto")
            blank = raw.apply(lambda s: s.isna() | (s.astype(str).str.strip() == ""))
            miss = blank.mean().reset_index()
            miss.columns = ["coluna", "pct"]
            miss = miss.sort_values("pct", ascending=False).head(6)
            fig = charts.bar_rate(miss, "coluna", "pct", mode, horizontal=True, height=330, color_idx=0,
                                  text_fmt=lambda v: ui.fmt_pct(v, 2), hover_label="Ausentes")
            st.plotly_chart(fig, config=charts.config(), width="stretch")
            if spec.texts.get("missing_note"):
                ui.note(spec.texts["missing_note"])

    st.write("")
    with ui.glass("rules"):
        ui.section("fa-scroll", "Contrato de dados: resultado por regra", ui.filter_note(grades).strip(" ·"))
        stage = st.segmented_control("Camada", ["bruto", "limpo"], default="bruto", key="q_stage",
                                     format_func=lambda s: "Dados brutos" if s == "bruto" else "Dados limpos")
        view = checks[checks["stage"] == (stage or "bruto")].copy()
        view["status"] = view["status"].map(lambda s: ui.pill("good", "aprovado") if s == "aprovado" else ui.pill("bad", "reprovado"))
        view["severity"] = view["severity"].map(lambda s: ui.pill(SEV[s][0], s, SEV[s][1]))
        out = view[["dimension", "description", "column", "severity", "failed_rows", "failed_pct", "status"]]
        out.columns = ["dimensão", "regra", "coluna", "severidade", "linhas", "% base", "status"]
        ui.html_table(out, formats={"linhas": ui.fmt_int, "% base": lambda v: ui.fmt_num(v, 2) + "%"},
                      raw_html=["severidade", "status"], max_height=560)

    st.write("")
    with ui.glass("outliers"):
        ui.section("fa-chart-simple", "Valores fora da regra no bruto", "boxplot · passe o mouse nos pontos")
        cols = st.columns(len(spec.impossible_vars))
        for col, (name, title) in zip(cols, spec.impossible_vars):
            series = raw[name]
            ref, ref_label = None, ""
            if spec.id == "br" and name == "taxa_juros_am":
                from vertice.datasets import br_rules as R
                series = raw.loc[raw["modalidade"].astype(str).str.strip() == "Consignado INSS", name]
                ref, ref_label = R.TETO_JUROS_INSS_AM, f"teto {ui.fmt_num(R.TETO_JUROS_INSS_AM)}%"
            with col:
                if series.dropna().empty:
                    st.caption("Sem registros neste recorte.")
                    continue
                fig, stats = _box(series, title, mode, ref, ref_label)
                st.plotly_chart(fig, config=charts.config(), width="stretch")
                st.markdown(f'<div class="vx-note" style="text-align:center;font-size:.8rem">{stats}</div>', unsafe_allow_html=True)
        ui.note(spec.texts.get("quality_note", ""))
