from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui
from vertice import context

ACTION_PILL = {"Remover": "bad", "Anular para imputar": "warn", "Imputar": "neutral", "Winsorizar": "warn",
               "Recalcular": "neutral", "Normalizar": "neutral", "Converter": "neutral", "Preencher": "neutral"}


def render(mode: str) -> None:
    spec = context.spec()
    ui.page_header("fa-broom", "Fase 3 de 8", "Limpeza",
                   spec.texts.get("cleaning_intro", "") + " Valores imputados ganham uma coluna-flag e voltam a ser "
                   "nulos na modelagem, onde a imputação é aprendida só no conjunto de treino. Isso evita vazamento "
                   "de informação do teste.")
    grades = ui.grade_filter(spec)

    log = data.cleaning_log(grades)
    raw = data.tg("raw_loans", grades)
    clean = data.tg("clean_loans", grades)
    if raw.empty:
        st.info("Nenhum registro nas notas escolhidas.")
        return

    removed = int(log.loc[log["action"] == "Remover", "affected"].sum())
    imputed = int(log.loc[log["action"] == "Imputar", "affected"].sum())
    changed = int(log.loc[log["action"].isin(["Normalizar", "Converter", "Recalcular", "Winsorizar"]), "affected"].sum())
    ui.kpi_row([
        ui.kpi("fa-filter", "Linhas removidas", ui.fmt_int(removed), f"{ui.fmt_pct(removed / len(raw), 2)} do bruto"),
        ui.kpi("fa-fill-drip", "Valores imputados", ui.fmt_int(imputed), "sempre com coluna-flag"),
        ui.kpi("fa-pen-to-square", "Valores corrigidos", ui.fmt_int(changed), "padronizados, convertidos ou recalculados"),
        ui.kpi("fa-list-check", "Regras aplicadas", str(len(log)), "na ordem do cleaning_log"),
    ])
    st.write("")

    c1, c2 = st.columns([1, 1.15])
    with c1:
        with ui.glass("timeline"):
            ui.section("fa-list-ol", "Regras aplicadas, em ordem", ui.filter_note(grades).strip(" ·"))
            items = []
            for _, r in log.iterrows():
                items.append(f"""<div class="item"><div class="dot">{int(r['order'])}</div>
<div class="title">{ui.esc(r['step'])}</div><div class="desc">{ui.esc(r['rule'])}</div>
<div class="meta">{ui.pill(ACTION_PILL.get(r['action'], 'neutral'), r['action'], 'fa-gear')}
{ui.pill('neutral', f"{ui.fmt_int(r['affected'])} linhas afetadas", 'fa-hashtag')}</div></div>""")
            st.markdown(f'<div class="vx-timeline">{"".join(items)}</div>', unsafe_allow_html=True)
    with c2:
        with ui.glass("funnel"):
            ui.section("fa-filter-circle-xmark", "Linhas ao longo da limpeza")
            rem = log[(log["action"] == "Remover") & (log["affected"] > 0)]
            steps = [("Bruto", len(raw))] + [(r["step"], r["rows_after"]) for _, r in rem.iterrows()]
            seq = [charts.SEQ[min(3 + i, len(charts.SEQ) - 1)] for i in range(len(steps))]
            fig = go.Figure(go.Funnel(
                y=[s for s, _ in steps], x=[n for _, n in steps],
                text=[f"{ui.fmt_int(n)} · {ui.fmt_pct(n / steps[0][1], 2)}" for _, n in steps], textinfo="text",
                textfont=dict(color="#fff", size=12), marker=dict(color=seq),
                connector=dict(line=dict(color="rgba(0,0,0,0)")), hovertemplate="%{y}<br>%{x:,} linhas<extra></extra>"))
            st.plotly_chart(charts.style(fig, mode, 70 + 48 * len(steps), legend=False), config=charts.config(), width="stretch")

        st.write("")
        col, flag, title, ylabel = spec.impute_view
        with ui.glass("impute"):
            ui.section("fa-fill-drip", title, "pontos laranja = imputados")
            s = clean.sample(n=min(6000, len(clean)), random_state=1)
            gidx = spec.grade_index(s[spec.grade])
            fig = go.Figure()
            for f, label, idx, size in ((0, "Observado", 0, 4), (1, "Imputado (mediana)", 1, 7)):
                m = s[flag] == f
                jitter = np.random.default_rng(2).uniform(-0.28, 0.28, int(m.sum()))
                fig.add_trace(go.Scattergl(x=gidx[m] + jitter, y=s.loc[m, col], mode="markers", name=label,
                                           marker=dict(color=charts.colors(mode)[idx], size=size,
                                                       opacity=.35 if f == 0 else .9, line=dict(width=0)),
                                           hovertemplate=f"{label}<br>%{{y:.2f}}<extra></extra>"))
            fig.update_xaxes(tickvals=list(range(1, len(spec.grade_order) + 1)), ticktext=list(spec.grade_order),
                             title="Nota" if spec.id == "us" else "Rating")
            fig.update_yaxes(title=ylabel)
            st.plotly_chart(charts.style(fig, mode, 300), config=charts.config(), width="stretch")

    st.write("")
    with ui.glass("compare"):
        ui.section("fa-code-compare", "Distribuição antes e depois")
        var = st.selectbox("Variável", list(spec.compare_vars), format_func=spec.compare_vars.get, key=f"clean_var_{spec.id}")
        raw_num = spec.validation_view(raw.drop(columns="record_id"))
        r = raw_num[var].dropna()
        c = clean[var].dropna()
        hi = float(c.quantile(0.995)) * 1.2
        r_plot = r[(r <= hi) & (r >= 0)]
        lo, top = float(min(r_plot.min(), c.min())), float(max(r_plot.max(), c.max()))
        bins = dict(start=lo, end=top, size=(top - lo) / 60 or 1)
        fig = go.Figure()
        for series, label, idx in ((r_plot, "Bruto", 1), (c, "Limpo", 0)):
            fig.add_trace(go.Histogram(x=series, name=label, xbins=bins, histnorm="percent", opacity=.62,
                                       marker=dict(color=charts.colors(mode)[idx], line=dict(width=0)),
                                       hovertemplate=f"{label}<br>%{{x}}<br>%{{y:.2f}}%<extra></extra>"))
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title=spec.compare_vars[var])
        fig.update_yaxes(title="% das linhas")
        st.plotly_chart(charts.style(fig, mode, 320), config=charts.config(), width="stretch")
        cols = st.columns(4)
        stats = [("Média bruto", r.mean()), ("Média limpo", c.mean()), ("Máx. bruto", r.max()), ("Máx. limpo", c.max())]
        for col_, (lab, v) in zip(cols, stats):
            col_.markdown(f'<div class="vx-note">{lab}<br><b style="font-family:Sora;font-size:1.15rem">{ui.fmt_num(v, 2)}</b></div>',
                          unsafe_allow_html=True)
        if len(r_plot) < len(r):
            ui.note(f"{ui.fmt_int(len(r) - len(r_plot))} valores extremos do bruto ficaram fora do gráfico para não achatar a escala.")
