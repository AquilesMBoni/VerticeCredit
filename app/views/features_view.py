from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui
from vertice import context

STRENGTH_PILL = {"Irrelevante": "neutral", "Fraco": "neutral", "Médio": "good", "Forte": "good", "Muito forte": "warn"}


def _short(v: float) -> str:
    if abs(v) >= 1000:
        return ui.fmt_num(v / 1000, 0 if abs(v) >= 10000 else 1) + "k"
    if abs(v) >= 10:
        return ui.fmt_num(v, 0)
    return ui.fmt_num(v, 3 if abs(v) < 1 else 1)


def _pretty_interval(label: str) -> str:
    lo, hi = (float(x) for x in label.strip("([]").split(","))
    return f"{_short(max(lo, 0) if lo < 0 else lo)} – {_short(hi)}"


def render(mode: str) -> None:
    spec = context.spec()
    ui.page_header("fa-wand-magic-sparkles", "Fase 4 de 8", "Atributos e poder preditivo",
                   "Criamos atributos com significado de negócio e medimos quanto cada variável separa bons e maus "
                   "pagadores com <b>WoE</b> (Weight of Evidence) e <b>IV</b> (Information Value), a métrica clássica "
                   "de modelos de crédito, antes de treinar qualquer modelo.")
    grades = ui.grade_filter(spec)

    woe, iv = data.iv(grades)
    clean = data.tg("clean_loans", grades)
    if clean.empty:
        st.info("Nenhum registro nas notas escolhidas.")
        return

    with ui.glass("new"):
        ui.section("fa-plus", "Atributos criados")
        rows = []
        for k, v in spec.feature_docs.items():
            s = clean[k]
            sample = ui.fmt_num(s.median(), 3) if pd.api.types.is_numeric_dtype(s) else ", ".join(sorted(s.astype(str).unique())[:4]) + "…"
            rows.append({"atributo": k, "definição": v, "mediana / valores": sample})
        ui.html_table(pd.DataFrame(rows))

    st.write("")
    c1, c2 = st.columns([1, 1.1])
    with c1:
        with ui.glass("iv"):
            ui.section("fa-ranking-star", "Information Value por variável", ui.filter_note(grades).strip(" ·"))
            d = iv.sort_values("iv")
            seq = charts.SEQ
            col = [seq[2] if v < .02 else seq[3] if v < .1 else seq[4] if v < .3 else seq[5] if v < .5 else seq[6] for v in d["iv"]]
            fig = go.Figure(go.Bar(y=d["feature"], x=d["iv"], orientation="h", marker=dict(color=col, cornerradius=4),
                                   text=[ui.fmt_num(v, 3) for v in d["iv"]], textposition="outside", cliponaxis=False,
                                   customdata=d["strength"], hovertemplate="%{y}<br>IV: %{text}<br>%{customdata}<extra></extra>"))
            for x, lab in ((0.1, "fraco"), (0.3, "médio"), (0.5, "forte")):
                fig.add_vline(x=x, line=dict(color=charts.INK[mode][2], width=1, dash="dot"), annotation_text=lab,
                              annotation_position="bottom right", annotation_font=dict(size=10, color=charts.INK[mode][2]))
            fig.update_traces(textfont=dict(color=charts.INK[mode][1], size=11))
            fig.update_xaxes(showgrid=True, title="IV", range=[0, max(float(d["iv"].max()), 0.55) * 1.2])
            st.plotly_chart(charts.style(fig, mode, 40 + 30 * len(d), legend=False), config=charts.config(), width="stretch")
    with c2:
        with ui.glass("ivtable"):
            ui.section("fa-table", "Leitura do IV")
            t = iv.copy()
            t["força"] = t["strength"].map(lambda s: ui.pill(STRENGTH_PILL[s], s, "fa-signal"))
            ui.html_table(t[["feature", "iv", "força", "n_bins"]].rename(columns={"feature": "variável", "n_bins": "faixas"}),
                          formats={"iv": lambda v: ui.fmt_num(v, 4)}, raw_html=["força"], max_height=330)
            ui.note(spec.texts.get("iv_note", ""))
            if grades:
                ui.note(f"Com o filtro de notas ativo, a própria {'nota' if spec.id == 'us' else 'rating'} perde poder de "
                        "separação: dentro do recorte ela varia pouco.")

    st.write("")
    with ui.glass("woe"):
        ui.section("fa-chart-column", "Weight of Evidence por faixa", "WoE negativo = faixa mais arriscada")
        feat = st.selectbox("Variável", iv["feature"].tolist(), key=f"woe_feat_{spec.id}")
        w = woe[woe["feature"] == feat].copy()
        if w["bin"].str.startswith("(").all():
            w["_o"] = w["bin"].str.extract(r"\(([-\d.e]+),")[0].astype(float)
            w = w.sort_values("_o")
            w["bin"] = w["bin"].map(_pretty_interval)
        elif feat == spec.grade:
            w["_o"] = spec.grade_index(w["bin"])
            w = w.sort_values("_o")
        fig = go.Figure(go.Bar(
            x=w["bin"], y=w["woe"],
            marker=dict(color=[charts.colors(mode)[0] if v >= 0 else charts.colors(mode)[1] for v in w["woe"]], cornerradius=4),
            customdata=w[["n", "bad_rate"]].values,
            hovertemplate="%{x}<br>WoE: %{y:.3f}<br>Registros: %{customdata[0]:,}<br>Inadimplência: %{customdata[1]:.1%}<extra></extra>"))
        fig.update_yaxes(title="WoE", zeroline=True, zerolinewidth=1.5)
        fig.update_xaxes(title=None, tickfont=dict(size=10.5))
        st.plotly_chart(charts.style(fig, mode, 340, legend=False), config=charts.config(), width="stretch")
