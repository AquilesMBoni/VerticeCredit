from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui
from vertice import context


def render(mode: str) -> None:
    spec = context.spec()
    ui.page_header("fa-magnifying-glass-chart", "Fase 5 de 8", "Análise exploratória",
                   "Explore a base limpa. O filtro de notas e os filtros abaixo valem para todos os indicadores e gráficos da página.")
    grades = ui.grade_filter(spec)
    base_all = data.t("clean_loans")
    clean = data.tg("clean_loans", grades)

    extra = [d for d in spec.segment_dims if d not in (spec.grade,)][:2]
    with ui.glass("filters"):
        cols = st.columns(len(extra))
        picks = {}
        for col, dim in zip(cols, extra):
            opts = spec.dim_orders.get(dim) or sorted(base_all[dim].astype(str).unique())
            picks[dim] = col.multiselect(spec.segment_dims[dim], opts, key=f"eda_{spec.id}_{dim}",
                                         placeholder=f"Todas ({len(opts)})")
    df = clean
    for dim, sel in picks.items():
        if sel:
            df = df[df[dim].astype(str).isin(sel)]
    if df.empty:
        st.info("Nenhum registro corresponde aos filtros. Remova algum filtro para continuar.")
        return

    st.write("")
    base = base_all[spec.target].mean()
    dr = df[spec.target].mean()
    diff = dr - base
    arrow, cls = ("down", "up") if diff < 0 else ("up", "down")
    if abs(diff) < 0.0005:
        arrow, cls = "equals", ""
    ui.kpi_row([
        ui.kpi("fa-users", "Registros no recorte", ui.fmt_int(len(df)), f"{ui.fmt_pct(len(df) / len(base_all))} da base"),
        ui.kpi("fa-triangle-exclamation", "Taxa de inadimplência", ui.fmt_pct(dr),
               f'<span class="{cls}"><i class="fa-solid fa-arrow-{arrow}"></i> {ui.fmt_num(abs(diff) * 100, 1)} p.p.</span> vs. base'),
        ui.kpi("fa-money-bill-wave", "Valor médio", spec.fmt_money(df[spec.amount].mean()),
               f"mediana {spec.fmt_money(df[spec.amount].median())}"),
        ui.kpi("fa-percent", spec.rate_label.split(" (")[0] + " média", f"{ui.fmt_num(df[spec.rate].mean(), 2)}%",
               spec.rate_label.split("(")[-1].rstrip(")")),
    ])
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        with ui.glass("seg"):
            ui.section("fa-chart-bar", "Inadimplência por segmento")
            dim = st.selectbox("Segmentar por", list(spec.segment_dims), format_func=spec.segment_dims.get, key=f"eda_dim_{spec.id}")
            g = df.groupby(dim, as_index=False).agg(rate=(spec.target, "mean"), n=(spec.target, "size"))
            g[dim] = g[dim].astype(str)
            if dim in spec.dim_orders:
                g["_o"] = g[dim].map({v: i for i, v in enumerate(spec.dim_orders[dim])})
                g = g.sort_values("_o")
            else:
                g = g.sort_values("rate", ascending=False)
            fig = charts.bar_rate(g, dim, "rate", mode, height=330)
            fig.add_hline(y=base, line=dict(color=charts.INK[mode][2], dash="dot", width=1),
                          annotation_text=f"média da base {ui.fmt_pct(base)}", annotation_position="top left",
                          annotation_font=dict(size=10.5, color=charts.INK[mode][2]))
            fig.update_traces(customdata=g["n"], hovertemplate="%{x}<br>Inadimplência: %{text}<br>Registros: %{customdata:,}<extra></extra>")
            st.plotly_chart(fig, config=charts.config(), width="stretch")
    with c2:
        with ui.glass("dist"):
            ui.section("fa-chart-area", "Distribuição por desfecho", "% dentro de cada grupo")
            num = st.selectbox("Variável", list(spec.dist_vars), format_func=spec.dist_vars.get, key=f"eda_num_{spec.id}")
            d = df[df[num] <= df[num].quantile(.98)] if num in (spec.income, spec.amount) else df
            d = d.rename(columns={spec.target: "loan_status"})
            st.plotly_chart(charts.hist_by_target(d, num, mode, height=330, xtitle=spec.dist_vars[num]),
                            config=charts.config(), width="stretch")

    st.write("")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        with ui.glass("cross"):
            ui.section("fa-table-cells", f"Inadimplência por {'nota' if spec.id == 'us' else 'rating'} × {spec.purpose_label.lower()}")
            pv = df.pivot_table(index=spec.purpose, columns=spec.grade, values=spec.target, aggfunc="mean")
            pv = pv.reindex(columns=[g for g in spec.grade_order if g in pv.columns])
            st.plotly_chart(charts.heatmap(pv * 100, mode, height=380, fmt=".0f", colorbar_title="% inad."),
                            config=charts.config(), width="stretch")
    with c2:
        with ui.glass("corr"):
            ui.section("fa-arrows-left-right", "Correlação de Spearman")
            corr = df[list(spec.corr_cols)].corr(method="spearman").rename(index=spec.corr_cols, columns=spec.corr_cols)
            st.plotly_chart(charts.heatmap(corr, mode, height=380, diverging=True, zmid=0, fmt=".2f"),
                            config=charts.config(), width="stretch")

    st.write("")
    sc = spec.scatter
    with ui.glass("scatter"):
        ui.section("fa-braille", f"{sc['xlabel'].split(' (')[0]} × {sc['ylabel'].split(' (')[0].lower()}", "amostra de até 4.000 registros")
        s = df.sample(n=min(4000, len(df)), random_state=7)
        s = s[s[sc["x"]] <= s[sc["x"]].quantile(.98)]
        fig = go.Figure()
        for k, label in ((0, "Adimplente"), (1, "Inadimplente")):
            p = s[s[spec.target] == k]
            fig.add_trace(go.Scattergl(
                x=p[sc["x"]], y=p[sc["y"]], mode="markers", name=label,
                marker=dict(color=charts.colors(mode)[k], size=8, opacity=.55, line=dict(width=1, color="rgba(255,255,255,.35)")),
                customdata=p[[spec.grade, spec.purpose]].values,
                hovertemplate=f"{label}<br>{sc['xlabel']}: %{{x:,.0f}}<br>{sc['ylabel']}: %{{y:,.0f}}"
                              "<br>%{customdata[0]} · %{customdata[1]}<extra></extra>"))
        xmax = float(s[sc["x"]].max())
        ymax = sc["ymax"] or float(s[sc["y"]].quantile(.995)) * 1.05
        for ratio in sc["ratios"]:
            fig.add_trace(go.Scatter(x=[0, xmax], y=[0, xmax * ratio], mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=charts.INK[mode][2], width=1, dash="dot")))
            x_lab = min(ymax / ratio, xmax) * .95
            fig.add_annotation(x=x_lab, y=x_lab * ratio, text=sc["ratio_label"].format(p=int(ratio * 100)), showarrow=False,
                               font=dict(size=10.5, color=charts.INK[mode][1]), yshift=9)
        fig.update_xaxes(title=sc["xlabel"], tickformat="~s", showgrid=True)
        fig.update_yaxes(title=sc["ylabel"], tickformat="~s", range=[0, ymax])
        st.plotly_chart(charts.style(fig, mode, 430), config=charts.config(), width="stretch")
        ui.note(sc["note"])
