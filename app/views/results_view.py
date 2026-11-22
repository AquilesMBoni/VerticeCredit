from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui
from vertice import context


def _field(col, f, key: str):
    if f.kind == "select":
        idx = list(f.options).index(f.default)
        return col.selectbox(f.label, f.options, index=idx, key=key,
                             format_func=(lambda v, m=f.labels: m.get(v, v)) if f.labels else str)
    if f.kind == "int":
        return col.number_input(f.label, int(f.min), int(f.max), int(f.default), step=int(f.step or 1), key=key)
    return col.number_input(f.label, float(f.min), float(f.max), float(f.default), step=float(f.step or 0.1),
                            format="%.2f", key=key)


def render(mode: str) -> None:
    spec = context.spec()
    cols = charts.colors(mode)
    ui.page_header("fa-scale-balanced", "Fase 7 de 8", "Resultados e política de crédito",
                   "O modelo vira decisão: <b>aprovar se a PD for menor que o corte</b>. Simulamos a política no "
                   "conjunto de teste. " + spec.texts.get("policy_intro", ""))
    grades = ui.grade_filter(spec)

    with ui.glass("params"):
        c1, c2 = st.columns(2)
        lgd = c1.slider("LGD: perda sobre o valor em caso de inadimplência", 0.10, 1.00, 0.60, 0.05, key="lgd",
                        help="Loss Given Default. 0,60 significa recuperar 40% do valor emprestado.")
        pol = data.policy(lgd, grades)
        best = pol.loc[pol["profit"].idxmax()]
        cut = c2.slider("Corte de PD da política", 0.02, 0.98, float(round(best["cutoff"], 2)), 0.01,
                        key=f"cut_{spec.id}_{'-'.join(grades)}")
    row = pol.iloc[(pol["cutoff"] - cut).abs().argmin()]
    all_in = pol.iloc[-1]

    st.write("")
    delta = row["profit"] - all_in["profit"]
    money = lambda v: ui.fmt_compact_usd(v).replace("US$", spec.currency)  # noqa: E731
    ui.kpi_row([
        ui.kpi("fa-circle-check", "Taxa de aprovação", ui.fmt_pct(row["approval_rate"]), f"no corte de {ui.fmt_pct(cut, 0)}"),
        ui.kpi("fa-triangle-exclamation", "Inadimplência entre aprovados", ui.fmt_pct(row["bad_rate_approved"]),
               f"contra {ui.fmt_pct(all_in['bad_rate_approved'])} aprovando todos"),
        ui.kpi("fa-shield-halved", "Inadimplentes barrados", ui.fmt_pct(row["bads_rejected_pct"]),
               f"{ui.fmt_pct(row['goods_rejected_pct'])} dos bons recusados"),
        ui.kpi("fa-sack-dollar", "Resultado simulado", money(row["profit"]),
               f'<span class="{"up" if delta >= 0 else "down"}">{"+" if delta >= 0 else ""}{money(delta)}</span> vs. aprovar todos'),
    ])
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        with ui.glass("profit"):
            ui.section("fa-chart-line", "Resultado por corte de PD", f"máximo em {ui.fmt_pct(best['cutoff'], 0)}{ui.filter_note(grades)}")
            fig = go.Figure(go.Scatter(x=pol["cutoff"], y=pol["profit"], mode="lines", name="Resultado",
                                       line=dict(color=cols[0], width=2.5), fill="tozeroy", fillcolor="rgba(42,120,214,.10)",
                                       hovertemplate=f"Corte %{{x:.0%}}<br>Resultado: {spec.currency} %{{y:,.0f}}<extra></extra>"))
            fig.add_vline(x=cut, line=dict(color=cols[1], width=1.5, dash="dash"), annotation_text="corte escolhido",
                          annotation_font=dict(size=10.5, color=charts.INK[mode][1]))
            fig.add_trace(go.Scatter(x=[best["cutoff"]], y=[best["profit"]], mode="markers", name="Ótimo",
                                     marker=dict(size=11, color=cols[2], line=dict(width=2, color=charts.SURFACE[mode])),
                                     hovertemplate=f"Ótimo<br>Corte %{{x:.0%}}<br>{spec.currency} %{{y:,.0f}}<extra></extra>"))
            fig.update_xaxes(title="Corte de PD", tickformat=".0%", showgrid=True)
            fig.update_yaxes(title=f"Resultado ({spec.currency})", tickprefix=f"{spec.currency} ", tickformat="~s")
            st.plotly_chart(charts.style(fig, mode, 350, legend=False), config=charts.config(), width="stretch")
    with c2:
        with ui.glass("tradeoff"):
            ui.section("fa-arrows-left-right", "Aprovação × inadimplência da carteira")
            fig = go.Figure(go.Scatter(x=pol["approval_rate"], y=pol["bad_rate_approved"], mode="lines",
                                       line=dict(color=cols[0], width=2.5), customdata=pol["cutoff"], name="Política",
                                       hovertemplate="Corte %{customdata:.0%}<br>Aprovação %{x:.1%}<br>Inadimplência %{y:.1%}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[row["approval_rate"]], y=[row["bad_rate_approved"]], mode="markers",
                                     marker=dict(size=12, color=cols[1], line=dict(width=2, color=charts.SURFACE[mode])),
                                     name="Corte escolhido", hovertemplate="Corte escolhido<extra></extra>"))
            fig.update_xaxes(title="Taxa de aprovação", tickformat=".0%", showgrid=True)
            fig.update_yaxes(title="Inadimplência entre aprovados", tickformat=".0%")
            st.plotly_chart(charts.style(fig, mode, 350, legend=False), config=charts.config(), width="stretch")

    st.write("")
    with ui.glass("simulator"):
        ui.section("fa-user-check", "Simular um pedido", "pontuação via API, com fallback local")
        if spec.id == "br":
            ui.note("A proposta passa pelas mesmas regras da API: teto do consignado INSS, margem consignável, "
                    "elegibilidade e prazo máximo. Se alguma for violada, a pontuação é recusada com o motivo.")
        with st.form(f"applicant_{spec.id}", border=False):
            fields = list(spec.form_fields)
            payload = {}
            for start in range(0, len(fields), 4):
                row_cols = st.columns(4)
                for col, f in zip(row_cols, fields[start:start + 4]):
                    payload[f.name] = _field(col, f, f"sim_{spec.id}_{f.name}")
            submitted = st.form_submit_button("Calcular probabilidade de default", icon=":material/calculate:")
        state_key = f"last_pred_{spec.id}"
        if submitted:
            st.session_state[state_key] = data.predict(payload, cut)
        if state_key in st.session_state:
            res, via, ms = st.session_state[state_key]
            if "error" in res:
                st.markdown(f'<div style="padding-top:.6rem">{ui.pill("bad", "Proposta fora das regras", "fa-gavel")}'
                            f'<p class="vx-note" style="margin-top:.6rem">{ui.esc(res["error"])}. Verificado por <b>{via}</b>.</p></div>',
                            unsafe_allow_html=True)
                return
            g1, g2 = st.columns([1, 1.2])
            g1.plotly_chart(charts.gauge(res["probability_default"], mode, cutoff=res["cutoff"]), config=charts.config(), width="stretch")
            kind = "good" if res["decision"] == "Aprovar" else "bad"
            band_kind = {"Baixo": "good", "Moderado": "warn", "Alto": "bad", "Muito alto": "bad"}[res["risk_band"]]
            g2.markdown(f"""<div style="padding-top:1.4rem">
<div style="display:flex;gap:.5rem;flex-wrap:wrap">{ui.pill(kind, res["decision"], "fa-gavel")}
{ui.pill(band_kind, "Risco " + res["risk_band"].lower(), "fa-signal")}</div>
<p class="vx-note" style="margin-top:.9rem">PD de <b>{ui.fmt_pct(res["probability_default"])}</b> contra corte de
<b>{ui.fmt_pct(res["cutoff"], 0)}</b>. Modelo <b>{ui.esc(res["model"])}</b> (<code>{ui.esc(res["model_id"])}</code>),
respondido por <b>{via}</b> em {ui.fmt_num(ms, 0)} ms. Cada pontuação é registrada no SQLite para auditoria.</p></div>""",
                        unsafe_allow_html=True)
