from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

import charts
import data
import ui
from vertice import context, services


def _n(v, d=3):
    return "—" if v is None or v != v else ui.fmt_num(v, d)


def render(mode: str) -> None:
    spec = context.spec()
    ui.page_header("fa-brain", "Fase 6 de 8", "Modelagem da probabilidade de default",
                   "Divisão estratificada 80/20, pré-processamento dentro de um <code>Pipeline</code> do scikit-learn, "
                   "validação cruzada de 5 folds e três candidatos: um baseline ingênuo, a Regressão Logística "
                   "(padrão do setor) e Gradient Boosting. O campeão é escolhido pela ROC-AUC média da validação cruzada.")
    grades = ui.grade_filter(spec)

    cv = data.t("model_metrics").set_index("model")
    preds = data.tg("test_predictions", grades)
    imp = data.t("feature_importance")
    if preds.empty:
        st.info("Nenhum registro de teste nas notas escolhidas.")
        return
    metrics = services.metrics_for(preds).set_index("model")
    metrics = metrics.join(cv[["cv_auc_mean", "cv_auc_std", "is_champion"]]).reset_index()
    champ = metrics.query("is_champion == 1").iloc[0]
    cols = charts.colors(mode)
    model_color = {"baseline": charts.INK[mode][2], "logistic": cols[0], "gbm": cols[2]}
    if grades:
        st.markdown(f'<div class="vx-note" style="margin:-.4rem 0 .9rem"><i class="fa-solid fa-circle-info"></i> Métricas de teste '
                    f'recalculadas só para as notas {", ".join(grades)}. A validação cruzada e a importância das variáveis são '
                    'da base inteira, porque o modelo foi treinado nela.</div>', unsafe_allow_html=True)

    ui.kpi_row([
        ui.kpi("fa-trophy", "Modelo campeão", f'<span style="font-size:1.25rem">{champ["label"]}</span>', f"CV AUC {ui.fmt_num(champ['cv_auc_mean'], 4)} ± {ui.fmt_num(champ['cv_auc_std'], 4)}"),
        ui.kpi("fa-chart-line", "ROC-AUC (teste)", _n(champ["roc_auc"]), f"Gini {_n(champ['gini'])}"),
        ui.kpi("fa-arrows-up-down", "KS", _n(champ["ks"]), "separação máxima bons × maus"),
        ui.kpi("fa-bullseye", "Brier score", ui.fmt_num(champ["brier"], 4), "erro quadrático da PD (menor é melhor)"),
    ])
    st.write("")

    with ui.glass("compare"):
        ui.section("fa-scale-unbalanced", "Comparação dos candidatos", f"{ui.fmt_int(len(preds))} registros no teste{ui.filter_note(grades)}")
        t = metrics.copy()
        t["campeão"] = t["is_champion"].map(lambda v: ui.pill("good", "campeão", "fa-trophy") if v else "")
        t["CV AUC"] = [f"{ui.fmt_num(m, 4)} ± {ui.fmt_num(s, 4)}" for m, s in zip(t["cv_auc_mean"], t["cv_auc_std"])]
        out = t[["label", "CV AUC", "roc_auc", "gini", "ks", "pr_auc", "brier", "f1", "campeão"]]
        out.columns = ["modelo", "CV AUC (treino)", "AUC teste", "Gini", "KS", "PR-AUC", "Brier", "F1 @0,5", "campeão"]
        ui.html_table(out, formats={c: (lambda v: _n(v, 4)) for c in ["AUC teste", "Gini", "KS", "PR-AUC", "Brier", "F1 @0,5"]},
                      raw_html=["campeão", "CV AUC (treino)"])

    st.write("")
    y = preds["y_true"].to_numpy()
    if len(set(y)) < 2:
        st.info("O recorte tem só uma classe no teste: curvas ROC, PR e calibração não se aplicam. Inclua mais notas.")
        return
    c1, c2 = st.columns(2)
    with c1:
        with ui.glass("roc"):
            ui.section("fa-chart-line", "Curva ROC")
            fig = go.Figure()
            for _, m in metrics.iterrows():
                fpr, tpr, _ = roc_curve(y, preds[f"p_{m['model']}"])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{m['label']} ({ui.fmt_num(m['roc_auc'], 3)})",
                                         line=dict(color=model_color[m["model"]], width=2, dash="dot" if m["model"] == "baseline" else None),
                                         hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>"))
            fig.update_xaxes(title="Taxa de falsos positivos", range=[0, 1], showgrid=True)
            fig.update_yaxes(title="Taxa de verdadeiros positivos", range=[0, 1.02])
            st.plotly_chart(charts.style(fig, mode, 360), config=charts.config(), width="stretch")
    with c2:
        with ui.glass("pr"):
            ui.section("fa-crosshairs", "Precisão × recall")
            fig = go.Figure()
            for _, m in metrics[metrics["model"] != "baseline"].iterrows():
                p, r, _ = precision_recall_curve(y, preds[f"p_{m['model']}"])
                fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=f"{m['label']} ({ui.fmt_num(m['pr_auc'], 3)})",
                                         line=dict(color=model_color[m["model"]], width=2),
                                         hovertemplate="Recall %{x:.3f}<br>Precisão %{y:.3f}<extra></extra>"))
            fig.add_hline(y=y.mean(), line=dict(color=charts.INK[mode][2], dash="dot", width=1),
                          annotation_text="taxa base", annotation_font=dict(size=10, color=charts.INK[mode][2]))
            fig.update_xaxes(title="Recall", range=[0, 1], showgrid=True)
            fig.update_yaxes(title="Precisão", range=[0, 1.02])
            st.plotly_chart(charts.style(fig, mode, 360), config=charts.config(), width="stretch")

    st.write("")
    with ui.glass("threshold"):
        ui.section("fa-sliders", "Ponto de corte e matriz de confusão", "modelo campeão")
        thr = st.slider("Classificar como inadimplente se PD ≥", 0.05, 0.95, 0.50, 0.01, key="thr")
        p = preds[f"p_{champ['model']}"].to_numpy()
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        c1, c2 = st.columns([1, 1.2])
        with c1:
            z = np.array([[tn, fp], [fn, tp]])
            labels = [["Verdadeiro negativo", "Falso positivo"], ["Falso negativo", "Verdadeiro positivo"]]
            text = [[f"{labels[i][j]}<br><b>{ui.fmt_int(z[i, j])}</b>" for j in range(2)] for i in range(2)]
            fig = go.Figure(go.Heatmap(z=z, x=["Previsto adimplente", "Previsto inadimplente"],
                                       y=["Real adimplente", "Real inadimplente"], text=text, texttemplate="%{text}",
                                       colorscale=[[0, charts.SEQ[0]], [1, charts.SEQ[5]]], showscale=False, xgap=3, ygap=3,
                                       textfont=dict(size=12.5), hovertemplate="%{y} · %{x}<br>%{z:,}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", showgrid=False)
            st.plotly_chart(charts.style(fig, mode, 300, legend=False), config=charts.config(), width="stretch")
        with c2:
            st.markdown("".join([
                f'<div class="vx-note" style="margin:.4rem 0"><b style="font-family:Sora;font-size:1.3rem">{ui.fmt_pct(prec)}</b>'
                f'&nbsp; precisão: dos pedidos marcados como risco, quantos inadimpliram</div>',
                f'<div class="vx-note" style="margin:.4rem 0"><b style="font-family:Sora;font-size:1.3rem">{ui.fmt_pct(rec)}</b>'
                f'&nbsp; recall: dos inadimplentes reais, quantos o modelo pegou</div>',
                f'<div class="vx-note" style="margin:.4rem 0"><b style="font-family:Sora;font-size:1.3rem">{ui.fmt_num(f1, 3)}</b>'
                f'&nbsp; F1, média harmônica das duas</div>',
                f'<div class="vx-note" style="margin:.4rem 0"><b style="font-family:Sora;font-size:1.3rem">{ui.fmt_pct(pred.mean())}</b>'
                f'&nbsp; dos pedidos seriam marcados como risco</div>',
            ]), unsafe_allow_html=True)
            ui.note("A escolha final do corte é uma decisão de negócio, feita na fase de Resultados com base em custo e receita.")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        with ui.glass("calib"):
            ui.section("fa-ruler", "Calibração da PD", "a diagonal é a calibração perfeita")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=charts.INK[mode][2], dash="dot", width=1)))
            for _, m in metrics[metrics["model"] != "baseline"].iterrows():
                fy, fx = calibration_curve(y, preds[f"p_{m['model']}"], n_bins=10, strategy="quantile")
                fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines+markers", name=m["label"],
                                         line=dict(color=model_color[m["model"]], width=2), marker=dict(size=8),
                                         hovertemplate="PD prevista %{x:.2f}<br>Default observado %{y:.2f}<extra></extra>"))
            fig.update_xaxes(title="PD prevista (média da faixa)", range=[0, 1], showgrid=True)
            fig.update_yaxes(title="Default observado", range=[0, 1])
            st.plotly_chart(charts.style(fig, mode, 340), config=charts.config(), width="stretch")
    with c2:
        with ui.glass("imp"):
            ui.section("fa-list-ol", "Importância por permutação", "queda de AUC ao embaralhar")
            d = imp.head(10).sort_values("importance")
            fig = go.Figure(go.Bar(y=d["feature"], x=d["importance"], orientation="h",
                                   error_x=dict(type="data", array=d["std"], color=charts.INK[mode][2], thickness=1),
                                   marker=dict(color=cols[0], cornerradius=4),
                                   hovertemplate="%{y}<br>Queda de AUC: %{x:.4f}<extra></extra>"))
            fig.update_xaxes(showgrid=True, title="Queda média de ROC-AUC")
            st.plotly_chart(charts.style(fig, mode, 340, legend=False), config=charts.config(), width="stretch")

    st.write("")
    with ui.glass("scores"):
        ui.section("fa-chart-area", "Distribuição da PD por desfecho real", "quanto menos sobreposição, melhor")
        d = preds.rename(columns={"y_true": "loan_status"})
        fig = charts.hist_by_target(d, f"p_{champ['model']}", mode, nbins=50, height=300, xtitle="PD prevista")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, config=charts.config(), width="stretch")
