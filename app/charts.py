"""Gráficos Plotly com tema próprio (claro/escuro selecionados, não invertidos).

Paleta categórica validada para daltonismo (ordem fixa, nunca reciclada).
Adimplente = slot 1 (azul) · Inadimplente = slot 2 (laranja).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

CAT = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
DIVERGING = {"light": [[0, "#c0392f"], [0.5, "#f0efec"], [1, "#2a78d6"]],
             "dark": [[0, "#e66767"], [0.5, "#383835"], [1, "#3987e5"]]}
INK = {"light": ("#12213a", "#44536d", "#66758f", "rgba(18,33,58,.08)"),
       "dark": ("#e8eef8", "#b3c0d6", "#8796b0", "rgba(255,255,255,.07)")}
SURFACE = {"light": "#ffffff", "dark": "#18253f"}


def colors(mode: str) -> list[str]:
    return CAT[mode]


def style(fig: go.Figure, mode: str, height: int = 360, legend: bool = True, title: str | None = None) -> go.Figure:
    ink, ink2, ink3, grid = INK[mode]
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12.5, color=ink2),
        title=dict(text=title, font=dict(family="Sora, Inter", size=14, color=ink), x=0, xanchor="left") if title else None,
        margin=dict(l=8, r=12, t=40 if title else 14, b=8),
        colorway=CAT[mode],
        hoverlabel=dict(bgcolor=SURFACE[mode], bordercolor=grid, font=dict(color=ink, family="Inter", size=12.5)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left", bgcolor="rgba(0,0,0,0)",
                    font=dict(color=ink2, size=12), title=None),
        bargap=0.28, bargroupgap=0.08,
        separators=",.",
    )
    axis = dict(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont=dict(color=ink3, size=11.5),
                title_font=dict(color=ink2, size=12), automargin=True)
    fig.update_xaxes(**axis, showgrid=False)
    fig.update_yaxes(**axis, showgrid=True)
    return fig


def config() -> dict:
    return {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
            "toImageButtonOptions": {"format": "png", "scale": 2}}


# ------------------------------------------------------------------ fábricas
def bar_rate(df: pd.DataFrame, x: str, y: str, mode: str, horizontal: bool = False, height: int = 320,
             text_fmt=lambda v: f"{v * 100:.1f}%".replace(".", ","), color_idx: int = 1, hover_label: str = "Taxa") -> go.Figure:
    c = CAT[mode][color_idx]
    texts = [text_fmt(v) for v in df[y]]
    if horizontal:
        fig = go.Figure(go.Bar(y=df[x], x=df[y], orientation="h", marker=dict(color=c, cornerradius=4),
                               text=texts, textposition="outside", cliponaxis=False,
                               hovertemplate=f"%{{y}}<br>{hover_label}: %{{text}}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=True, showticklabels=False, range=[0, float(df[y].max()) * 1.18])
    else:
        fig = go.Figure(go.Bar(x=df[x], y=df[y], marker=dict(color=c, cornerradius=4), text=texts,
                               textposition="outside", cliponaxis=False,
                               hovertemplate=f"%{{x}}<br>{hover_label}: %{{text}}<extra></extra>"))
        fig.update_yaxes(showticklabels=False, range=[0, float(df[y].max()) * 1.15])
    fig.update_traces(textfont=dict(color=INK[mode][1], size=11.5))
    return style(fig, mode, height, legend=False)


def hist_by_target(df: pd.DataFrame, col: str, mode: str, nbins: int = 40, height: int = 320,
                   xtitle: str | None = None, normalize: bool = True) -> go.Figure:
    fig = go.Figure()
    labels = {0: "Adimplente", 1: "Inadimplente"}
    for k in (0, 1):
        fig.add_trace(go.Histogram(
            x=df.loc[df["loan_status"] == k, col], name=labels[k], nbinsx=nbins, opacity=0.78,
            histnorm="percent" if normalize else None, marker=dict(color=CAT[mode][k], line=dict(width=0)),
            hovertemplate=f"{labels[k]}<br>%{{x}}<br>%{{y:.1f}}% do grupo<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title="% do grupo" if normalize else "Registros")
    return style(fig, mode, height)


def heatmap(z: pd.DataFrame, mode: str, height: int = 420, diverging: bool = False, fmt: str = ".2f",
            zmid: float | None = None, colorbar_title: str = "") -> go.Figure:
    scale = DIVERGING[mode] if diverging else [[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)]
    text = np.vectorize(lambda v: "" if pd.isna(v) else format(v, fmt).replace(".", ","))(z.values)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=list(z.columns), y=list(z.index), colorscale=scale, zmid=zmid, text=text,
        texttemplate="%{text}", textfont=dict(size=11), xgap=2, ygap=2,
        colorbar=dict(title=dict(text=colorbar_title, font=dict(size=11)), thickness=10, outlinewidth=0,
                      tickfont=dict(color=INK[mode][2])),
        hovertemplate="%{y} × %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return style(fig, mode, height, legend=False)


def gauge(value: float, mode: str, title: str = "Probabilidade de default", cutoff: float = 0.3) -> go.Figure:
    ink, ink2, ink3, grid = INK[mode]
    bar_color = "#0ca30c" if value < 0.10 else "#fab219" if value < cutoff else "#d03b3b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value * 100,
        number=dict(suffix="%", valueformat=".1f", font=dict(family="Sora", size=40, color=ink)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=ink3, tickfont=dict(color=ink3), ticksuffix="%"),
                   bar=dict(color=bar_color, thickness=0.32), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, 100], color=grid)],
                   threshold=dict(line=dict(color=ink, width=2), thickness=0.8, value=cutoff * 100)),
        title=dict(text=title, font=dict(size=13, color=ink2))))
    return style(fig, mode, 260, legend=False)
