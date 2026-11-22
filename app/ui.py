"""Sistema visual: tokens de tema, CSS glassmorphism, ícones Font Awesome e componentes."""
from __future__ import annotations

import html
from collections.abc import Iterable

import pandas as pd
import streamlit as st

THEME_OPTIONS = {"light": "Claro", "dark": "Escuro", "auto": "Auto"}

LIGHT = dict(
    bg="#eaf0f7", ink="#12213a", ink2="#44536d", ink3="#66758f",
    glass="rgba(255,255,255,.50)", glass_strong="rgba(255,255,255,.80)", glass_border="rgba(255,255,255,.80)",
    hairline="rgba(18,33,58,.10)", shadow="0 10px 30px -12px rgba(31,56,99,.22)",
    accent="#2a78d6", accent_soft="rgba(42,120,214,.12)",
    orb1="rgba(85,152,231,.55)", orb2="rgba(27,175,122,.38)", orb3="rgba(98,80,214,.34)",
    good="#067a06", good_bg="rgba(12,163,12,.12)", warn="#8a5a00", warn_bg="rgba(250,178,25,.18)",
    bad="#b42b2b", bad_bg="rgba(208,59,59,.12)", code_bg="rgba(18,33,58,.06)",
)
DARK = dict(
    bg="#0c1424", ink="#e8eef8", ink2="#b3c0d6", ink3="#8796b0",
    glass="rgba(24,37,63,.52)", glass_strong="rgba(28,43,72,.86)", glass_border="rgba(255,255,255,.09)",
    hairline="rgba(255,255,255,.08)", shadow="0 10px 30px -12px rgba(0,0,0,.6)",
    accent="#5598e7", accent_soft="rgba(85,152,231,.16)",
    orb1="rgba(28,92,171,.55)", orb2="rgba(18,107,79,.40)", orb3="rgba(75,58,160,.40)",
    good="#4fd14f", good_bg="rgba(12,163,12,.18)", warn="#fab219", warn_bg="rgba(250,178,25,.16)",
    bad="#f08080", bad_bg="rgba(208,59,59,.20)", code_bg="rgba(255,255,255,.06)",
)

# ícone Font Awesome de cada fase da navegação (mesma ordem de app.PAGES)
NAV_ICONS = ["\\f14e", "\\f56f", "\\f46c", "\\f51a", "\\f0d0", "\\e473", "\\f5dc", "\\f24e", "\\f1e6", "\\f83e"]


def resolve_mode(choice: str) -> str:
    """Converte a escolha do usuário em 'light'/'dark' para gráficos Plotly."""
    if choice in ("light", "dark"):
        return choice
    try:
        return st.context.theme.type or "light"
    except Exception:  # pragma: no cover
        return "light"


def _vars(t: dict) -> str:
    return ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in t.items())


def inject_css(choice: str) -> None:
    if choice == "light":
        root = f":root{{{_vars(LIGHT)};color-scheme:light}}"
    elif choice == "dark":
        root = f":root{{{_vars(DARK)};color-scheme:dark}}"
    else:
        root = (f":root{{{_vars(LIGHT)};color-scheme:light}}"
                f"@media (prefers-color-scheme: dark){{:root{{{_vars(DARK)};color-scheme:dark}}}}")

    nav_rules = "".join(
        f'.st-key-nav [role="radiogroup"] > div:nth-of-type({i + 1}) [data-testid="stMarkdownContainer"]::before'
        f'{{content:"{icon}"}}' for i, icon in enumerate(NAV_ICONS))

    st.markdown('<link rel="stylesheet" href="app/static/fontawesome/css/fontawesome.min.css">'
                '<link rel="stylesheet" href="app/static/fontawesome/css/solid.min.css">'
                '<link rel="stylesheet" href="app/static/fontawesome/css/regular.min.css">',
                unsafe_allow_html=True)
    css = f"""
{root}
html, body, .stApp {{ font-family: Inter, system-ui, sans-serif; font-feature-settings: "tnum" 1, "cv11" 1; }}
.stApp {{ background: var(--bg) !important; color: var(--ink); }}
.stApp::before {{
  content:""; position:fixed; inset:-20vmax; z-index:0; pointer-events:none;
  background:
    radial-gradient(38vmax 38vmax at 18% 12%, var(--orb1), transparent 62%),
    radial-gradient(34vmax 34vmax at 88% 22%, var(--orb3), transparent 60%),
    radial-gradient(40vmax 40vmax at 62% 96%, var(--orb2), transparent 62%);
  filter: blur(10px);
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stMain"], [data-testid="stAppViewContainer"] {{ background: transparent !important; position:relative; z-index:1; }}
.block-container {{ padding-top: 2.2rem !important; max-width: 1320px; }}
h1, h2, h3, h4, .vx-display {{ font-family: Sora, Inter, sans-serif !important; color: var(--ink) !important; letter-spacing: -0.02em; }}
p, li, label, span, div {{ color: inherit; }}
[data-testid="stMarkdownContainer"] p {{ color: var(--ink2); }}
a {{ color: var(--accent) !important; }}

/* ---------- sidebar de vidro ---------- */
[data-testid="stSidebar"] {{
  background: var(--glass) !important; backdrop-filter: blur(22px) saturate(150%);
  -webkit-backdrop-filter: blur(22px) saturate(150%); border-right: 1px solid var(--glass-border) !important;
}}
[data-testid="stSidebar"] * {{ color: var(--ink); }}
[data-testid="stSidebarHeader"] {{ height: .5rem; min-height: .5rem; padding: 0; }}
.vx-brand {{ display:flex; gap:.7rem; align-items:center; margin:.2rem 0 1.1rem; }}
.vx-brand .mark {{ width:38px; height:38px; border-radius:11px; display:grid; place-items:center;
  background: linear-gradient(140deg, var(--accent), #1baf7a); color:#fff; font-size:1rem;
  box-shadow: 0 6px 18px -6px var(--accent); }}
.vx-brand .name {{ font-family:Sora; font-weight:700; font-size:1.02rem; line-height:1.1; color:var(--ink); }}
.vx-brand .sub {{ font-size:.75rem; color:var(--ink3); }}
.vx-side-label {{ font-size:.74rem; color:var(--ink3); margin:.9rem 0 .35rem; font-weight:600; }}

/* trilho da pipeline: cada fase é um nó conectado por uma linha */
.st-key-nav [role="radiogroup"] {{ gap: 0 !important; position:relative; }}
.st-key-nav [role="radiogroup"] > div {{
  position:relative; margin:0 !important; padding:.42rem .6rem .42rem 2.65rem !important; border-radius:10px;
  width:100%; cursor:pointer; transition: background .15s ease;
}}
.st-key-nav [role="radiogroup"] > div {{ display:block; width:100%; }}
.st-key-nav [role="radiogroup"] > div label {{ width:100%; padding:0 !important; margin:0 !important; cursor:pointer; }}
.st-key-nav [role="radiogroup"] > div label > div > div:first-child {{ display:none !important; }}
.st-key-nav [role="radiogroup"] > div:hover {{ background: var(--accent-soft); }}
.st-key-nav [role="radiogroup"] > div::after {{
  content:""; position:absolute; left:1.22rem; top:0; bottom:0; width:2px; background: var(--hairline);
}}
.st-key-nav [role="radiogroup"] > div:first-of-type::after {{ top:50%; }}
.st-key-nav [role="radiogroup"] > div:last-of-type::after {{ bottom:50%; }}
.st-key-nav [role="radiogroup"] > div [data-testid="stMarkdownContainer"] p {{ color:var(--ink2); font-size:.9rem; margin:0; }}
.st-key-nav [role="radiogroup"] > div [data-testid="stMarkdownContainer"]::before {{
  font-family:"Font Awesome 6 Free"; font-weight:900; position:absolute; left:.52rem; top:50%;
  transform:translateY(-50%); width:1.45rem; height:1.45rem; border-radius:50%; display:grid; place-items:center;
  font-size:.66rem; background: var(--glass-strong); border:1px solid var(--hairline); color:var(--ink3); z-index:2;
}}
{nav_rules}
.st-key-nav [role="radiogroup"] > div[data-selected="true"] {{ background: var(--accent-soft); }}
.st-key-nav [role="radiogroup"] > div[data-selected="true"] [data-testid="stMarkdownContainer"] p {{ color:var(--ink); font-weight:600; }}
.st-key-nav [role="radiogroup"] > div[data-selected="true"] [data-testid="stMarkdownContainer"]::before {{
  background: var(--accent); color:#fff; border-color: transparent; box-shadow: 0 0 0 4px var(--accent-soft);
}}

/* seletor de tema com ícones */
.st-key-theme [data-baseweb="button-group"] button p::before,
.st-key-theme button [data-testid="stMarkdownContainer"] p::before {{ font-family:"Font Awesome 6 Free"; font-weight:900; margin-right:.4rem; }}
.st-key-theme [role="radiogroup"] {{ flex-wrap:nowrap !important; width:100%; }}
.st-key-theme button[data-variant="segmented_control"] {{ flex:1 1 0; min-width:0; padding-left:.3rem !important; padding-right:.3rem !important;
  background: var(--glass-strong) !important; border-color: var(--hairline) !important; }}
.st-key-theme button[data-variant="segmented_control"] p {{ color: var(--ink2) !important; font-size:.84rem; }}
.st-key-theme button[aria-checked="true"] {{ background: var(--accent) !important; border-color: transparent !important; }}
.st-key-theme button[aria-checked="true"] p {{ color:#fff !important; }}
button[data-variant="segmented_control"] {{ background: var(--glass-strong) !important; border-color: var(--hairline) !important; }}
button[data-variant="segmented_control"] p {{ color: var(--ink2) !important; }}
button[data-variant="segmented_control"][aria-checked="true"] {{ background: var(--accent) !important; border-color: transparent !important; }}
button[data-variant="segmented_control"][aria-checked="true"] p {{ color:#fff !important; }}
.st-key-theme button:nth-of-type(1) p::before {{ content:"\\f185"; }}
.st-key-theme button:nth-of-type(2) p::before {{ content:"\\f186"; }}
.st-key-theme button:nth-of-type(3) p::before {{ content:"\\f042"; }}

/* ---------- vidro ---------- */
.vx-glass, [class*="st-key-glass"] {{
  background: var(--glass) !important; border:1px solid var(--glass-border) !important; border-radius:18px !important;
  backdrop-filter: blur(18px) saturate(140%); -webkit-backdrop-filter: blur(18px) saturate(140%);
  box-shadow: var(--shadow);
}}
[class*="st-key-glass"] {{ padding: 1.1rem 1.2rem 1rem !important; }}
.vx-glass {{ padding: 1.05rem 1.2rem; }}

/* ---------- cabeçalho da página ---------- */
.vx-page-head {{ display:flex; gap:1rem; align-items:flex-start; margin: 0 0 1.4rem; }}
.vx-page-head .ico {{ width:52px; height:52px; flex:0 0 52px; border-radius:15px; display:grid; place-items:center;
  font-size:1.25rem; color:var(--accent); background: var(--glass-strong); border:1px solid var(--glass-border);
  box-shadow: var(--shadow); }}
.vx-page-head .step {{ font-size:.8rem; color:var(--ink3); font-weight:600; margin-bottom:.15rem; }}
.vx-page-head h1 {{ font-size:1.85rem !important; margin:0 !important; padding:0 !important; line-height:1.15; }}
.vx-page-head p {{ margin:.35rem 0 0; color:var(--ink2); max-width:72ch; font-size:.96rem; }}

.vx-section {{ display:flex; align-items:center; gap:.55rem; margin:.1rem 0 .7rem; }}
.vx-section i {{ color: var(--accent); font-size:.9rem; }}
.vx-section h3 {{ font-size:1.02rem !important; margin:0 !important; padding:0 !important; font-weight:600; }}
.vx-section .hint {{ margin-left:auto; font-size:.78rem; color:var(--ink3); }}
.vx-note {{ font-size:.86rem; color:var(--ink2); line-height:1.55; margin:.2rem 0 0; }}
.vx-note b {{ color: var(--ink); }}

/* ---------- KPIs ---------- */
.vx-kpi {{ height:100%; }}
.vx-kpi .top {{ display:flex; align-items:center; gap:.5rem; color:var(--ink3); font-size:.82rem; font-weight:500; }}
.vx-kpi .top i {{ width:26px; height:26px; border-radius:8px; display:grid; place-items:center; font-size:.72rem;
  background: var(--accent-soft); color: var(--accent); }}
.vx-kpi .val {{ font-family:Sora; font-weight:650; font-size:1.75rem; color:var(--ink); margin:.45rem 0 .1rem; letter-spacing:-.02em; }}
.vx-kpi .foot {{ font-size:.8rem; color:var(--ink3); }}
.vx-kpi .foot .up {{ color: var(--good); font-weight:600; }}
.vx-kpi .foot .down {{ color: var(--bad); font-weight:600; }}

/* ---------- pills de status (ícone + rótulo, nunca só cor) ---------- */
.vx-pill {{ display:inline-flex; align-items:center; gap:.35rem; padding:.16rem .55rem; border-radius:999px;
  font-size:.76rem; font-weight:600; white-space:nowrap; }}
.vx-pill.good {{ color:var(--good); background:var(--good-bg); }}
.vx-pill.warn {{ color:var(--warn); background:var(--warn-bg); }}
.vx-pill.bad {{ color:var(--bad); background:var(--bad-bg); }}
.vx-pill.neutral {{ color:var(--ink2); background:var(--accent-soft); }}

/* ---------- tabelas ---------- */
.vx-table-wrap {{ overflow:auto; max-height: 460px; border-radius:12px; border:1px solid var(--hairline); }}
table.vx-table {{ width:100%; border-collapse:collapse; font-size:.84rem; }}
table.vx-table th {{ position:sticky; top:0; background: var(--glass-strong); backdrop-filter: blur(12px);
  text-align:left; font-weight:600; color:var(--ink2); padding:.55rem .7rem; border-bottom:1px solid var(--hairline); white-space:nowrap; }}
table.vx-table td {{ padding:.5rem .7rem; border-bottom:1px solid var(--hairline); color:var(--ink); white-space:normal; vertical-align:middle; }}
table.vx-table td.num, table.vx-table td .vx-pill, table.vx-table td code {{ white-space:nowrap; }}
table.vx-table td code {{ font-size:.8rem; padding:.1rem .35rem; border-radius:6px; }}
table.vx-table td.num, table.vx-table th.num {{ text-align:right; font-variant-numeric: tabular-nums; }}
table.vx-table tr:last-child td {{ border-bottom:none; }}
table.vx-table tr:hover td {{ background: var(--accent-soft); }}

/* ---------- fluxo da pipeline ---------- */
.vx-flow {{ display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; }}
.vx-flow .node {{ display:flex; align-items:center; gap:.5rem; padding:.5rem .75rem; border-radius:12px;
  background: var(--glass-strong); border:1px solid var(--glass-border); font-size:.84rem; color:var(--ink); }}
.vx-flow .node i {{ color: var(--accent); }}
.vx-flow .node small {{ color: var(--ink3); margin-left:.2rem; }}
.vx-flow .sep {{ color: var(--ink3); font-size:.7rem; }}

/* timeline da limpeza */
.vx-timeline {{ position:relative; margin-left:.4rem; }}
.vx-timeline .item {{ position:relative; padding: 0 0 1rem 2rem; }}
.vx-timeline .item::before {{ content:""; position:absolute; left:.62rem; top:1.4rem; bottom:0; width:2px; background:var(--hairline); }}
.vx-timeline .item:last-child::before {{ display:none; }}
.vx-timeline .dot {{ position:absolute; left:0; top:.1rem; width:1.35rem; height:1.35rem; border-radius:50%;
  background:var(--accent-soft); color:var(--accent); font-size:.68rem; font-weight:700; display:grid; place-items:center; }}
.vx-timeline .title {{ font-weight:600; color:var(--ink); font-size:.92rem; }}
.vx-timeline .desc {{ color:var(--ink2); font-size:.84rem; margin-top:.1rem; }}
.vx-timeline .meta {{ margin-top:.3rem; display:flex; gap:.4rem; flex-wrap:wrap; }}

/* ---------- widgets nativos sob tema forçado ---------- */
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"],
[data-testid="stNumberInput"] [role="group"], [data-testid="stNumberInput"] > div > div, [data-testid="stTextInput"] [role="group"] {{
  background: var(--glass-strong) !important; color: var(--ink) !important; border-color: var(--hairline) !important; }}
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {{
  color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; background: transparent !important; }}
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg, [data-testid="stNumberInput"] button {{ color: var(--ink2) !important; fill: currentColor; }}
[data-testid="stNumberInput"] button {{ background: transparent !important; }}
[data-testid="stMultiSelect"] [role="group"] span, [data-testid="stMultiSelect"] [role="row"], [data-testid="stMultiSelect"] [role="gridcell"] {{ color: var(--ink) !important; }}
[role="listbox"], [role="dialog"]:has([role="listbox"]), [data-testid="stSelectboxVirtualDropdown"] {{ background: var(--glass-strong) !important; backdrop-filter: blur(16px); color: var(--ink) !important; }}
[role="option"] {{ color: var(--ink) !important; }}
[role="option"]:hover, [role="option"][data-focused="true"] {{ background: var(--accent-soft) !important; }}
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="input"] input, [data-testid="stNumberInput"] input,
[data-baseweb="base-input"] {{ background: var(--glass-strong) !important; color: var(--ink) !important; border-color: var(--hairline) !important; }}
[data-baseweb="select"] span, [data-baseweb="tag"] span {{ color: var(--ink) !important; }}
[data-baseweb="popover"] ul, [data-baseweb="menu"] {{ background: var(--glass-strong) !important; backdrop-filter: blur(16px); }}
[data-baseweb="popover"] li {{ color: var(--ink) !important; }}
[data-testid="stWidgetLabel"] p {{ color: var(--ink2) !important; font-size:.84rem !important; font-weight:500; }}
.stButton > button, [data-testid="stFormSubmitButton"] > button, [data-testid="stBaseButton-segmented_control"],
[data-testid="stBaseButton-segmented_controlActive"] {{
  background: var(--glass-strong) !important; color: var(--ink) !important; border:1px solid var(--hairline) !important; }}
[data-testid="stBaseButton-segmented_controlActive"] {{ background: var(--accent) !important; color:#fff !important; border-color:transparent !important; }}
[data-testid="stBaseButton-segmented_controlActive"] p {{ color:#fff !important; }}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button {{ background: var(--accent) !important; color:#fff !important; border-color:transparent !important; }}
.stButton > button[kind="primary"] p, [data-testid="stFormSubmitButton"] > button p {{ color:#fff !important; }}
[data-testid="stTabs"] button p {{ color: var(--ink2) !important; }}
[data-testid="stTabs"] button[aria-selected="true"] p {{ color: var(--ink) !important; font-weight:600; }}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{ color: var(--ink2) !important; }}
[data-testid="stExpander"] details {{ background: var(--glass) !important; border:1px solid var(--glass-border) !important; border-radius:14px !important; }}
[data-testid="stExpander"] summary p {{ color: var(--ink) !important; }}
pre, code {{ background: var(--code-bg) !important; color: var(--ink) !important; }}
[data-testid="stCode"] pre {{ border-radius:12px !important; }}
[data-testid="stPlotlyChart"] {{ border-radius:12px; }}
[data-testid="stAlert"] {{ background: var(--glass) !important; border:1px solid var(--glass-border); border-radius:14px; }}
[data-testid="stAlert"] p {{ color: var(--ink) !important; }}
:focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 2px; }}
/* multiselect: "Select all" traduzido */
[data-key="__select_all__"] > div {{ font-size:0 !important; }}
[data-key="__select_all__"] > div::after {{ content:"Selecionar todas"; font-size:.88rem; color: var(--ink); }}

/* pills (filtro por nota) sob tema forçado */
button[data-variant="pills"] {{ background: var(--glass-strong) !important; border:1px solid var(--hairline) !important;
  min-width:2.6rem; justify-content:center; }}
button[data-variant="pills"] p {{ color: var(--ink2) !important; font-weight:600; }}
button[data-variant="pills"]:hover {{ border-color: var(--accent) !important; }}
button[data-variant="pills"][aria-pressed="true"] {{ background: var(--accent) !important; border-color: transparent !important; }}
button[data-variant="pills"][aria-pressed="true"] p {{ color:#fff !important; }}
.st-key-glass_gradefilter {{ padding:.7rem 1.1rem !important; margin-bottom:1.1rem; }}
.st-key-glass_gradefilter [data-testid="stHorizontalBlock"] {{ align-items:center; }}
.st-key-glass_gradefilter .stButton > button {{ min-height:2.1rem; }}
.vx-filter-label {{ display:flex; align-items:center; gap:.5rem; font-weight:600; color:var(--ink); font-size:.9rem; white-space:nowrap; }}
.vx-filter-label i {{ color: var(--accent); }}
.vx-filter-hint {{ font-size:.78rem; color:var(--ink3); white-space:nowrap; }}

/* seletor de base */
.st-key-dataset [role="radiogroup"] {{ flex-wrap:nowrap !important; width:100%; }}
.st-key-dataset button[data-variant="segmented_control"] {{ flex:1 1 0; min-width:0; }}
.st-key-dataset button p::before {{ font-family:"Font Awesome 6 Free"; font-weight:900; margin-right:.4rem; }}
.st-key-dataset button:nth-of-type(1) p::before {{ content:"\\24"; }}
.st-key-dataset button:nth-of-type(2) p::before {{ content:"\\e46c"; }}

/* camada de carregamento: aparece quando o Streamlit está processando por mais de 350 ms */
[data-testid="stElementContainer"]:has(.vx-loader) {{ position:fixed !important; inset:0; z-index:999990; opacity:1 !important;
  pointer-events:none; margin:0 !important; height:0; }}
.vx-loader {{ position:fixed; inset:0; z-index:999990; display:grid; place-items:center; pointer-events:none;
  opacity:0; visibility:hidden; transition: opacity .2s ease, visibility 0s linear .2s;
  background: color-mix(in srgb, var(--bg) 45%, transparent); backdrop-filter: blur(3px); -webkit-backdrop-filter: blur(3px); }}
[data-test-script-state="running"] .vx-loader {{ opacity:1; visibility:visible; pointer-events:auto;
  transition: opacity .25s ease .35s, visibility 0s linear .35s; }}
.stApp:has(.st-key-pipeline_progress) .vx-loader {{ display:none; }}
.vx-loader .card {{ display:flex; align-items:center; gap:.9rem; padding:1rem 1.3rem; border-radius:16px;
  background: var(--glass-strong); border:1px solid var(--glass-border); box-shadow: var(--shadow);
  backdrop-filter: blur(18px) saturate(140%); }}
.vx-loader .spin {{ width:30px; height:30px; border-radius:50%; border:3px solid var(--accent-soft);
  border-top-color: var(--accent); animation: vx-spin .8s linear infinite; }}
.vx-loader b {{ display:block; font-family:Sora; color:var(--ink); font-size:.95rem; }}
.vx-loader small {{ color:var(--ink3); font-size:.8rem; }}
@keyframes vx-spin {{ to {{ transform: rotate(360deg); }} }}

@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
"""
    st.html(f"<style>{css}</style>")
    st.markdown('<div class="vx-loader" role="status" aria-live="polite"><div class="card"><span class="spin"></span>'
                '<div><b>Carregando</b><small>processando dados e gráficos…</small></div></div></div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- componentes
def grade_filter(spec) -> list[str]:
    """Filtro global por nota/rating. Vale para todos os gráficos de todas as páginas."""
    key = f"grades_{spec.id}"

    def _clear():
        st.session_state[key] = []

    with st.container(key="glass_gradefilter"):
        c1, c2, c3 = st.columns([1.25, 5, 1.35], vertical_alignment="center")
        label = "Nota" if spec.id == "us" else "Rating"
        c1.markdown(f'<div class="vx-filter-label"><i class="fa-solid fa-filter"></i>Filtrar por {label.lower()}</div>',
                    unsafe_allow_html=True)
        sel = c2.pills(label, list(spec.grade_order), selection_mode="multi", key=key, label_visibility="collapsed")
        n = len(sel or [])
        c3.button("Todas as notas" if n else "Mostrando todas", on_click=_clear, disabled=not n, key=f"{key}_clear",
                  icon=":material/filter_alt_off:" if n else None, width="stretch")
    return list(sel or [])


def filter_note(grades: list[str]) -> str:
    return f" · notas {', '.join(grades)}" if grades else ""



def esc(text) -> str:
    return html.escape(str(text))


def page_header(icon: str, step: str, title: str, desc: str) -> None:
    st.markdown(f"""<div class="vx-page-head"><div class="ico"><i class="fa-solid {icon}"></i></div>
<div><div class="step">{esc(step)}</div><h1>{esc(title)}</h1><p>{desc}</p></div></div>""", unsafe_allow_html=True)


def section(icon: str, title: str, hint: str = "") -> None:
    hint_html = f'<span class="hint">{esc(hint)}</span>' if hint else ""
    st.markdown(f'<div class="vx-section"><i class="fa-solid {icon}"></i><h3>{esc(title)}</h3>{hint_html}</div>',
                unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<p class="vx-note">{text}</p>', unsafe_allow_html=True)


def kpi(icon: str, label: str, value: str, foot: str = "") -> str:
    return (f'<div class="vx-glass vx-kpi"><div class="top"><i class="fa-solid {icon}"></i>{esc(label)}</div>'
            f'<div class="val">{value}</div><div class="foot">{foot}</div></div>')


def kpi_row(items: Iterable[str]) -> None:
    items = list(items)
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(item, unsafe_allow_html=True)


def pill(kind: str, label: str, icon: str | None = None) -> str:
    icon = icon or {"good": "fa-circle-check", "warn": "fa-triangle-exclamation",
                    "bad": "fa-circle-xmark", "neutral": "fa-circle-info"}[kind]
    return f'<span class="vx-pill {kind}"><i class="fa-solid {icon}"></i>{esc(label)}</span>'


def glass(key: str):
    return st.container(key=f"glass_{key}")


def html_table(df: pd.DataFrame, formats: dict | None = None, raw_html: Iterable[str] = (),
               max_height: int = 460) -> None:
    formats = formats or {}
    raw_html = set(raw_html)
    head = "".join(
        f'<th class="{"num" if pd.api.types.is_numeric_dtype(df[c]) and c not in raw_html else ""}">{esc(c)}</th>'
        for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            is_num = pd.api.types.is_numeric_dtype(df[c]) and c not in raw_html
            if c in raw_html:
                txt = str(v)
            elif pd.isna(v):
                txt = '<span style="color:var(--ink3)">—</span>'
            elif c in formats:
                txt = esc(formats[c](v))
            else:
                txt = esc(v)
            cells.append(f'<td class="{"num" if is_num else ""}">{txt}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(f'<div class="vx-table-wrap" style="max-height:{max_height}px"><table class="vx-table">'
                f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- formatação pt-BR
def fmt_int(v) -> str:
    return f"{int(v):,}".replace(",", ".")


def fmt_pct(v, digits: int = 1) -> str:
    return f"{v * 100:,.{digits}f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_num(v, digits: int = 2) -> str:
    return f"{v:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_usd(v, digits: int = 0) -> str:
    return "US$ " + fmt_num(v, digits)


def fmt_compact_usd(v) -> str:
    for div, suf in ((1e9, " bi"), (1e6, " mi"), (1e3, " mil")):
        if abs(v) >= div:
            return f"US$ {fmt_num(v / div, 1)}{suf}"
    return fmt_usd(v)
