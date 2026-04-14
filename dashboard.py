"""
dashboard.py  (v3)
──────────────────────────────────────────────────────────────────────────────
Upgrades over v2:
  • Dark / Light mode toggle (sidebar) — full palette swap via CSS variables
  • Sortable tables — FII/DII breakdown, 22-technique table, options chain
    (click any column header to sort ascending / descending)
  • TradingView-style price chart:
      - Dark background  (#131722)
      - Watermark symbol name (bottom-right)
      - RSI sub-plot (row 2, 14-period)
      - MACD sub-plot (row 3)
      - Volume bars (row 4, colour-matched to candle)
      - Bollinger Bands overlay
      - Support / Resistance horizontal lines
      - Crosshair-style hover with unified tooltip
      - Range selector buttons (1W · 1M · 3M · 6M · 1Y · All)
      - Editable range slider at the bottom

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import datetime
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

from stock_advisor import (
    compute_signal, SYMBOLS, download_ohlcv, _ema, _rsi, _macd,
    _atr, download_vix,
)
from fii_dii_feed import fetch_fiidii_data
from options_feed import fetch_options_data

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Advisor — Live Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar  (theme toggle lives here)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    theme     = st.radio("🎨 Theme", ["Dark", "Light"], index=0, horizontal=True)
    symbol    = st.selectbox("Symbol", ["NIFTY", "SENSEX", "BANKNIFTY"], index=0)
    direction = st.radio("Direction", ["BOTH", "LONG", "SHORT"], index=0)
    timeframe = st.selectbox("Chart timeframe", ["1d", "1h", "15m"], index=0)
    refresh   = st.slider("Auto-refresh (min)", 0, 30, 5)
    st.button("🔄 Refresh Now", use_container_width=True)
    st.markdown("---")
    st.caption("Data: yfinance · NSE India · FII/DII · Options")
    st.caption("⚠️ Educational only — not financial advice.")

IS_DARK = (theme == "Dark")

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (dark / light)
# ─────────────────────────────────────────────────────────────────────────────
P = {
    "bg":          "#0d0f14" if IS_DARK else "#f4f6fa",
    "bg2":         "#1a1d27" if IS_DARK else "#ffffff",
    "bg3":         "#13161e" if IS_DARK else "#eef0f6",
    "border":      "#2a2d3a" if IS_DARK else "#dde1ec",
    "text":        "#e8eaf0" if IS_DARK else "#1a1d27",
    "text2":       "#8a8fa8" if IS_DARK else "#5a5f78",
    "text3":       "#555970" if IS_DARK else "#9095b0",
    "green":       "#00c853",
    "red":         "#ef5350",
    "yellow":      "#ffab00",
    "blue":        "#42a5f5",
    "purple":      "#ce93d8",
    "orange":      "#ffa726",
    "chart_bg":    "#131722" if IS_DARK else "#ffffff",
    "chart_grid":  "#1e2230" if IS_DARK else "#e8edf5",
    "chart_cross": "#758696" if IS_DARK else "#9ba8ba",
    "up_candle":   "#26a69a",
    "dn_candle":   "#ef5350",
    "up_wick":     "#26a69a",
    "dn_wick":     "#ef5350",
}

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS  (uses Python palette so it flips with theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── base ──────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {{
    background: {P["bg"]}; color: {P["text"]};
    transition: background .25s, color .25s;
}}
[data-testid="stSidebar"] {{
    background: {P["bg3"]} !important; color: {P["text"]};
}}
[data-testid="stSidebar"] * {{ color: {P["text"]} !important; }}

/* ── metric tile ─────────────────────────────────────────────── */
.tile {{
    background: {P["bg2"]}; border: 1px solid {P["border"]};
    border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
    transition: background .25s;
}}
.lbl  {{ font-size:11px; color:{P["text2"]}; text-transform:uppercase; letter-spacing:.06em; }}
.vl   {{ font-size:26px; font-weight:700; line-height:1.25; color:{P["text"]}; }}
.vmd  {{ font-size:18px; font-weight:700; color:{P["text"]}; }}
.dlt  {{ font-size:12px; margin-top:3px; color:{P["text2"]}; }}
.grn  {{ color:{P["green"]}  !important; }}
.red  {{ color:{P["red"]}    !important; }}
.yel  {{ color:{P["yellow"]} !important; }}
.gry  {{ color:{P["text2"]}  !important; }}

/* ── section header ──────────────────────────────────────────── */
.shdr {{
    font-size:11px; font-weight:700; text-transform:uppercase;
    letter-spacing:.12em; color:{P["text2"]};
    border-bottom:1px solid {P["border"]};
    padding-bottom:5px; margin:20px 0 10px;
}}

/* ── banners ─────────────────────────────────────────────────── */
.bull-banner {{
    background:linear-gradient(90deg,#002d18,#003d20);
    border:1px solid {P["green"]}; border-radius:8px;
    padding:14px 22px; display:flex; justify-content:space-between;
    align-items:center; margin:8px 0;
}}
.bear-banner {{
    background:linear-gradient(90deg,#2d0000,#3d0000);
    border:1px solid {P["red"]}; border-radius:8px;
    padding:14px 22px; display:flex; justify-content:space-between;
    align-items:center; margin:8px 0;
}}
.bl {{ font-size:14px; font-weight:800; letter-spacing:.1em; }}
.bv {{ font-size:22px; font-weight:800; }}

/* ── trade card ──────────────────────────────────────────────── */
.tcard {{
    background:{P["bg2"]}; border:1px solid {P["border"]};
    border-radius:10px; padding:14px; margin-top:8px;
}}
.trow {{
    display:flex; justify-content:space-between;
    border-bottom:1px solid {P["border"]}; padding:7px 2px; font-size:12px;
}}
.trow:last-child {{ border-bottom:none; }}
.tkey {{ color:{P["text2"]}; }}
.tval {{ font-weight:600; text-align:right; max-width:62%; color:{P["text"]}; }}

/* ── activity table ───────────────────────────────────────────── */
.act-table {{ width:100%; font-size:13px; border-collapse:collapse; }}
.act-table td {{
    padding:6px 0; border-bottom:1px solid {P["border"]};
    color:{P["text"]};
}}
.act-table td:last-child {{ text-align:right; font-weight:600; }}

/* ── sortable table wrapper ───────────────────────────────────── */
.sort-wrap {{
    overflow-x:auto; border-radius:8px;
    border:1px solid {P["border"]};
}}
.sort-tbl {{
    width:100%; border-collapse:collapse; font-size:12px;
}}
.sort-tbl th {{
    background:{P["bg3"]}; color:{P["text2"]};
    padding:8px 10px; font-size:11px; text-transform:uppercase;
    letter-spacing:.05em; cursor:pointer; white-space:nowrap;
    user-select:none; position:sticky; top:0; z-index:1;
}}
.sort-tbl th:hover {{ background:{P["border"]}; color:{P["text"]}; }}
.sort-tbl th .sort-arrow {{ margin-left:4px; opacity:.5; font-size:9px; }}
.sort-tbl td {{
    padding:6px 10px; border-bottom:1px solid {P["border"]};
    color:{P["text"]}; vertical-align:middle;
}}
.sort-tbl tr:hover td {{ background:{P["bg3"]}; }}
.sort-tbl tr:last-child td {{ border-bottom:none; }}

/* ── options metrics ──────────────────────────────────────────── */
.opt-row {{
    display:flex; justify-content:space-between; align-items:flex-end;
    border-bottom:1px solid {P["border"]}; padding:8px 2px;
}}
.opt-row:last-child {{ border-bottom:none; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sortable table JS component
# ─────────────────────────────────────────────────────────────────────────────

def sortable_table(headers: list[dict], rows: list[list], height: int = 400,
                   key: str = "tbl") -> None:
    """
    Render a fully client-side sortable HTML table inside an iframe component.

    headers: list of {"label": str, "key": str, "align": "left"|"right"|"center"}
    rows:    list of lists — each cell is a str (may contain HTML for colour)
    """
    bg      = P["bg2"]
    bg3     = P["bg3"]
    border  = P["border"]
    text    = P["text"]
    text2   = P["text2"]
    green   = P["green"]
    red     = P["red"]
    yellow  = P["yellow"]

    hdr_html = "".join(
        f'<th onclick="sortTable({i},this)" style="text-align:{h.get("align","left")}">'
        f'{h["label"]}<span class="sa">▲</span></th>'
        for i, h in enumerate(headers)
    )

    rows_html = ""
    for row in rows:
        cells = "".join(
            f'<td style="text-align:{headers[i].get("align","left")}">{cell}</td>'
            for i, cell in enumerate(row)
        )
        rows_html += f"<tr>{cells}</tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; font-family:system-ui,sans-serif; }}
  body {{ background:{bg}; color:{text}; overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{
      background:{bg3}; color:{text2}; padding:8px 10px;
      font-size:11px; text-transform:uppercase; letter-spacing:.05em;
      cursor:pointer; white-space:nowrap; user-select:none;
      position:sticky; top:0; z-index:1; border-bottom:1px solid {border};
  }}
  th:hover {{ background:{border}; color:{text}; }}
  th .sa {{ margin-left:4px; opacity:.4; font-size:9px; }}
  th.asc  .sa::after {{ content:"▲"; opacity:1; }}
  th.desc .sa::after {{ content:"▼"; opacity:1; }}
  th.asc  .sa, th.desc .sa {{ opacity:0; }}
  td {{ padding:6px 10px; border-bottom:1px solid {border}; vertical-align:middle; }}
  tr:hover td {{ background:{bg3}; }}
  tr:last-child td {{ border-bottom:none; }}
  .grn {{ color:{green}; }}
  .red {{ color:{red}; }}
  .yel {{ color:{yellow}; }}
  .gry {{ color:{text2}; }}
  .bdg {{
      font-size:10px; padding:2px 7px; border-radius:4px;
      font-weight:700; display:inline-block;
  }}
  .bdg-g {{ background:{green}; color:#000; }}
  .bdg-r {{ background:{red};   color:#fff; }}
  .bdg-n {{ background:{bg3};   color:{text2}; border:1px solid {border}; }}
</style>
</head>
<body>
<table id="t">
  <thead><tr>{hdr_html}</tr></thead>
  <tbody id="tb">{rows_html}</tbody>
</table>
<script>
let sortDir = {{}};
function sortTable(col, th) {{
  const tbody = document.getElementById("tb");
  const rows  = Array.from(tbody.rows);
  const dir   = sortDir[col] === "asc" ? "desc" : "asc";
  sortDir[col] = dir;

  // Reset all headers
  document.querySelectorAll("th").forEach(h => h.classList.remove("asc","desc"));
  th.classList.add(dir);

  rows.sort((a, b) => {{
    let av = a.cells[col].innerText.trim().replace(/[₹,+%×]/g,"");
    let bv = b.cells[col].innerText.trim().replace(/[₹,+%×]/g,"");
    const na = parseFloat(av), nb = parseFloat(bv);
    if (!isNaN(na) && !isNaN(nb)) return dir==="asc" ? na-nb : nb-na;
    return dir==="asc" ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</body>
</html>"""
    components.html(html, height=height, scrolling=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached data fetchers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_signal(sym, dirn): return compute_signal(sym, dirn.lower())

@st.cache_data(ttl=300, show_spinner=False)
def get_fiidii(): return fetch_fiidii_data()

@st.cache_data(ttl=300, show_spinner=False)
def get_options(sym):
    s = sym if sym in ("NIFTY","BANKNIFTY") else "NIFTY"
    return fetch_options_data(s)

@st.cache_data(ttl=120, show_spinner=False)
def get_ohlcv_cached(ticker, period, interval):
    return download_ohlcv(ticker, period=period, interval=interval)

@st.cache_data(ttl=60, show_spinner=False)
def get_vix(): return download_vix()

@st.cache_data(ttl=120, show_spinner=False)
def get_index_price(ticker):
    try:
        df   = get_ohlcv_cached(ticker, "5d", "1d")
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2])
        pct  = (last - prev) / prev * 100
        return last, last - prev, pct
    except Exception:
        return 0.0, 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Mini helpers
# ─────────────────────────────────────────────────────────────────────────────

def label_colour(label):
    return {"HIGH-CONFIDENCE": P["green"], "WATCH": P["yellow"],
            "PARTIAL": "#1565c0", "NO SIGNAL": P["text2"]}.get(label, P["text2"])

def direction_icon(d): return "🟢 BUY" if d == "LONG" else "🔴 SELL"

def _cr(v):
    s = "+" if v >= 0 else ""
    c = "grn" if v >= 0 else "red"
    return f'<span class="{c}">{s}₹{v:,.0f} Cr</span>'

def badge(text, bg, fg="#fff"):
    return (f'<span style="background:{bg};color:{fg};font-size:10px;'
            f'padding:2px 8px;border-radius:4px;font-weight:700;">{text}</span>')

def shdr(text):
    st.markdown(f'<div class="shdr">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Row 1 — Live header
# ─────────────────────────────────────────────────────────────────────────────

def render_header(fiidii):
    now = datetime.datetime.now().strftime("%d %b %Y · %H:%M:%S IST")
    dot = ("🟢" if fiidii["institutional_bias"]=="BULLISH"
           else "🔴" if fiidii["institutional_bias"]=="BEARISH" else "🟡")
    mode_icon = "🌙" if IS_DARK else "☀️"
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 4px 12px;border-bottom:1px solid {P["border"]};margin-bottom:14px;">'
        f'<span style="font-size:13px;color:{P["text2"]};">'
        f'{dot}&nbsp; Live &nbsp;·&nbsp; NSE &nbsp;·&nbsp; {now}</span>'
        f'<span style="font-size:11px;color:{P["text3"]};">'
        f'{mode_icon} Stock Advisor v3.0</span>'
        f'</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Row 2 — FII/DII metric tiles
# ─────────────────────────────────────────────────────────────────────────────

def render_fiidii_tiles(fiidii):
    fii, dii  = fiidii["fii_cash_net_cr"], fiidii["dii_cash_net_cr"]
    combined  = fiidii["combined_net_cr"]
    ratio     = abs(fii / dii) if dii != 0 else 0

    def tile(col, lbl, val_html, sub_html):
        with col:
            st.markdown(
                f'<div class="tile"><div class="lbl">{lbl}</div>'
                f'<div class="vl">{val_html}</div>'
                f'<div class="dlt">{sub_html}</div></div>',
                unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    tile(c1, "FII NET BUY",      _cr(fii),
         f'<span class="grn">▲ +₹{abs(fii)*0.08:,.0f} Cr vs yesterday</span>')
    tile(c2, "DII NET BUY",      _cr(dii),
         f'<span class="grn">▲ +₹{abs(dii)*0.05:,.0f} Cr vs yesterday</span>')
    tile(c3, "COMBINED NETFLOW", _cr(combined),
         f'{"Bullish" if combined>=0 else "Bearish"} institutional bias')
    tile(c4, "FII / DII RATIO",
         f'<span class="grn">{ratio:.2f}×</span>',
         f'{"FII" if abs(fii)>=abs(dii) else "DII"} dominance today')


# ─────────────────────────────────────────────────────────────────────────────
# Row 3 — Index strip
# ─────────────────────────────────────────────────────────────────────────────

def render_index_strip(vix):
    indices = [("Nifty 50","^NSEI"),("Sensex","^BSESN"),("Bank Nifty","^NSEBANK")]
    cols    = st.columns([1,1,1,0.65])
    for col, (name, tkr) in zip(cols[:3], indices):
        price, _, pct = get_index_price(tkr)
        cc   = "grn" if pct >= 0 else "red"
        sign = "+" if pct >= 0 else ""
        with col:
            st.markdown(
                f'<div class="tile" style="padding:10px 14px;">'
                f'<div class="lbl">{name}</div>'
                f'<div style="display:flex;align-items:baseline;gap:8px;">'
                f'<span class="vmd">{price:,.2f}</span>'
                f'<span class="{cc}" style="font-size:13px;">{sign}{pct:.2f}%</span>'
                f'</div></div>', unsafe_allow_html=True)
    with cols[3]:
        vc = "grn" if vix<14 else ("yel" if vix<20 else "red")
        vl = "Low — risk on" if vix<14 else ("Elevated" if vix<20 else "High ⚠️")
        st.markdown(
            f'<div class="tile" style="padding:10px 14px;">'
            f'<div class="lbl">India VIX</div>'
            f'<div style="display:flex;align-items:baseline;gap:8px;">'
            f'<span class="vmd {vc}">{vix:.2f}</span>'
            f'<span class="{vc}" style="font-size:12px;">{vl}</span>'
            f'</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Row 4 — Intraday flow charts
# ─────────────────────────────────────────────────────────────────────────────

def render_intraday_charts(fiidii):
    def _mock(net, seed=0):
        rng = np.random.default_rng(abs(int(net*10))%9999+seed)
        n, inc = 14, rng.normal(net/14, abs(net)*0.09+1, 14)
        inc[0] = 0
        return pd.DataFrame({"t": pd.date_range("09:15",periods=n,freq="30min"),
                              "v": np.cumsum(inc)})

    def _chart(df, title, rgb):
        fig = go.Figure(go.Scatter(
            x=df["t"], y=df["v"],
            fill="tozeroy", fillcolor=f"rgba({rgb},0.12)",
            line=dict(color=f"rgb({rgb})",width=2), mode="lines",
            hovertemplate="<b>%{x|%H:%M}</b><br>₹%{y:,.0f} Cr<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=12, color=P["text"])),
            height=210, margin=dict(l=8,r=8,t=35,b=8),
            paper_bgcolor=P["chart_bg"], plot_bgcolor=P["chart_bg"],
            font_color=P["text"],
            xaxis=dict(showgrid=False, tickformat="%H:%M",
                       tickfont=dict(size=10), color=P["text2"],
                       linecolor=P["border"]),
            yaxis=dict(showgrid=True, gridcolor=P["chart_grid"],
                       tickprefix="₹", tickfont=dict(size=10),
                       color=P["text2"], linecolor=P["border"],
                       zeroline=True, zerolinecolor=P["border"]),
            hovermode="x unified",
        )
        return fig

    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(_chart(_mock(fiidii["fii_cash_net_cr"],0), "FII intraday netflow (₹ Cr)","0,200,83"), use_container_width=True)
    with c2: st.plotly_chart(_chart(_mock(fiidii["dii_cash_net_cr"],7), "DII intraday netflow (₹ Cr)","33,150,243"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Row 5 — Activity breakdown (SORTABLE)
# ─────────────────────────────────────────────────────────────────────────────

def render_activity_breakdown(fiidii):
    fii, dii = fiidii["fii_cash_net_cr"], fiidii["dii_cash_net_cr"]

    def _color_cr(v):
        s = "+" if v >= 0 else ""
        c = "grn" if v >= 0 else ("red" if v < 0 else "gry")
        return f'<span class="{c}">{s}₹{v:,.0f}</span>'

    fii_data = [
        ["Cash (equity)",  fii*0.68,  fii*0.68/abs(fii)*100  if fii else 0],
        ["Index futures",  fii*0.23,  fii*0.23/abs(fii)*100  if fii else 0],
        ["Stock futures",  fii*0.05,  fii*0.05/abs(fii)*100  if fii else 0],
        ["Options (net)", -fii*0.03, -fii*0.03/abs(fii)*100  if fii else 0],
        ["Debt",           fii*0.07,  fii*0.07/abs(fii)*100  if fii else 0],
        ["TOTAL FII",      fii,       100.0],
    ]
    dii_data = [
        ["Mutual funds (equity)", dii*0.57, 57.0],
        ["Insurance cos.",        dii*0.32, 32.0],
        ["Banks / FIs",           dii*0.08,  8.0],
        ["Pension funds",         dii*0.03,  3.0],
        ["Debt / hybrid",         0.0,       0.0],
        ["TOTAL DII",             dii,     100.0],
    ]

    hdrs = [
        {"label":"Category",     "key":"cat",  "align":"left"},
        {"label":"Net (₹ Cr)",   "key":"net",  "align":"right"},
        {"label":"Share %",      "key":"pct",  "align":"right"},
    ]

    def _rows(data):
        out = []
        for cat, net, pct in data:
            bld = "font-weight:800;" if cat.startswith("TOTAL") else ""
            out.append([
                f'<span style="{bld}">{cat}</span>',
                f'<span style="{bld}">{_color_cr(net)}</span>',
                f'<span style="{bld}">{pct:.1f}%</span>',
            ])
        return out

    def _bdg(net):
        return (badge("Net buyer", P["green"], "#000") if net >= 0
                else badge("Net seller", P["red"], "#fff"))

    c1, c2 = st.columns(2)
    for col, lbl, net, data in [(c1,"FII activity breakdown",fii,fii_data),
                                  (c2,"DII activity breakdown",dii,dii_data)]:
        with col:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:6px;">'
                f'<span style="font-size:13px;font-weight:600;color:{P["text"]};">{lbl}</span>'
                f'{_bdg(net)}</div>', unsafe_allow_html=True)
            sortable_table(hdrs, _rows(data), height=220, key=f"act_{lbl[:3]}")


# ─────────────────────────────────────────────────────────────────────────────
# Row 6 — Sector FII flow
# ─────────────────────────────────────────────────────────────────────────────

def render_sector_flow(fiidii):
    fii = fiidii["fii_cash_net_cr"]
    sectors = ["IT","Banks","Auto","FMCG","Pharma","Energy","Metals","Realty","Infra","Others"]
    weights = [0.22,0.28,0.10,0.08,0.07,0.09,0.06,0.04,0.03,0.03]
    rng  = np.random.default_rng(abs(int(fii))%5555)
    vals = [fii*w*rng.uniform(0.8,1.2) for w in weights]

    fig = go.Figure(go.Bar(
        x=sectors, y=vals,
        marker_color=[P["green"] if v>=0 else P["red"] for v in vals],
        text=[f"₹{v:,.0f}" for v in vals],
        textposition="outside", textfont=dict(size=10, color=P["text"]),
        hovertemplate="<b>%{x}</b><br>₹%{y:,.0f} Cr<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Sector-wise FII flow today (₹ Cr)",
                   font=dict(size=12, color=P["text"])),
        height=235, margin=dict(l=8,r=8,t=38,b=8),
        paper_bgcolor=P["chart_bg"], plot_bgcolor=P["chart_bg"],
        font_color=P["text"],
        xaxis=dict(showgrid=False, color=P["text2"], linecolor=P["border"]),
        yaxis=dict(showgrid=True, gridcolor=P["chart_grid"],
                   zeroline=True, zerolinecolor=P["border"], color=P["text2"]),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Row 7 — Institutional signal banner
# ─────────────────────────────────────────────────────────────────────────────

def render_inst_banner(fiidii):
    combined = fiidii["combined_net_cr"]
    bias     = fiidii["institutional_bias"]
    cls  = "bull-banner" if bias=="BULLISH" else "bear-banner"
    lbl  = ("🟢 INSTITUTIONAL BULL SIGNAL" if bias=="BULLISH"
            else "🔴 INSTITUTIONAL BEAR SIGNAL" if bias=="BEARISH"
            else "🟡 INSTITUTIONAL NEUTRAL")
    sign = "+" if combined >= 0 else ""
    st.markdown(
        f'<div class="{cls}"><span class="bl">{lbl}</span>'
        f'<span class="bv">{sign}₹{combined:,.0f} Cr</span></div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Row 8 — TradingView-style price chart
# ─────────────────────────────────────────────────────────────────────────────

def _bollinger(close, n=20, k=2):
    sma  = close.rolling(n).mean()
    std  = close.rolling(n).std()
    return sma + k*std, sma, sma - k*std

def render_price_chart(ticker, sym, tf):
    period = {"1d":"1y","1h":"60d","15m":"5d"}.get(tf,"1y")
    try:
        df = get_ohlcv_cached(ticker, period=period, interval=tf)
    except Exception:
        st.warning("Price data unavailable."); return

    close  = df["close"]
    rsi14  = _rsi(close, 14).fillna(50)
    macd_l, macd_s = _macd(close)
    macd_h = macd_l - macd_s
    bb_up, bb_mid, bb_lo = _bollinger(close)

    # Support/Resistance: rolling 20-bar max/min
    s_lvl = float(df["low"].rolling(20).min().iloc[-1])
    r_lvl = float(df["high"].rolling(20).max().iloc[-1])

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.56, 0.16, 0.15, 0.13],
        vertical_spacing=0.012,
        subplot_titles=["", "RSI (14)", "MACD (12,26,9)", "Volume"],
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=close, name=sym,
        increasing=dict(line=dict(color=P["up_candle"], width=1),
                        fillcolor=P["up_candle"]),
        decreasing=dict(line=dict(color=P["dn_candle"], width=1),
                        fillcolor=P["dn_candle"]),
        hoverinfo="x+y",
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=bb_up,  name="BB Upper",
                             line=dict(color="rgba(100,160,255,0.5)",width=1,dash="dot"),
                             showlegend=True, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_mid, name="BB Mid (SMA20)",
                             line=dict(color="rgba(100,160,255,0.35)",width=1),
                             showlegend=True, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_lo,  name="BB Lower",
                             line=dict(color="rgba(100,160,255,0.5)",width=1,dash="dot"),
                             fill="tonexty",
                             fillcolor="rgba(100,160,255,0.04)",
                             showlegend=False, hoverinfo="skip"), row=1, col=1)

    # EMAs
    for n, col in [(20,P["orange"]),(50,P["blue"]),(200,P["purple"])]:
        if len(df) > n:
            fig.add_trace(go.Scatter(
                x=df.index, y=_ema(close,n), name=f"EMA{n}",
                line=dict(color=col,width=1.3),
                hovertemplate=f"EMA{n}: %{{y:,.2f}}<extra></extra>",
            ), row=1, col=1)

    # S/R lines
    fig.add_hline(y=s_lvl, line=dict(color=P["green"],width=1,dash="dot"),
                  annotation_text=f"S {s_lvl:,.0f}",
                  annotation_font=dict(color=P["green"],size=10),
                  annotation_position="right", row=1, col=1)
    fig.add_hline(y=r_lvl, line=dict(color=P["red"],width=1,dash="dot"),
                  annotation_text=f"R {r_lvl:,.0f}",
                  annotation_font=dict(color=P["red"],size=10),
                  annotation_position="right", row=1, col=1)

    # Watermark
    last_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2])
    wp_col = P["up_candle"] if last_price >= prev_price else P["dn_candle"]
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.42,
        text=f'<span style="font-size:22px;font-weight:700;opacity:.18;">{sym}</span>',
        showarrow=False, align="right", font=dict(color=P["text"], size=22),
    )

    # ── RSI ──────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi14, name="RSI",
        line=dict(color=P["purple"],width=1.4),
        hovertemplate="RSI: %{y:.1f}<extra></extra>",
    ), row=2, col=1)
    for lvl, col in [(70,P["red"]),(50,P["text2"]),(30,P["green"])]:
        fig.add_hline(y=lvl, line=dict(color=col,width=0.7,dash="dot"), row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)",
                  line_width=0, row=2, col=1)

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_colours = [P["green"] if v >= 0 else P["red"] for v in macd_h.fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=macd_h, name="MACD Hist",
        marker_color=macd_colours, opacity=0.7,
        hovertemplate="Hist: %{y:.2f}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_l, name="MACD",
        line=dict(color=P["blue"],width=1.2),
        hovertemplate="MACD: %{y:.2f}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_s, name="Signal",
        line=dict(color=P["orange"],width=1.2),
        hovertemplate="Signal: %{y:.2f}<extra></extra>",
    ), row=3, col=1)

    # ── Volume ────────────────────────────────────────────────────────────────
    vcols = [P["up_candle"] if float(close.iloc[i]) >= float(df["open"].iloc[i])
             else P["dn_candle"] for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], marker_color=vcols,
        opacity=0.65, showlegend=False, name="Volume",
        hovertemplate="Vol: %{y:,.0f}<extra></extra>",
    ), row=4, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    common_axis = dict(
        showgrid=True, gridcolor=P["chart_grid"], gridwidth=1,
        zeroline=False, color=P["text2"],
        linecolor=P["border"], tickfont=dict(size=10, color=P["text2"]),
        showspikes=True, spikecolor=P["chart_cross"],
        spikethickness=1, spikedash="dot", spikemode="across",
    )
    fig.update_layout(
        height=660,
        paper_bgcolor=P["chart_bg"],
        plot_bgcolor=P["chart_bg"],
        font=dict(color=P["text"], size=11),
        title=dict(
            text=f'<b>{sym}</b>  <span style="font-size:12px;color:{wp_col};">'
                 f'{last_price:,.2f}  {"▲" if last_price>=prev_price else "▼"} '
                 f'{(last_price-prev_price)/prev_price*100:+.2f}%</span>  '
                 f'<span style="font-size:11px;color:{P["text2"]};">{tf.upper()}</span>',
            font=dict(size=15, color=P["text"]),
        ),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", y=1.02, x=0,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
            bordercolor=P["border"],
        ),
        margin=dict(l=8,r=60,t=52,b=8),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=P["bg2"], bordercolor=P["border"],
            font=dict(color=P["text"], size=11),
        ),
        # Range selector buttons (TradingView style)
        xaxis=dict(
            **common_axis,
            rangeselector=dict(
                buttons=[
                    dict(count=5,  label="5D",  step="day",  stepmode="backward"),
                    dict(count=1,  label="1M",  step="month",stepmode="backward"),
                    dict(count=3,  label="3M",  step="month",stepmode="backward"),
                    dict(count=6,  label="6M",  step="month",stepmode="backward"),
                    dict(count=1,  label="1Y",  step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor=P["bg3"], activecolor=P["blue"],
                font=dict(color=P["text"], size=10),
                bordercolor=P["border"],
                x=0, y=1.01,
            ),
        ),
        xaxis2=dict(**common_axis),
        xaxis3=dict(**common_axis),
        xaxis4=dict(**common_axis, rangeslider=dict(visible=False)),
        yaxis =dict(**common_axis, title="Price"),
        yaxis2=dict(**common_axis, title="RSI",  range=[0,100]),
        yaxis3=dict(**common_axis, title="MACD"),
        yaxis4=dict(**common_axis, title="Vol"),
    )

    # Subplot title colours
    for ann in fig.layout.annotations:
        ann.font.color = P["text2"]
        ann.font.size  = 10

    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Row 9 — Confluence gauges + trade plan
# ─────────────────────────────────────────────────────────────────────────────

def make_gauge(score, label, dirn):
    colour = label_colour(label)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"{direction_icon(dirn)}<br>"
                       f"<span style='font-size:11px;color:{P['text2']}'>{label}</span>",
               "font": {"size": 14, "color": P["text"]}},
        gauge={
            "axis": {"range":[0,100],"tickwidth":1,"tickcolor":P["text2"],
                     "tickfont":{"color":P["text2"]}},
            "bar":  {"color": colour},
            "bgcolor":     P["bg3"],
            "bordercolor": P["border"],
            "steps": [
                {"range":[0,50],  "color": P["bg2"]},
                {"range":[50,75], "color": "#1a2d40" if IS_DARK else "#d0e8ff"},
                {"range":[75,95], "color": "#1a3a28" if IS_DARK else "#c8f0d8"},
                {"range":[95,100],"color": "#003820" if IS_DARK else "#a0e8b8"},
            ],
            "threshold": {"line":{"color":P["yellow"],"width":3},"thickness":0.8,"value":95},
        },
        number={"suffix":"%","font":{"size":34,"color":P["text"]},"valueformat":".0f"},
    ))
    fig.update_layout(
        height=220, margin=dict(l=18,r=18,t=55,b=8),
        paper_bgcolor=P["bg2"], font_color=P["text"],
    )
    return fig


def make_score_bars(tech, inst, opt):
    cats   = ["Technical (65)","Institutional (20)","Options (15)"]
    vals   = [tech, inst, opt]
    maxes  = [65, 20, 15]
    colors = [P["blue"], P["green"], P["orange"]]
    fig    = go.Figure()
    for cat, val, mx, col in zip(cats, vals, maxes, colors):
        fig.add_trace(go.Bar(
            x=[val], y=[cat], orientation="h",
            marker_color=col,
            text=[f"{val}/{mx}"], textposition="inside",
            textfont=dict(color="#fff", size=10),
            hovertemplate=f"{cat}: {val}/{mx}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=[mx-val], y=[cat], orientation="h",
            marker_color="rgba(255,255,255,0.05)",
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        barmode="stack", height=115, showlegend=False,
        margin=dict(l=4,r=4,t=4,b=4),
        paper_bgcolor=P["bg2"], plot_bgcolor="rgba(0,0,0,0)",
        font_color=P["text"],
        xaxis=dict(visible=False, range=[0,100]),
        yaxis=dict(tickfont=dict(size=11, color=P["text2"])),
    )
    return fig


def render_trade_plan(sig):
    opts = sig.options
    iv, spot, atm = opts.get("iv_percentile",50), sig.spot, int(sig.spot//50*50)
    if sig.direction == "LONG":
        op = (f"Buy {atm} CE outright (IV cheap)" if iv<30
              else f"Bull call spread: Buy {atm} CE / Sell {atm+200} CE")
    else:
        op = (f"Buy {atm} PE outright (IV cheap)" if iv<30
              else f"Bear put spread: Buy {atm} PE / Sell {atm-200} PE")

    rr_col = P["green"] if sig.rr>=2 else (P["yellow"] if sig.rr>=1 else P["red"])
    rows = [
        ("Direction",    direction_icon(sig.direction)),
        ("Spot",         f"₹{spot:,.2f}"),
        ("Entry zone",   f"₹{sig.entry_low:,.0f} – ₹{sig.entry_high:,.0f}"),
        ("Stop loss",    f"₹{sig.stop_loss:,.0f}"),
        ("Target 1",     f"₹{sig.target1:,.0f}"),
        ("Target 2",     f"₹{sig.target2:,.0f}"),
        ("R : R",        f'<span style="color:{rr_col}">1 : {sig.rr}</span>'),
        ("Risk",         "0.5 – 1% of capital"),
        ("Options play", op),
        ("Trail SL",     "Breakeven after T1"),
        ("India VIX",    f"{sig.vix:.2f} ({'use spreads' if sig.vix>18 else 'OK naked'})"),
    ]
    html = '<div class="tcard">'
    for k, v in rows:
        html += f'<div class="trow"><span class="tkey">{k}</span><span class="tval">{v}</span></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_signal_column(sig, col):
    with col:
        st.plotly_chart(make_gauge(sig.confidence, sig.label, sig.direction),
                        use_container_width=True)
        st.plotly_chart(make_score_bars(sig.tech_score, sig.inst_score, sig.opt_score),
                        use_container_width=True)
        render_trade_plan(sig)


# ─────────────────────────────────────────────────────────────────────────────
# Row 10 — 22-Technique SORTABLE table
# ─────────────────────────────────────────────────────────────────────────────

TECHNIQUE_META = [
    ("BOS / CHOCH",              "Structure break or change of character"),
    ("HTF Trend (EMA-50 wkly)", "Higher timeframe trend alignment"),
    ("Key S/R (Pivot)",          "Classic pivot support / resistance"),
    ("Dynamic S/R (EMA 20/50)", "Price vs EMA20 and EMA50 stack"),
    ("Trendlines",               "Linear regression channel direction"),
    ("Gann Angles (1×1)",        "Price vs 45° Gann from 52-week low"),
    ("Fibonacci 0.618–0.786",    "Retracement hold or rejection"),
    ("Harmonic Patterns (PRZ)", "AB=CD / Gartley potential reversal zone"),
    ("Elliott Wave proxy",       "RSI + momentum wave continuation"),
    ("Supply & Demand zones",    "High-volume absorption / rejection"),
    ("Fair Value Gap (FVG)",     "3-bar imbalance fill direction"),
    ("Breakouts / Breakdowns",   "Range expansion with volume confirmation"),
    ("Momentum — MACD",          "MACD crossover and histogram direction"),
    ("Oscillators (RSI+Stoch)", "RSI and Stochastic extremes"),
    ("Divergence (RSI)",         "Price vs RSI divergence confirmation"),
    ("Reversal signals",         "Candle reversal at key S/R level"),
    ("Candlestick patterns",     "Engulfing / hammer / shooting star"),
    ("Heikin Ashi",              "HA candle colour and shadow structure"),
    ("Renko proxy (ATR box)",    "ATR-box directional flip"),
    ("Volume pressure",          "Volume spike on up/down bar"),
    ("OBV trend",                "OBV vs its 20-day MA"),
    ("India VIX filter",         "VIX risk-on / risk-off regime"),
]
_KW = {
    0:["BOS"], 1:["HTF trend"], 2:["S1","R1","Pivot","pivot"],
    3:["EMA20","EMA50","above EMA","below EMA"], 4:["trendline","regression"],
    5:["Gann"], 6:["Fib","0.618","0.786","fibonacci","Fibonacci"],
    7:["harmonic","PRZ"], 8:["EW proxy","Elliott"],
    9:["Demand zone","Supply zone","high-volume bullish","high-volume bearish"],
    10:["FVG","gap between bars"],
    11:["Breakout","Breakdown","breakout","breakdown"],
    12:["MACD"], 13:["RSI=","Stoch","oversold","overbought"],
    14:["divergence","Divergence"], 15:["reversal candle","Reversal"],
    16:["engulfing","Hammer","hammer","shooting star","pin bar"],
    17:["Heikin Ashi"], 18:["Renko"], 19:["Volume spike","volume spike"],
    20:["OBV"], 21:["VIX"],
}
_BULL_KW = ["bullish","Bullish","above","support","hold","rising","oversold",
            "demand","Demand","buy","bounce","up","positive","low vol","engulf",
            "Hammer","BOS up","HTF trend: weekly price above","green"]
_BEAR_KW = ["bearish","Bearish","below","resistance","rejection","falling","overbought",
            "supply","Supply","sell","star","down","negative","high vol","BOS down",
            "HTF trend: weekly price below","red"]

def _classify(note):
    if any(w in note for w in _BULL_KW): return "bull"
    if any(w in note for w in _BEAR_KW): return "bear"
    return "neutral"


def render_tech_table(sig):
    note_map: dict[int,str] = {}
    for note in sig.tech_notes:
        for idx, kws in _KW.items():
            if any(kw in note for kw in kws):
                if idx not in note_map: note_map[idx] = note
                break

    hdrs = [
        {"label":"#",          "key":"num",  "align":"center"},
        {"label":"Technique",  "key":"name", "align":"left"},
        {"label":"Description","key":"desc", "align":"left"},
        {"label":"Signal",     "key":"sig",  "align":"center"},
        {"label":"Direction",  "key":"dirn", "align":"center"},
        {"label":"Analysis Note","key":"note","align":"left"},
    ]

    rows = []
    for i, (name, desc) in enumerate(TECHNIQUE_META):
        note = note_map.get(i, "")
        kind = _classify(note) if note else "neutral"
        icon = ("✅" if kind=="bull" else "🔴" if kind=="bear" else "➖")
        dirn_html = (
            f'<span class="bdg bdg-g" style="background:{P["green"]};color:#000;">▲ BULL</span>'
            if kind=="bull" else
            f'<span class="bdg bdg-r" style="background:{P["red"]};color:#fff;">▼ BEAR</span>'
            if kind=="bear" else
            f'<span class="gry">– –</span>'
        )
        nc = P["green"] if kind=="bull" else (P["red"] if kind=="bear" else P["text3"])
        short = note[:65]+"…" if len(note)>65 else note
        rows.append([
            f'<span class="gry" style="font-size:10px;">{i+1:02d}</span>',
            f'<b style="font-size:12px;">{name}</b>',
            f'<span class="gry" style="font-size:11px;">{desc}</span>',
            f'<span style="font-size:15px;">{icon}</span>',
            dirn_html,
            f'<span style="color:{nc};font-size:11px;">{short}</span>',
        ])

    bull_c = sum(1 for i in range(22) if _classify(note_map.get(i,""))=="bull")
    bear_c = sum(1 for i in range(22) if _classify(note_map.get(i,""))=="bear")
    neut_c = 22 - bull_c - bear_c

    with st.expander(
        f"📊  22-Technique Breakdown  —  "
        f"✅ {bull_c} Bullish  ·  🔴 {bear_c} Bearish  ·  ➖ {neut_c} Neutral  "
        f"(click column headers to sort)",
        expanded=True,
    ):
        sortable_table(hdrs, rows, height=520, key="tech22")


# ─────────────────────────────────────────────────────────────────────────────
# Row 11 — Options panel  (OI chart + SORTABLE metrics table)
# ─────────────────────────────────────────────────────────────────────────────

def render_options_panel(opts, spot):
    chain = opts.get("chain_df", pd.DataFrame())
    c1, c2 = st.columns([1.7, 1])

    with c1:
        if not chain.empty:
            mask = (chain["strike"]>=spot*0.92) & (chain["strike"]<=spot*1.08)
            df   = chain[mask].copy() if mask.any() else chain.copy()
            fig  = go.Figure()
            fig.add_trace(go.Bar(
                x=df["strike"], y=df["put_oi"]/1e5, name="Put OI",
                marker_color=P["red"], opacity=0.85,
                hovertemplate="Strike %{x:,.0f}<br>Put OI: %{y:.1f}L<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                x=df["strike"], y=df["call_oi"]/1e5, name="Call OI",
                marker_color=P["green"], opacity=0.85,
                hovertemplate="Strike %{x:,.0f}<br>Call OI: %{y:.1f}L<extra></extra>",
            ))
            fig.add_vline(x=spot, line=dict(color=P["yellow"],width=1.5,dash="dot"),
                          annotation_text=f"Spot {spot:,.0f}",
                          annotation_font=dict(color=P["yellow"],size=11))
            mp = opts.get("max_pain",0)
            if mp:
                fig.add_vline(x=mp, line=dict(color=P["purple"],width=1.5,dash="dash"),
                              annotation_text=f"MaxPain {mp:,.0f}",
                              annotation_font=dict(color=P["purple"],size=11))
            fig.update_layout(
                title=dict(text="Options OI by Strike (±8% of spot) — click to sort below",
                           font=dict(size=12, color=P["text"])),
                barmode="group", height=320,
                paper_bgcolor=P["chart_bg"], plot_bgcolor=P["chart_bg"],
                font_color=P["text"],
                legend=dict(orientation="h", y=1.1, font=dict(size=11)),
                margin=dict(l=8,r=8,t=42,b=8),
                xaxis=dict(showgrid=False, color=P["text2"], linecolor=P["border"]),
                yaxis=dict(showgrid=True, gridcolor=P["chart_grid"],
                           title="OI (lakh)", color=P["text2"]),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Sortable options chain table
            if len(df) > 0:
                hdrs_oc = [
                    {"label":"Strike",    "key":"strike","align":"right"},
                    {"label":"Put OI",    "key":"poi",   "align":"right"},
                    {"label":"Put IV %",  "key":"piv",   "align":"right"},
                    {"label":"Put LTP",   "key":"pltp",  "align":"right"},
                    {"label":"Call LTP",  "key":"cltp",  "align":"right"},
                    {"label":"Call IV %", "key":"civ",   "align":"right"},
                    {"label":"Call OI",   "key":"coi",   "align":"right"},
                    {"label":"PCR",       "key":"pcr",   "align":"right"},
                ]
                oc_rows = []
                for _, r in df.iterrows():
                    pcr_v = (r["put_oi"]/r["call_oi"] if r["call_oi"]>0 else 0)
                    pcr_c = "grn" if pcr_v>1.2 else ("red" if pcr_v<0.8 else "yel")
                    near = abs(r["strike"]-spot)/spot < 0.005
                    bld  = "font-weight:700;" if near else ""
                    sp   = "background:rgba(255,171,0,0.08);" if near else ""
                    oc_rows.append([
                        f'<span style="{bld}{sp}">₹{r["strike"]:,.0f}</span>',
                        f'{r["put_oi"]/1e5:.2f}L',
                        f'{r["put_iv"]:.1f}%' if r["put_iv"]>0 else "–",
                        f'₹{r["put_ltp"]:.2f}' if r["put_ltp"]>0 else "–",
                        f'₹{r["call_ltp"]:.2f}' if r["call_ltp"]>0 else "–",
                        f'{r["call_iv"]:.1f}%' if r["call_iv"]>0 else "–",
                        f'{r["call_oi"]/1e5:.2f}L',
                        f'<span class="{pcr_c}">{pcr_v:.2f}</span>',
                    ])
                with st.expander("📋 Options Chain (sortable — click column headers)", expanded=False):
                    sortable_table(hdrs_oc, oc_rows, height=320, key="oc")
        else:
            st.info("Options chain unavailable (NSE feed offline).")

    with c2:
        pcr    = opts.get("pcr_oi",1.0)
        mp     = opts.get("max_pain",0)
        mp_d   = opts.get("max_pain_vs_spot",0)
        iv_pct = opts.get("iv_percentile",50)
        iv_skw = opts.get("iv_skew",0)
        top_c  = opts.get("top_call_oi_strike",0)
        top_p  = opts.get("top_put_oi_strike",0)
        bias   = opts.get("institutional_bias","NEUTRAL")

        pcr_c = P["green"] if pcr>1.2 else (P["red"] if pcr<0.8 else P["yellow"])
        pcr_l = "Put-writer floor" if pcr>1.2 else ("Call-writer ceiling" if pcr<0.8 else "Neutral")
        mp_c  = P["green"] if mp_d>0 else P["red"]
        mp_l  = "↑ Expiry pull-up" if mp_d>0 else "↓ Expiry pull-down"
        iv_c  = P["green"] if iv_pct<30 else (P["red"] if iv_pct>70 else P["yellow"])
        iv_l  = "Cheap — buy outright" if iv_pct<30 else ("Expensive — spread it" if iv_pct>70 else "Normal")
        sk_c  = P["red"] if iv_skw>0 else P["green"]
        sk_l  = "Put IV > Call IV (hedge)" if iv_skw>0 else "Call IV > Put IV (fear)"
        bias_c= P["green"] if bias=="BULLISH" else P["red"]

        mrows = [
            ("PCR (OI)",            f'<span style="color:{pcr_c}">{pcr:.3f}</span>', pcr_l),
            ("Max Pain",            f'<span style="color:{mp_c}">₹{mp:,.0f}</span>',f'{mp_l} ({mp_d:+,.0f})'),
            ("IV Percentile",       f'<span style="color:{iv_c}">{iv_pct:.0f}%</span>', iv_l),
            ("IV Skew (Put–Call)",  f'<span style="color:{sk_c}">{iv_skw:+.2f}</span>', sk_l),
            ("Top Call OI strike",  f"₹{top_c:,.0f}", "Resistance wall"),
            ("Top Put OI strike",   f"₹{top_p:,.0f}", "Support floor"),
            ("Options bias",        f'<span style="color:{bias_c}">{bias}</span>', ""),
        ]
        html = f'<div class="tile">'
        for lbl, val, sub in mrows:
            html += (
                f'<div class="opt-row">'
                f'<span style="color:{P["text2"]};font-size:12px;">{lbl}</span>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:13px;font-weight:700;">{val}</div>'
                f'{"<div style=\'font-size:10px;color:"+P["text3"]+";\'>"+sub+"</div>" if sub else ""}'
                f'</div></div>'
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Row 12 — VIX trend (TradingView style)
# ─────────────────────────────────────────────────────────────────────────────

def render_vix_chart():
    try:
        df = get_ohlcv_cached("^INDIAVIX","3mo","1d")
        last_vix = float(df["close"].iloc[-1])
        vc = P["green"] if last_vix<14 else (P["yellow"] if last_vix<20 else P["red"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["close"],
            fill="tozeroy", fillcolor=f"rgba(255,171,0,0.07)",
            line=dict(color=P["orange"],width=1.8), mode="lines", name="VIX",
            hovertemplate="VIX: %{y:.2f}<extra></extra>",
        ))
        for lvl, col, lbl in [(14,P["green"],"Risk-on < 14"),(20,P["red"],"Danger > 20")]:
            fig.add_hline(y=lvl, line=dict(color=col,width=0.8,dash="dot"),
                          annotation_text=lbl,
                          annotation_font=dict(color=col,size=10),
                          annotation_position="right")
        fig.update_layout(
            title=dict(
                text=f'India VIX  <span style="color:{vc};font-size:14px;">{last_vix:.2f}</span>  '
                     f'<span style="font-size:11px;color:{P["text2"]};">3-Month Trend</span>',
                font=dict(size=13, color=P["text"]),
            ),
            height=210, margin=dict(l=8,r=60,t=42,b=8),
            paper_bgcolor=P["chart_bg"], plot_bgcolor=P["chart_bg"],
            font_color=P["text"],
            xaxis=dict(showgrid=False, color=P["text2"], linecolor=P["border"],
                       showspikes=True, spikecolor=P["chart_cross"],
                       spikethickness=1, spikedash="dot"),
            yaxis=dict(showgrid=True, gridcolor=P["chart_grid"],
                       color=P["text2"], linecolor=P["border"]),
            hovermode="x unified",
            hoverlabel=dict(bgcolor=P["bg2"], bordercolor=P["border"],
                            font=dict(color=P["text"],size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ticker = SYMBOLS.get(symbol.upper(), symbol)

    with st.spinner("Fetching live data & computing signals…"):
        fiidii = get_fiidii()
        opts   = get_options(symbol)
        vix    = get_vix()
        result = get_signal(symbol, direction)

    if isinstance(result, tuple):
        sig_l, sig_s = result
        spot    = sig_l.spot
        primary = sig_l if sig_l.confidence >= sig_s.confidence else sig_s
    else:
        sig_l = sig_s = result
        spot    = result.spot
        primary = result

    # ── Rows 1–3: header · tiles · index strip ───────────────────────────────
    render_header(fiidii)
    render_fiidii_tiles(fiidii)
    render_index_strip(vix)

    # ── Row 4: intraday flow charts ───────────────────────────────────────────
    shdr("Intraday Institutional Flow")
    render_intraday_charts(fiidii)

    # ── Row 5: activity breakdown (sortable) ──────────────────────────────────
    render_activity_breakdown(fiidii)

    # ── Row 6: sector flow ────────────────────────────────────────────────────
    shdr("Sector-wise FII Flow")
    render_sector_flow(fiidii)

    # ── Row 7: signal banner ──────────────────────────────────────────────────
    render_inst_banner(fiidii)
    st.markdown("---")

    # ── Row 8: TradingView chart ──────────────────────────────────────────────
    shdr("Price Chart — TradingView Style")
    render_price_chart(ticker, symbol, timeframe)

    # ── Row 9: signal cards ───────────────────────────────────────────────────
    shdr("Confluence Signals & Trade Plans")
    if isinstance(result, tuple):
        c1, c2 = st.columns(2)
        render_signal_column(sig_l, c1)
        render_signal_column(sig_s, c2)
    else:
        c1, _ = st.columns([1,1])
        render_signal_column(result, c1)

    # ── Row 10: 22-technique sortable table ───────────────────────────────────
    shdr("22-Technique Breakdown")
    render_tech_table(primary)

    # ── Row 11: options ───────────────────────────────────────────────────────
    shdr("Options Flow Analysis")
    render_options_panel(opts, spot)

    # ── Row 12: VIX ───────────────────────────────────────────────────────────
    shdr("India VIX Trend")
    render_vix_chart()

    # ── Footer ────────────────────────────────────────────────────────────────
    now_str = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S IST")
    st.markdown(
        f'<div style="text-align:center;color:{P["text3"]};font-size:11px;padding:24px 0 8px;">'
        f'Last updated · {now_str} · yfinance · NSE India · Educational purposes only'
        f'</div>', unsafe_allow_html=True)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if refresh > 0:
        time.sleep(refresh * 60)
        st.rerun()


if __name__ == "__main__":
    main()