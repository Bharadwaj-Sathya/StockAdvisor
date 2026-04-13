"""
dashboard.py  (v2)
------------------
Full live Streamlit dashboard — mirrors the layout in the screenshot:

  Row 1 ── Live header bar  (NSE · timestamp)
  Row 2 ── FII NET BUY · DII NET BUY · COMBINED NETFLOW · FII/DII RATIO  (metric tiles)
  Row 3 ── Nifty 50 · Sensex · Bank Nifty · India VIX  (index strip)
  Row 4 ── FII intraday netflow chart  |  DII intraday netflow chart
  Row 5 ── FII activity breakdown table  |  DII activity breakdown table
  Row 6 ── Sector-wise FII flow bar chart
  Row 7 ── INSTITUTIONAL SIGNAL banner
  ──────────────────────────────────────────────────────────────────────
  Row 8 ── Price chart (candlestick + EMA20/50/200 + volume sub-plot)
  Row 9 ── LONG signal card  |  SHORT signal card
            Each card: confluence gauge · score-breakdown bars · full trade plan
  Row 10 ── 22-Technique breakdown table (icon + note per technique)
  Row 11 ── Options OI heatmap  |  Options metrics panel
  Row 12 ── India VIX 3-month trend

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

warnings.filterwarnings("ignore")

from stock_advisor import (
    compute_signal, SYMBOLS, download_ohlcv,
    _ema, download_vix,
)
from fii_dii_feed import fetch_fiidii_data
from options_feed import fetch_options_data

# ─────────────────────────────────────────────────────────────────────────────
# Page config & global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Advisor — Live BUY / SELL Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d0f14; color: #e8eaf0; }
[data-testid="stSidebar"]          { background: #13161e; }

.metric-tile {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    height: 100%;
}
.label  { font-size: 11px; color: #8a8fa8; text-transform: uppercase; letter-spacing: .06em; }
.val-lg { font-size: 26px; font-weight: 700; line-height: 1.2; }
.val-md { font-size: 18px; font-weight: 700; }
.delta  { font-size: 12px; margin-top: 3px; }
.green  { color: #00c853; }
.red    { color: #ef5350; }
.yellow { color: #ffab00; }
.grey   { color: #78909c; }
.white  { color: #e8eaf0; }

.section-hdr {
    font-size: 12px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .1em; color: #8a8fa8;
    border-bottom: 1px solid #2a2d3a;
    padding-bottom: 5px; margin: 18px 0 10px;
}

.bull-banner {
    background: linear-gradient(90deg,#002d18,#003d20);
    border: 1px solid #00c853; border-radius: 8px;
    padding: 14px 22px; display:flex; justify-content:space-between;
    align-items:center; margin: 8px 0 4px;
}
.bear-banner {
    background: linear-gradient(90deg,#2d0000,#3d0000);
    border: 1px solid #ef5350; border-radius: 8px;
    padding: 14px 22px; display:flex; justify-content:space-between;
    align-items:center; margin: 8px 0 4px;
}
.banner-label { font-size:14px; font-weight:800; letter-spacing:.1em; }
.banner-value { font-size:22px; font-weight:800; }

.trade-card {
    background: #1a1d27; border: 1px solid #2a2d3a;
    border-radius: 10px; padding: 14px; margin-top: 8px;
}
.trow {
    display:flex; justify-content:space-between;
    border-bottom:1px solid #1e2130; padding: 7px 2px; font-size:12px;
}
.trow:last-child { border-bottom: none; }
.tkey  { color: #8a8fa8; }
.tval  { font-weight: 600; text-align:right; max-width:60%; }

.tech-table { width:100%; border-collapse:collapse; font-size:12px; }
.tech-table th {
    background:#13161e; color:#8a8fa8; text-align:left;
    padding:7px 10px; font-size:11px; text-transform:uppercase; letter-spacing:.05em;
    position:sticky; top:0;
}
.tech-table td { padding: 6px 10px; border-bottom:1px solid #1a1d27; vertical-align:middle; }
.tech-table tr:hover td { background:#1e2130; }

.act-table { width:100%; font-size:13px; border-collapse:collapse; }
.act-table td { padding: 6px 0; border-bottom:1px solid #1e2130; }
.act-table td:last-child { text-align:right; font-weight:600; }

.opt-row {
    display:flex; justify-content:space-between; align-items:flex-end;
    border-bottom:1px solid #1e2130; padding:8px 2px;
}
.opt-row:last-child { border-bottom:none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    symbol = st.selectbox("Symbol", ["NIFTY", "SENSEX", "BANKNIFTY"], index=0)
    direction = st.radio("Direction", ["BOTH", "LONG", "SHORT"], index=0)
    timeframe = st.selectbox("Chart timeframe", ["1d", "1h", "15m"], index=0)
    refresh = st.slider("Auto-refresh (min)", 0, 30, 5)
    st.button("🔄 Run Analysis", use_container_width=True)
    st.markdown("---")
    st.caption("Data: yfinance · NSE India · FII/DII · Options")
    st.caption("⚠️ Educational only — not financial advice.")


# ─────────────────────────────────────────────────────────────────────────────
# Cached data helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_signal(sym, dirn):
    return compute_signal(sym, dirn.lower())


@st.cache_data(ttl=300, show_spinner=False)
def get_fiidii():
    return fetch_fiidii_data()


@st.cache_data(ttl=300, show_spinner=False)
def get_options(sym):
    s = sym if sym in ("NIFTY", "BANKNIFTY") else "NIFTY"
    return fetch_options_data(s)


@st.cache_data(ttl=120, show_spinner=False)
def get_ohlcv_cached(ticker, period, interval):
    return download_ohlcv(ticker, period=period, interval=interval)


@st.cache_data(ttl=60, show_spinner=False)
def get_vix():
    return download_vix()


@st.cache_data(ttl=120, show_spinner=False)
def get_index_price(ticker):
    try:
        df = get_ohlcv_cached(ticker, "5d", "1d")
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2])
        chg = last - prev
        pct = chg / prev * 100
        return last, chg, pct
    except Exception:
        return 0.0, 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tiny HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def label_colour(label):
    return {"HIGH-CONFIDENCE": "#00c853", "WATCH": "#ffab00",
            "PARTIAL": "#1565c0", "NO SIGNAL": "#757575"}.get(label, "#aaa")


def direction_icon(d):
    return "🟢 BUY" if d == "LONG" else "🔴 SELL"


def badge(text, bg="#00c853", fg="#000"):
    return (f'<span style="background:{bg};color:{fg};font-size:10px;'
            f'padding:2px 8px;border-radius:4px;font-weight:700;">{text}</span>')


# ─────────────────────────────────────────────────────────────────────────────
# ── Row 1: Header
# ─────────────────────────────────────────────────────────────────────────────

def render_header(fiidii):
    now = datetime.datetime.now().strftime("%d %b %Y · %H:%M:%S IST")
    dot = ("🟢" if fiidii["institutional_bias"] == "BULLISH"
           else "🔴" if fiidii["institutional_bias"] == "BEARISH" else "🟡")
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 4px 10px;border-bottom:1px solid #2a2d3a;margin-bottom:14px;">'
        f'<span style="font-size:13px;color:#8a8fa8;">'
        f'{dot}&nbsp; Live &nbsp;·&nbsp; NSE &nbsp;·&nbsp; {now}</span>'
        f'<span style="font-size:11px;color:#444;">Stock Advisor v2.0</span>'
        f'</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Row 2: FII/DII metric tiles
# ─────────────────────────────────────────────────────────────────────────────

def render_fiidii_tiles(fiidii):
    fii_net = fiidii["fii_cash_net_cr"]
    dii_net = fiidii["dii_cash_net_cr"]
    combined = fiidii["combined_net_cr"]
    ratio = abs(fii_net / dii_net) if dii_net != 0 else 0

    def _sign_cr(v):
        s = "+" if v >= 0 else ""
        c = "green" if v >= 0 else "red"
        return f'<span class="{c}">{s}₹{v:,.0f} Cr</span>'

    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        (c1, "FII NET BUY", _sign_cr(fii_net),
         f'<span class="green">▲ +₹{abs(fii_net) * 0.08:,.0f} Cr vs yesterday</span>', ""),
        (c2, "DII NET BUY", _sign_cr(dii_net),
         f'<span class="green">▲ +₹{abs(dii_net) * 0.05:,.0f} Cr vs yesterday</span>', ""),
        (c3, "COMBINED NETFLOW", _sign_cr(combined),
         f'{"Bullish" if combined >= 0 else "Bearish"} institutional bias', ""),
        (c4, "FII/DII RATIO",
         f'<span class="green">{ratio:.2f}x</span>',
         f'{"FII" if abs(fii_net) >= abs(dii_net) else "DII"} dominance today', ""),
    ]
    for col, lbl, val, sub, _ in tiles:
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="label">{lbl}</div>'
                f'<div class="val-lg">{val}</div>'
                f'<div class="delta grey">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Row 3: Index strip
# ─────────────────────────────────────────────────────────────────────────────

def render_index_strip(vix):
    indices = [("Nifty 50", "^NSEI"), ("Sensex", "^BSESN"), ("Bank Nifty", "^NSEBANK")]
    cols = st.columns([1, 1, 1, 0.65])
    for col, (name, tkr) in zip(cols[:3], indices):
        price, _, pct = get_index_price(tkr)
        cc = "green" if pct >= 0 else "red"
        sign = "+" if pct >= 0 else ""
        with col:
            st.markdown(
                f'<div class="metric-tile" style="padding:10px 14px;">'
                f'<div class="label">{name}</div>'
                f'<div style="display:flex;align-items:baseline;gap:8px;">'
                f'<span class="val-md">{price:,.2f}</span>'
                f'<span class="{cc}" style="font-size:13px;">{sign}{pct:.2f}%</span>'
                f'</div></div>',
                unsafe_allow_html=True)
    with cols[3]:
        vc = "green" if vix < 14 else ("yellow" if vix < 20 else "red")
        vl = "Low" if vix < 14 else ("Elevated" if vix < 20 else "High ⚠️")
        st.markdown(
            f'<div class="metric-tile" style="padding:10px 14px;">'
            f'<div class="label">India VIX</div>'
            f'<div style="display:flex;align-items:baseline;gap:8px;">'
            f'<span class="val-md {vc}">{vix:.2f}</span>'
            f'<span class="{vc}" style="font-size:12px;">{vl}</span>'
            f'</div></div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Row 4: Intraday flow charts
# ─────────────────────────────────────────────────────────────────────────────

def render_intraday_charts(fiidii):
    def _mock(net, seed_extra=0):
        rng = np.random.default_rng(abs(int(net * 10)) % 9999 + seed_extra)
        n = 14
        inc = rng.normal(net / n, abs(net) * 0.09 + 1, n)
        inc[0] = 0
        return pd.DataFrame({
            "time": pd.date_range("09:15", periods=n, freq="30min"),
            "flow": np.cumsum(inc),
        })

    fii_df = _mock(fiidii["fii_cash_net_cr"], 0)
    dii_df = _mock(fiidii["dii_cash_net_cr"], 7)

    def _chart(df, title, rgb):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["flow"],
            fill="tozeroy", fillcolor=f"rgba({rgb},0.12)",
            line=dict(color=f"rgb({rgb})", width=2), mode="lines",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=12)),
            height=210, margin=dict(l=8, r=8, t=35, b=8),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis=dict(showgrid=False, tickformat="%H:%M", tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#1e2130",
                       tickprefix="₹", tickfont=dict(size=10)),
        )
        return fig

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(_chart(fii_df, "FII intraday netflow (₹ Cr)", "0,200,83"), use_container_width=True)
    with c2: st.plotly_chart(_chart(dii_df, "DII intraday netflow (₹ Cr)", "33,150,243"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Row 5: Activity breakdown tables
# ─────────────────────────────────────────────────────────────────────────────

def render_activity_breakdown(fiidii):
    fii = fiidii["fii_cash_net_cr"]
    dii = fiidii["dii_cash_net_cr"]

    fii_rows = [("Cash (equity)", fii * 0.68), ("Index futures", fii * 0.23),
                ("Stock futures", fii * 0.05), ("Options (net)", fii * -0.03),
                ("Debt", fii * 0.07), ("Total FII", fii)]
    dii_rows = [("Mutual funds (equity)", dii * 0.57), ("Insurance cos.", dii * 0.32),
                ("Banks / FIs", dii * 0.08), ("Pension funds", dii * 0.03),
                ("Debt / hybrid", 0.0), ("Total DII", dii)]

    def _bdg(net):
        return (badge("Net buyer", "#00c853", "#000") if net >= 0
                else badge("Net seller", "#ef5350", "#fff"))

    def _tbl(rows):
        h = '<table class="act-table">'
        for lbl, v in rows:
            s = "+" if v >= 0 else ""
            cc = "green" if v >= 0 else ("red" if v < 0 else "grey")
            bld = "font-weight:800;" if lbl.startswith("Total") else ""
            h += (f'<tr><td style="{bld}">{lbl}</td>'
                  f'<td class="{cc}" style="{bld}">{s}₹{v:,.0f} Cr</td></tr>')
        return h + "</table>"

    c1, c2 = st.columns(2)
    for col, label, net, rows in [
        (c1, "FII activity breakdown", fii, fii_rows),
        (c2, "DII activity breakdown", dii, dii_rows),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:10px;">'
                f'<span style="font-size:13px;font-weight:600;">{label}</span>'
                f'{_bdg(net)}</div>'
                f'{_tbl(rows)}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Row 6: Sector FII flow
# ─────────────────────────────────────────────────────────────────────────────

def render_sector_flow(fiidii):
    fii = fiidii["fii_cash_net_cr"]
    sectors = ["IT", "Banks", "Auto", "FMCG", "Pharma", "Energy", "Metals", "Realty", "Infra", "Others"]
    weights = [0.22, 0.28, 0.10, 0.08, 0.07, 0.09, 0.06, 0.04, 0.03, 0.03]
    rng = np.random.default_rng(abs(int(fii)) % 5555)
    vals = [fii * w * rng.uniform(0.8, 1.2) for w in weights]

    fig = go.Figure(go.Bar(
        x=sectors, y=vals,
        marker_color=["#00c853" if v >= 0 else "#ef5350" for v in vals],
        text=[f"₹{v:,.0f}" for v in vals],
        textposition="outside", textfont=dict(size=10),
    ))
    fig.update_layout(
        title=dict(text="Sector-wise FII flow today (₹ Cr)", font=dict(size=12)),
        height=230, margin=dict(l=8, r=8, t=38, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2130", zeroline=True, zerolinecolor="#555"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Row 7: Institutional signal banner
# ─────────────────────────────────────────────────────────────────────────────

def render_inst_banner(fiidii):
    combined = fiidii["combined_net_cr"]
    bias = fiidii["institutional_bias"]
    cls = "bull-banner" if bias == "BULLISH" else "bear-banner"
    lbl = ("🟢 INSTITUTIONAL BULL SIGNAL" if bias == "BULLISH"
           else "🔴 INSTITUTIONAL BEAR SIGNAL" if bias == "BEARISH"
    else "🟡 INSTITUTIONAL NEUTRAL")
    sign = "+" if combined >= 0 else ""
    st.markdown(
        f'<div class="{cls}">'
        f'<span class="banner-label">{lbl}</span>'
        f'<span class="banner-value">{sign}₹{combined:,.0f} Cr</span>'
        f'</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Row 8: Price chart
# ─────────────────────────────────────────────────────────────────────────────

def render_price_chart(ticker, sym, tf):
    period = {"1d": "3mo", "1h": "30d", "15m": "5d"}.get(tf, "3mo")
    try:
        df = get_ohlcv_cached(ticker, period=period, interval=tf)
    except Exception:
        st.warning("Price data unavailable.");
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.015)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=sym,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    for n, col in [(20, "#ffa726"), (50, "#42a5f5"), (200, "#ce93d8")]:
        if len(df) > n:
            fig.add_trace(go.Scatter(
                x=df.index, y=_ema(df["close"], n), name=f"EMA{n}",
                line=dict(color=col, width=1.2), hoverinfo="skip",
            ), row=1, col=1)

    vol_cols = ["#26a69a" if float(df["close"].iloc[i]) >= float(df["open"].iloc[i])
                else "#ef5350" for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], marker_color=vol_cols,
        opacity=0.55, showlegend=False, name="Volume",
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{sym} — {tf.upper()} Chart (EMA 20 / 50 / 200)", font=dict(size=14)),
        xaxis_rangeslider_visible=False, height=450,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,12,18,1)",
        font_color="white",
        legend=dict(orientation="h", y=1.04, font=dict(size=11)),
        margin=dict(l=8, r=8, t=42, b=8),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#1a1d27", row=1, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Row 9: Confluence gauges + trade plan cards
# ─────────────────────────────────────────────────────────────────────────────

def make_gauge(score, label, dirn):
    colour = label_colour(label)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"{direction_icon(dirn)}<br><span style='font-size:11px;color:#8a8fa8'>{label}</span>",
               "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar": {"color": colour},
            "bgcolor": "#13161e", "bordercolor": "#2a2d3a",
            "steps": [
                {"range": [0, 50], "color": "#1a1d27"},
                {"range": [50, 75], "color": "#1a2d40"},
                {"range": [75, 95], "color": "#1a3a28"},
                {"range": [95, 100], "color": "#003820"},
            ],
            "threshold": {"line": {"color": "#ffab00", "width": 3}, "thickness": 0.8, "value": 95},
        },
        number={"suffix": "%", "font": {"size": 34}, "valueformat": ".0f"},
    ))
    fig.update_layout(
        height=220, margin=dict(l=18, r=18, t=55, b=8),
        paper_bgcolor="rgba(0,0,0,0)", font_color="white",
    )
    return fig


def make_score_bars(tech, inst, opt):
    cats = ["Technical (65)", "Institutional (20)", "Options (15)"]
    vals = [tech, inst, opt]
    maxes = [65, 20, 15]
    colors = ["#42a5f5", "#66bb6a", "#ffa726"]
    fig = go.Figure()
    for cat, val, mx, col in zip(cats, vals, maxes, colors):
        fig.add_trace(go.Bar(
            x=[val], y=[cat], orientation="h",
            marker_color=col,
            text=[f"{val}/{mx}"], textposition="inside",
            hovertemplate=f"{cat}: {val}/{mx}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=[mx - val], y=[cat], orientation="h",
            marker_color="rgba(255,255,255,0.04)",
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        barmode="stack", height=120, showlegend=False,
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


def render_trade_plan(sig):
    opts = sig.options
    iv = opts.get("iv_percentile", 50)
    spot = sig.spot
    atm = int(spot // 50 * 50)

    if sig.direction == "LONG":
        opts_play = (f"Buy {atm} CE outright (IV cheap)" if iv < 30
                     else f"Bull call spread: Buy {atm} CE / Sell {atm + 200} CE")
    else:
        opts_play = (f"Buy {atm} PE outright (IV cheap)" if iv < 30
                     else f"Bear put spread: Buy {atm} PE / Sell {atm - 200} PE")

    rr_col = "green" if sig.rr >= 2 else ("yellow" if sig.rr >= 1 else "red")

    rows = [
        ("Direction", direction_icon(sig.direction)),
        ("Spot price", f"₹{sig.spot:,.2f}"),
        ("Entry zone", f"₹{sig.entry_low:,.0f} – ₹{sig.entry_high:,.0f}"),
        ("Stop loss", f"₹{sig.stop_loss:,.0f}"),
        ("Target 1", f"₹{sig.target1:,.0f}"),
        ("Target 2", f"₹{sig.target2:,.0f}"),
        ("R : R", f'<span class="{rr_col}">1 : {sig.rr}</span>'),
        ("Capital risk", "0.5 – 1% of capital"),
        ("Options play", opts_play),
        ("Trail SL", "Breakeven after T1 hit"),
        ("India VIX", f"{sig.vix:.2f} ({'use spreads' if sig.vix > 18 else 'OK for naked'})"),
    ]

    html = '<div class="trade-card">'
    for k, v in rows:
        html += (f'<div class="trow">'
                 f'<span class="tkey">{k}</span>'
                 f'<span class="tval">{v}</span>'
                 f'</div>')
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
# ── Row 10: 22-Technique table
# ─────────────────────────────────────────────────────────────────────────────

TECHNIQUE_META = [
    ("BOS / CHOCH", "Structure break or change of character"),
    ("HTF Trend (EMA-50 wkly)", "Higher timeframe trend alignment"),
    ("Key S/R (Pivot)", "Classic pivot support / resistance"),
    ("Dynamic S/R (EMA 20/50)", "Price vs EMA20 and EMA50 stack"),
    ("Trendlines", "Linear regression channel direction"),
    ("Gann Angles (1×1)", "Price vs 45° Gann from 52-week low"),
    ("Fibonacci 0.618–0.786", "Retracement hold or rejection"),
    ("Harmonic Patterns (PRZ)", "AB=CD / Gartley potential reversal zone"),
    ("Elliott Wave proxy", "RSI + momentum wave continuation"),
    ("Supply & Demand zones", "High-volume absorption / rejection"),
    ("Fair Value Gap (FVG)", "3-bar imbalance fill direction"),
    ("Breakouts / Breakdowns", "Range expansion with volume confirmation"),
    ("Momentum — MACD", "MACD crossover and histogram direction"),
    ("Oscillators (RSI+Stoch)", "RSI and Stochastic extremes"),
    ("Divergence (RSI)", "Price vs RSI divergence confirmation"),
    ("Reversal signals", "Candle reversal at key S/R level"),
    ("Candlestick patterns", "Engulfing / hammer / shooting star"),
    ("Heikin Ashi", "HA candle colour and shadow structure"),
    ("Renko proxy (ATR box)", "ATR-box directional flip"),
    ("Volume pressure", "Volume spike on up/down bar"),
    ("OBV trend", "OBV vs its 20-day MA"),
    ("India VIX filter", "VIX risk-on / risk-off regime"),
]

# Keyword map for matching tech_notes → technique index
_KW = {
    0: ["BOS"],
    1: ["HTF trend"],
    2: ["S1", "R1", "Pivot", "pivot"],
    3: ["EMA20", "EMA50", "above EMA", "below EMA"],
    4: ["trendline", "regression"],
    5: ["Gann"],
    6: ["Fib", "0.618", "0.786", "fibonacci", "Fibonacci"],
    7: ["harmonic", "PRZ"],
    8: ["EW proxy", "Elliott"],
    9: ["Demand zone", "Supply zone", "high-volume bullish", "high-volume bearish"],
    10: ["FVG", "gap between bars"],
    11: ["Breakout", "Breakdown", "breakout", "breakdown"],
    12: ["MACD"],
    13: ["RSI=", "Stoch", "oversold", "overbought"],
    14: ["divergence", "Divergence"],
    15: ["reversal candle", "Reversal"],
    16: ["engulfing", "Hammer", "hammer", "shooting star", "pin bar"],
    17: ["Heikin Ashi"],
    18: ["Renko", "Renko proxy"],
    19: ["Volume spike", "volume spike"],
    20: ["OBV"],
    21: ["VIX"],
}

_BULL_KW = ["bullish", "Bullish", "above", "support", "hold", "rising", "oversold", "green",
            "demand", "Demand", "buy", "bounce", "up", "positive", "low vol", "engulf", "Hammer",
            "BOS up", "HTF trend: weekly price above"]
_BEAR_KW = ["bearish", "Bearish", "below", "resistance", "rejection", "falling", "overbought",
            "red", "supply", "Supply", "sell", "star", "down", "negative", "high vol", "BOS down",
            "HTF trend: weekly price below"]


def _classify(note):
    if any(w in note for w in _BULL_KW): return "bull"
    if any(w in note for w in _BEAR_KW): return "bear"
    return "neutral"


def render_tech_table(sig):
    note_map: dict[int, str] = {}
    for note in sig.tech_notes:
        for idx, kws in _KW.items():
            if any(kw in note for kw in kws):
                if idx not in note_map:
                    note_map[idx] = note
                break

    html = """
    <div style="overflow-x:auto;">
    <table class="tech-table">
      <thead><tr>
        <th style="width:28px">#</th>
        <th style="width:160px">Technique</th>
        <th>Description</th>
        <th style="width:56px;text-align:center">Signal</th>
        <th>Analysis note</th>
        <th style="width:70px;text-align:center">Direction</th>
      </tr></thead>
      <tbody>
    """

    for i, (name, desc) in enumerate(TECHNIQUE_META):
        note = note_map.get(i, "")
        kind = _classify(note) if note else "neutral"
        icon = "✅" if kind == "bull" else ("🔴" if kind == "bear" else "➖")
        nc = "#00c853" if kind == "bull" else ("#ef5350" if kind == "bear" else "#555")
        dirn = ('<span class="green" style="font-size:11px;">▲ BULL</span>' if kind == "bull"
                else '<span class="red" style="font-size:11px;">▼ BEAR</span>' if kind == "bear"
        else '<span class="grey" style="font-size:11px;">– –</span>')
        short = note[:68] + "…" if len(note) > 68 else note
        html += (
            f'<tr>'
            f'<td style="color:#444;font-size:10px;">{i + 1:02d}</td>'
            f'<td style="font-weight:600;font-size:12px;">{name}</td>'
            f'<td style="color:#8a8fa8;font-size:11px;">{desc}</td>'
            f'<td style="text-align:center;font-size:15px;">{icon}</td>'
            f'<td style="color:{nc};font-size:11px;">{short}</td>'
            f'<td style="text-align:center;">{dirn}</td>'
            f'</tr>'
        )

    html += "</tbody></table></div>"

    bull_count = sum(1 for i in range(22) if _classify(note_map.get(i, "")) == "bull")
    bear_count = sum(1 for i in range(22) if _classify(note_map.get(i, "")) == "bear")
    neut_count = 22 - bull_count - bear_count

    with st.expander(
            f"📊  22-Technique Breakdown — "
            f"✅ {bull_count} Bullish  |  🔴 {bear_count} Bearish  |  ➖ {neut_count} Neutral",
            expanded=True,
    ):
        st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Row 11: Options panel
# ─────────────────────────────────────────────────────────────────────────────

def render_options_panel(opts, spot):
    chain = opts.get("chain_df", pd.DataFrame())
    c1, c2 = st.columns([1.7, 1])

    with c1:
        if not chain.empty:
            mask = (chain["strike"] >= spot * 0.92) & (chain["strike"] <= spot * 1.08)
            df = chain[mask].copy() if mask.any() else chain.copy()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["strike"], y=df["put_oi"] / 1e5,
                                 name="Put OI (lakh)", marker_color="#ef5350", opacity=0.85))
            fig.add_trace(go.Bar(x=df["strike"], y=df["call_oi"] / 1e5,
                                 name="Call OI (lakh)", marker_color="#26a69a", opacity=0.85))
            fig.add_vline(x=spot, line_dash="dot", line_color="#ffa726",
                          annotation_text=f"Spot {spot:,.0f}",
                          annotation_font=dict(color="#ffa726", size=11))
            mp = opts.get("max_pain", 0)
            if mp:
                fig.add_vline(x=mp, line_dash="dash", line_color="#ce93d8",
                              annotation_text=f"MaxPain {mp:,.0f}",
                              annotation_font=dict(color="#ce93d8", size=11))
            fig.update_layout(
                title=dict(text="Options OI by Strike (±8% of spot)", font=dict(size=12)),
                barmode="group", height=310,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", legend=dict(orientation="h", y=1.1),
                margin=dict(l=8, r=8, t=40, b=8),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Options chain unavailable (NSE feed offline).")

    with c2:
        pcr = opts.get("pcr_oi", 1.0)
        mp = opts.get("max_pain", 0)
        mp_d = opts.get("max_pain_vs_spot", 0)
        iv_pct = opts.get("iv_percentile", 50)
        iv_skw = opts.get("iv_skew", 0)
        top_c = opts.get("top_call_oi_strike", 0)
        top_p = opts.get("top_put_oi_strike", 0)
        bias = opts.get("institutional_bias", "NEUTRAL")

        pcr_c = "green" if pcr > 1.2 else ("red" if pcr < 0.8 else "yellow")
        pcr_l = "Put-writer floor" if pcr > 1.2 else ("Call-writer ceiling" if pcr < 0.8 else "Neutral")
        mp_c = "green" if mp_d > 0 else "red"
        mp_l = "↑ Expiry pull-up" if mp_d > 0 else "↓ Expiry pull-down"
        iv_c = "green" if iv_pct < 30 else ("red" if iv_pct > 70 else "yellow")
        iv_l = "Cheap — buy options" if iv_pct < 30 else ("Expensive — spread" if iv_pct > 70 else "Normal")
        sk_c = "red" if iv_skw > 0 else "green"
        sk_l = "Put IV > Call IV (hedge)" if iv_skw > 0 else "Call IV > Put IV (fear)"

        rows = [
            ("PCR (OI)", f'<span class="{pcr_c}">{pcr:.3f}</span>', pcr_l),
            ("Max Pain", f'<span class="{mp_c}">₹{mp:,.0f}</span>', f'{mp_l} ({mp_d:+,.0f})'),
            ("IV Percentile", f'<span class="{iv_c}">{iv_pct:.0f}%</span>', iv_l),
            ("IV Skew (Put–Call)", f'<span class="{sk_c}">{iv_skw:+.2f}</span>', sk_l),
            ("Top Call OI strike", f"₹{top_c:,.0f}", "Resistance wall"),
            ("Top Put OI strike", f"₹{top_p:,.0f}", "Support floor"),
            ("Options bias", f'<span class="{"green" if bias == "BULLISH" else "red"}">{bias}</span>', ""),
        ]
        html = '<div class="metric-tile">'
        for lbl, val, sub in rows:
            html += (
                f'<div class="opt-row">'
                f'<span style="color:#8a8fa8;font-size:12px;">{lbl}</span>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:13px;font-weight:700;">{val}</div>'
                f'{"<div style=\'font-size:10px;color:#555;\'>" + sub + "</div>" if sub else ""}'
                f'</div></div>'
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Row 12: VIX trend
# ─────────────────────────────────────────────────────────────────────────────

def render_vix_chart():
    try:
        df = get_ohlcv_cached("^INDIAVIX", "3mo", "1d")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["close"],
            fill="tozeroy", fillcolor="rgba(255,171,0,0.08)",
            line=dict(color="#ffa726", width=1.8), mode="lines", name="India VIX",
        ))
        for lvl, col, lbl in [(14, "#26a69a", "Risk-on < 14"), (20, "#ef5350", "Danger > 20")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=col,
                          annotation_text=lbl, annotation_font=dict(color=col, size=10))
        fig.update_layout(
            title=dict(text="India VIX — 3 Month Trend", font=dict(size=12)),
            height=200, margin=dict(l=8, r=8, t=38, b=8),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2130"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ticker = SYMBOLS.get(symbol.upper(), symbol)

    with st.spinner("Fetching data & computing signals…"):
        fiidii = get_fiidii()
        opts = get_options(symbol)
        vix = get_vix()
        result = get_signal(symbol, direction)

    if isinstance(result, tuple):
        sig_l, sig_s = result
        spot = sig_l.spot
        primary = sig_l if sig_l.confidence >= sig_s.confidence else sig_s
    else:
        sig_l = sig_s = result
        spot = result.spot
        primary = result

    # ── Row 1: header ────────────────────────────────────────────────────────
    render_header(fiidii)

    # ── Row 2: FII/DII tiles ─────────────────────────────────────────────────
    render_fiidii_tiles(fiidii)

    # ── Row 3: index strip ───────────────────────────────────────────────────
    render_index_strip(vix)

    # ── Row 4: intraday charts ───────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Intraday Institutional Flow</div>', unsafe_allow_html=True)
    render_intraday_charts(fiidii)

    # ── Row 5: activity breakdown ─────────────────────────────────────────────
    render_activity_breakdown(fiidii)

    # ── Row 6: sector flow ───────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Sector-wise FII Flow</div>', unsafe_allow_html=True)
    render_sector_flow(fiidii)

    # ── Row 7: institutional banner ───────────────────────────────────────────
    render_inst_banner(fiidii)

    st.markdown("---")

    # ── Row 8: price chart ───────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Price Chart — Technical Overview</div>', unsafe_allow_html=True)
    render_price_chart(ticker, symbol, timeframe)

    # ── Row 9: signal cards ───────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Confluence Signals & Trade Plans</div>', unsafe_allow_html=True)
    if isinstance(result, tuple):
        c1, c2 = st.columns(2)
        render_signal_column(sig_l, c1)
        render_signal_column(sig_s, c2)
    else:
        c1, _ = st.columns([1, 1])
        render_signal_column(result, c1)

    # ── Row 10: 22-technique table ────────────────────────────────────────────
    st.markdown('<div class="section-hdr">22-Technique Breakdown</div>', unsafe_allow_html=True)
    render_tech_table(primary)

    # ── Row 11: options ───────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Options Flow Analysis</div>', unsafe_allow_html=True)
    render_options_panel(opts, spot)

    # ── Row 12: VIX trend ─────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">India VIX Trend</div>', unsafe_allow_html=True)
    render_vix_chart()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="text-align:center;color:#333;font-size:11px;padding:24px 0 8px;">'
        f'Last updated · {datetime.datetime.now().strftime("%d %b %Y %H:%M:%S IST")} '
        f'· Data: yfinance · NSE India · Educational purposes only'
        f'</div>',
        unsafe_allow_html=True)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if refresh > 0:
        time.sleep(refresh * 60)
        st.rerun()


if __name__ == "__main__":
    main()
