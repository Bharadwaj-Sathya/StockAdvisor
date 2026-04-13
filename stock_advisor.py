"""
stock_advisor.py
----------------
Core pipeline — downloads OHLCV data, scores 22 technical techniques,
combines with FII/DII and Options layers, and fires BUY / SELL signals.

Usage
-----
    python stock_advisor.py --symbol NIFTY --direction both
    python stock_advisor.py --symbol SENSEX --direction long
    python stock_advisor.py --symbol NIFTY  --direction short
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    import pandas_ta as ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False
    print("[stock_advisor] WARNING: pandas_ta not installed — some indicators use fallbacks.")

from fii_dii_feed import fetch_fiidii_data
from options_feed import fetch_options_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOLS = {
    "NIFTY":   "^NSEI",
    "SENSEX":  "^BSESN",
    "BANKNIFTY": "^NSEBANK",
}
VIX_TICKER = "^INDIAVIX"

THRESHOLD_HIGH   = 95   # ≥ 95 → HIGH-CONFIDENCE signal
THRESHOLD_WATCH  = 75   # 75–94 → WATCH
THRESHOLD_PARTIAL = 50  # 50–74 → PARTIAL


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def download_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False, multi_level_index=False)
    df.columns = [c.lower() for c in df.columns]
    df.dropna(inplace=True)
    return df


def download_vix() -> float:
    try:
        vix = yf.download(VIX_TICKER, period="5d", interval="1d",
                          auto_adjust=True, progress=False, multi_level_index=False)
        return float(vix["Close"].iloc[-1])
    except Exception:
        return 15.0


# ---------------------------------------------------------------------------
# Technical indicator helpers (pandas-ta or manual fallbacks)
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    if _HAS_TA:
        return ta.rsi(s, length=n)
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(s: pd.Series):
    fast, slow, sig = _ema(s, 12), _ema(s, 26), None
    macd_line = fast - slow
    sig_line  = _ema(macd_line, 9)
    return macd_line, sig_line


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    if _HAS_TA:
        stoch = ta.stoch(high, low, close, k=k, d=d)
        return stoch.iloc[:, 0], stoch.iloc[:, 1]
    lowest_low   = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    pct_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    pct_d = pct_k.rolling(d).mean()
    return pct_k, pct_d


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open  = ((df["open"] + df["close"]) / 2).shift(1)
    ha_high  = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low   = pd.concat([df["low"],  ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close})


def _fibonacci_levels(high: float, low: float) -> dict:
    diff = high - low
    return {
        0.0:   high,
        0.236: high - 0.236 * diff,
        0.382: high - 0.382 * diff,
        0.500: high - 0.500 * diff,
        0.618: high - 0.618 * diff,
        0.786: high - 0.786 * diff,
        1.0:   low,
    }


# ---------------------------------------------------------------------------
# 22-Technique scorer
# ---------------------------------------------------------------------------

@dataclass
class TechScore:
    long:  int = 0
    short: int = 0
    notes: list[str] = field(default_factory=list)

    def add(self, long_pts: int, short_pts: int, note: str):
        self.long  += long_pts
        self.short += short_pts
        if note:
            self.notes.append(note)

    @property
    def max_points(self) -> int:
        return 65


def score_technical(df: pd.DataFrame, htf_df: pd.DataFrame) -> TechScore:
    """
    Score all 22 techniques.  Returns a TechScore (long/short, 0–65 each).

    *df*     = primary timeframe daily OHLCV
    *htf_df* = higher timeframe weekly OHLCV
    """
    sc   = TechScore()
    c    = df["close"]
    h    = df["high"]
    lo   = df["low"]
    o    = df["open"]
    vol  = df["volume"]
    last = float(c.iloc[-1])

    # ── 1. BOS / CHOCH ──────────────────────────────────────────────────────
    recent_high = float(h.iloc[-20:-1].max())
    recent_low  = float(lo.iloc[-20:-1].min())
    if last > recent_high:
        sc.add(15, 0, "BOS up: price broke recent 20-bar high (bullish BOS)")
    elif last < recent_low:
        sc.add(0, 15, "BOS down: price broke recent 20-bar low (bearish BOS)")
    else:
        sc.add(0, 0, "")

    # ── 2. HTF trend ────────────────────────────────────────────────────────
    htf_c     = htf_df["close"]
    htf_ema50 = _ema(htf_c, 50)
    htf_trend = "up" if float(htf_c.iloc[-1]) > float(htf_ema50.iloc[-1]) else "down"
    if htf_trend == "up":
        sc.add(5, 0, "HTF trend: weekly price above EMA-50 (uptrend)")
    else:
        sc.add(0, 5, "HTF trend: weekly price below EMA-50 (downtrend)")

    # ── 3. Key S/R ──────────────────────────────────────────────────────────
    pivot = (float(h.iloc[-1]) + float(lo.iloc[-1]) + last) / 3
    r1    = 2 * pivot - float(lo.iloc[-1])
    s1    = 2 * pivot - float(h.iloc[-1])
    if abs(last - s1) / last < 0.005:
        sc.add(5, 0, f"Key support S1={s1:.1f} — price bouncing")
    elif abs(last - r1) / last < 0.005:
        sc.add(0, 5, f"Key resistance R1={r1:.1f} — price rejecting")

    # ── 4. Dynamic S/R — EMA 20/50 ──────────────────────────────────────────
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
    if last > e20 > e50:
        sc.add(3, 0, f"Price above EMA20({e20:.0f}) & EMA50({e50:.0f})")
    elif last < e20 < e50:
        sc.add(0, 3, f"Price below EMA20({e20:.0f}) & EMA50({e50:.0f})")

    # ── 5. Trendlines (simplified: price vs linear regression channel) ───────
    x   = np.arange(len(c))
    m, b = np.polyfit(x, c.values, 1)
    trend_val = m * x[-1] + b
    if m > 0 and last > trend_val:
        sc.add(2, 0, "Ascending trendline intact — price above regression")
    elif m < 0 and last < trend_val:
        sc.add(0, 2, "Descending trendline — price below regression")

    # ── 6. Gann angles ──────────────────────────────────────────────────────
    # 1x1 = 45°. Simplified as slope of price from 52-week low.
    # Use a direct slice (up to 252 bars) and positional argmin — no rolling NaN.
    lookback_lo = lo.iloc[-252:] if len(lo) >= 252 else lo
    low_52w     = float(lookback_lo.min())
    _argmin_pos = int(lookback_lo.values.argmin())          # 0-based position in slice
    bars_ago    = len(lookback_lo) - 1 - _argmin_pos        # bars since the low
    if bars_ago > 0:
        gann_1x1 = low_52w + bars_ago * (last - low_52w) / bars_ago  # simplifies to last; keep formula explicit
        if last >= gann_1x1:
            sc.add(2, 0, "Price above 1×1 Gann angle from 52-week low")
        else:
            sc.add(0, 2, "Price below 1×1 Gann angle from 52-week low")

    # ── 7. Fibonacci retracements ────────────────────────────────────────────
    swing_high = float(h.rolling(60).max().iloc[-1])
    swing_low  = float(lo.rolling(60).min().iloc[-1])
    fibs       = _fibonacci_levels(swing_high, swing_low)
    tol        = (swing_high - swing_low) * 0.01
    near_618   = abs(last - fibs[0.618]) < tol
    near_786   = abs(last - fibs[0.786]) < tol
    if near_618 or near_786:
        if last > fibs[0.786]:
            sc.add(7, 0, f"Price holding Fib 0.618–0.786 zone ({fibs[0.618]:.0f})")
        else:
            sc.add(0, 7, f"Price rejected at Fib 0.618 zone ({fibs[0.618]:.0f})")

    # ── 8. Harmonic patterns (simplified: AB=CD check) ───────────────────────
    # Very simplified heuristic — real harmonics need swing-point detection
    if abs(last - fibs[0.618]) < tol:
        sc.add(2, 0, "Potential bullish harmonic PRZ near 0.618")
    elif abs(last - fibs[0.236]) < tol:
        sc.add(0, 2, "Potential bearish harmonic PRZ near 0.236 retracement")

    # ── 9. Elliott Wave (heuristic wave-3 proxy) ─────────────────────────────
    # Proxy: strong momentum continuation after a 3-wave correction
    rsi14 = _rsi(c, 14)
    rsi_v = float(rsi14.iloc[-1])
    if rsi_v > 60 and float(c.iloc[-1]) > float(c.iloc[-5]):
        sc.add(2, 0, f"EW proxy: RSI={rsi_v:.1f} + price advancing — possible wave 3 up")
    elif rsi_v < 40 and float(c.iloc[-1]) < float(c.iloc[-5]):
        sc.add(0, 2, f"EW proxy: RSI={rsi_v:.1f} + price falling — possible wave 3 down")

    # ── 10. Supply & demand zones ────────────────────────────────────────────
    # Identify high-volume consolidation areas
    avg_vol = float(vol.rolling(20).mean().iloc[-1])
    high_vol_bars = vol.rolling(20).max()
    if float(vol.iloc[-1]) > 1.5 * avg_vol and last > float(c.iloc[-2]):
        sc.add(3, 0, "Demand zone absorption: high-volume bullish bar")
    elif float(vol.iloc[-1]) > 1.5 * avg_vol and last < float(c.iloc[-2]):
        sc.add(0, 3, "Supply zone: high-volume bearish bar")

    # ── 11. Fair value gap (FVG) ─────────────────────────────────────────────
    # Bullish FVG: low[-1] > high[-3]   Bearish FVG: high[-1] < low[-3]
    if len(df) >= 4:
        if float(lo.iloc[-1]) > float(h.iloc[-3]):
            sc.add(3, 0, "Bullish FVG: gap between bars -3 and -1")
        elif float(h.iloc[-1]) < float(lo.iloc[-3]):
            sc.add(0, 3, "Bearish FVG: gap between bars -3 and -1")

    # ── 12. Breakouts ────────────────────────────────────────────────────────
    consolidation_high = float(h.iloc[-10:-1].max())
    consolidation_low  = float(lo.iloc[-10:-1].min())
    if last > consolidation_high and float(vol.iloc[-1]) > avg_vol:
        sc.add(4, 0, f"Breakout above {consolidation_high:.0f} with volume")
    elif last < consolidation_low and float(vol.iloc[-1]) > avg_vol:
        sc.add(0, 4, f"Breakdown below {consolidation_low:.0f} with volume")

    # ── 13. Momentum indicators (MACD) ───────────────────────────────────────
    macd_line, sig_line = _macd(c)
    m_now   = float(macd_line.iloc[-1])
    m_prev  = float(macd_line.iloc[-2])
    s_now   = float(sig_line.iloc[-1])
    s_prev  = float(sig_line.iloc[-2])
    if m_prev < s_prev and m_now > s_now:
        sc.add(3, 0, "MACD bullish crossover")
    elif m_prev > s_prev and m_now < s_now:
        sc.add(0, 3, "MACD bearish crossover")
    elif m_now > 0:
        sc.add(1, 0, "MACD histogram positive")
    elif m_now < 0:
        sc.add(0, 1, "MACD histogram negative")

    # ── 14. Oscillators (RSI + Stochastic) ───────────────────────────────────
    stoch_k, stoch_d = _stochastic(h, lo, c)
    sk, sd = float(stoch_k.iloc[-1]), float(stoch_d.iloc[-1])
    if rsi_v < 35 and sk < 20:
        sc.add(6, 0, f"RSI={rsi_v:.1f} oversold + Stoch K={sk:.1f} oversold")
    elif rsi_v > 65 and sk > 80:
        sc.add(0, 6, f"RSI={rsi_v:.1f} overbought + Stoch K={sk:.1f} overbought")
    elif rsi_v < 45:
        sc.add(2, 0, f"RSI={rsi_v:.1f} moderately oversold")
    elif rsi_v > 55:
        sc.add(0, 2, f"RSI={rsi_v:.1f} moderately overbought")

    # ── 15. Divergence ───────────────────────────────────────────────────────
    # Bullish: price lower low, RSI higher low  |  Bearish: price higher high, RSI lower high
    if len(rsi14) >= 10:
        price_ll = float(lo.iloc[-1]) < float(lo.iloc[-10:-2].min())
        rsi_hl   = float(rsi14.iloc[-1]) > float(rsi14.iloc[-10:-2].min())
        price_hh = float(h.iloc[-1])  > float(h.iloc[-10:-2].max())
        rsi_lh   = float(rsi14.iloc[-1]) < float(rsi14.iloc[-10:-2].max())
        if price_ll and rsi_hl:
            sc.add(6, 0, "Bullish RSI divergence (price lower low, RSI higher low)")
        elif price_hh and rsi_lh:
            sc.add(0, 6, "Bearish RSI divergence (price higher high, RSI lower high)")

    # ── 16. Reversal signals ─────────────────────────────────────────────────
    # Near S/R with opposite candle
    near_support    = (last - float(lo.rolling(20).min().iloc[-1])) / last < 0.01
    near_resistance = (float(h.rolling(20).max().iloc[-1]) - last) / last < 0.01
    bull_candle = float(c.iloc[-1]) > float(o.iloc[-1])
    bear_candle = float(c.iloc[-1]) < float(o.iloc[-1])
    if near_support and bull_candle:
        sc.add(3, 0, "Bullish reversal candle at 20-bar support")
    elif near_resistance and bear_candle:
        sc.add(0, 3, "Bearish reversal candle at 20-bar resistance")

    # ── 17. Candlestick patterns ─────────────────────────────────────────────
    body      = abs(float(c.iloc[-1]) - float(o.iloc[-1]))
    candle_rng = float(h.iloc[-1]) - float(lo.iloc[-1])
    lower_wick = float(min(o.iloc[-1], c.iloc[-1])) - float(lo.iloc[-1])
    upper_wick = float(h.iloc[-1]) - float(max(o.iloc[-1], c.iloc[-1]))
    # Bullish engulfing
    if (float(c.iloc[-1]) > float(o.iloc[-1])
            and float(o.iloc[-1]) < float(c.iloc[-2])
            and float(c.iloc[-1]) > float(o.iloc[-2])):
        sc.add(4, 0, "Bullish engulfing candle")
    # Bearish engulfing
    elif (float(c.iloc[-1]) < float(o.iloc[-1])
            and float(o.iloc[-1]) > float(c.iloc[-2])
            and float(c.iloc[-1]) < float(o.iloc[-2])):
        sc.add(0, 4, "Bearish engulfing candle")
    # Hammer
    elif lower_wick > 2 * body and upper_wick < body:
        sc.add(3, 0, "Hammer candle (bullish reversal)")
    # Shooting star
    elif upper_wick > 2 * body and lower_wick < body:
        sc.add(0, 3, "Shooting star candle (bearish reversal)")

    # ── 18. Heikin Ashi ──────────────────────────────────────────────────────
    ha = _heikin_ashi(df)
    ha_green = float(ha["close"].iloc[-1]) > float(ha["open"].iloc[-1])
    ha_no_lower = float(ha["low"].iloc[-1]) >= float(ha["open"].iloc[-1])
    ha_no_upper = float(ha["high"].iloc[-1]) <= float(ha["open"].iloc[-1])
    if ha_green and ha_no_lower:
        sc.add(2, 0, "Heikin Ashi: green candle, no lower shadow — strong uptrend")
    elif not ha_green and ha_no_upper:
        sc.add(0, 2, "Heikin Ashi: red candle, no upper shadow — strong downtrend")

    # ── 19. Renko (ATR-based box flip proxy) ─────────────────────────────────
    atr14 = _atr(h, lo, c, 14)
    box   = float(atr14.iloc[-1])
    if last > float(c.iloc[-3]) + box:
        sc.add(2, 0, f"Renko proxy: price advanced > 1 ATR box ({box:.0f})")
    elif last < float(c.iloc[-3]) - box:
        sc.add(0, 2, f"Renko proxy: price fell > 1 ATR box ({box:.0f})")

    # ── 20. Volume ───────────────────────────────────────────────────────────
    vol_ratio = float(vol.iloc[-1]) / avg_vol if avg_vol > 0 else 1
    if vol_ratio > 1.5 and last > float(c.iloc[-2]):
        sc.add(3, 0, f"Volume spike (×{vol_ratio:.1f}) on up-bar — buying pressure")
    elif vol_ratio > 1.5 and last < float(c.iloc[-2]):
        sc.add(0, 3, f"Volume spike (×{vol_ratio:.1f}) on down-bar — selling pressure")

    # ── 21. OBV trend ────────────────────────────────────────────────────────
    obv = _obv(c, vol)
    obv_ma = obv.rolling(20).mean()
    if float(obv.iloc[-1]) > float(obv_ma.iloc[-1]) and last > float(c.iloc[-5]):
        sc.add(2, 0, "OBV rising above its 20-day MA — accumulation")
    elif float(obv.iloc[-1]) < float(obv_ma.iloc[-1]) and last < float(c.iloc[-5]):
        sc.add(0, 2, "OBV falling below its 20-day MA — distribution")

    # ── 22. India VIX ───────────────────────────────────────────────────────
    # Handled in main scorer (requires separate download)
    # Placeholder — main() injects VIX score after calling this function.

    return sc


# ---------------------------------------------------------------------------
# Confluence scorer
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    direction:   str    # "LONG" | "SHORT"
    confidence:  float  # 0–100
    label:       str    # "HIGH-CONFIDENCE" | "WATCH" | "PARTIAL" | "NO SIGNAL"
    tech_score:  int
    inst_score:  int
    opt_score:   int
    total_score: int
    spot:        float
    entry_low:   float
    entry_high:  float
    stop_loss:   float
    target1:     float
    target2:     float
    rr:          float
    tech_notes:  list[str]
    fiidii:      dict
    options:     dict
    vix:         float


def compute_signal(
    symbol: str = "NIFTY",
    direction: str = "both",
) -> Signal | tuple[Signal, Signal]:

    ticker    = SYMBOLS.get(symbol.upper(), symbol)
    df        = download_ohlcv(ticker, period="1y",  interval="1d")
    htf_df    = download_ohlcv(ticker, period="3y",  interval="1wk")
    vix       = download_vix()
    fiidii    = fetch_fiidii_data()
    opts      = fetch_options_data(symbol.upper() if symbol.upper() in ("NIFTY","BANKNIFTY") else "NIFTY")

    spot = float(df["close"].iloc[-1])

    # Technical scoring
    tech = score_technical(df, htf_df)

    # VIX adjustment (technique 22)
    if vix < 14:
        tech.long  += 2
        tech.notes.append(f"VIX={vix:.1f} — risk-on (low vol)")
    elif vix > 20:
        tech.short += 2
        tech.notes.append(f"VIX={vix:.1f} — risk-off (elevated vol)")

    # Cap at 65
    tech.long  = min(tech.long,  65)
    tech.short = min(tech.short, 65)

    def _build(dirn: str) -> Signal:
        if dirn == "LONG":
            t_sc   = tech.long
            i_sc   = fiidii["score_long"]
            o_sc   = opts["score_long"]
        else:
            t_sc   = tech.short
            i_sc   = fiidii["score_short"]
            o_sc   = opts["score_short"]

        total = t_sc + i_sc + o_sc
        conf  = round(total / 100 * 100, 1)  # max 100

        if conf >= THRESHOLD_HIGH:
            label = "HIGH-CONFIDENCE"
        elif conf >= THRESHOLD_WATCH:
            label = "WATCH"
        elif conf >= THRESHOLD_PARTIAL:
            label = "PARTIAL"
        else:
            label = "NO SIGNAL"

        # Trade plan
        atr14 = float(_atr(df["high"], df["low"], df["close"], 14).iloc[-1])

        if dirn == "LONG":
            entry_low  = round(spot - atr14 * 0.3, 0)
            entry_high = round(spot + atr14 * 0.1, 0)
            stop_loss  = round(spot - atr14 * 1.5, 0)
            target1    = round(spot + atr14 * 1.5, 0)
            target2    = round(spot + atr14 * 3.0, 0)
        else:
            entry_low  = round(spot - atr14 * 0.1, 0)
            entry_high = round(spot + atr14 * 0.3, 0)
            stop_loss  = round(spot + atr14 * 1.5, 0)
            target1    = round(spot - atr14 * 1.5, 0)
            target2    = round(spot - atr14 * 3.0, 0)

        risk   = abs(spot - stop_loss)
        reward = abs(target2 - spot)
        rr     = round(reward / risk, 1) if risk > 0 else 0

        return Signal(
            direction   = dirn,
            confidence  = conf,
            label       = label,
            tech_score  = t_sc,
            inst_score  = i_sc,
            opt_score   = o_sc,
            total_score = total,
            spot        = spot,
            entry_low   = entry_low,
            entry_high  = entry_high,
            stop_loss   = stop_loss,
            target1     = target1,
            target2     = target2,
            rr          = rr,
            tech_notes  = tech.notes,
            fiidii      = fiidii,
            options     = opts,
            vix         = vix,
        )

    if direction == "long":
        return _build("LONG")
    elif direction == "short":
        return _build("SHORT")
    else:
        sig_l = _build("LONG")
        sig_s = _build("SHORT")
        return (sig_l, sig_s)


# ---------------------------------------------------------------------------
# Trade plan printer
# ---------------------------------------------------------------------------

def print_signal(sig: Signal, symbol: str):
    sep = "=" * 60
    pcr  = sig.options.get("pcr_oi", 0)
    mp   = sig.options.get("max_pain", 0)
    iv   = sig.options.get("iv_percentile", 50)

    opts_play = ""
    if sig.direction == "LONG":
        if iv < 30:
            opts_play = (
                f"Bull call spread — Buy {int(sig.spot // 50 * 50)} CE, "
                f"Sell {int(sig.spot // 50 * 50) + 200} CE (nearest expiry)\n"
                f"           OR: Buy {int(sig.spot // 50 * 50)} CE outright (IV cheap)"
            )
        else:
            opts_play = (
                f"Bull call spread — Buy {int(sig.spot // 50 * 50)} CE, "
                f"Sell {int(sig.spot // 50 * 50) + 200} CE (spread; IV elevated)"
            )
    else:
        if iv < 30:
            opts_play = (
                f"Bear put spread — Buy {int(sig.spot // 50 * 50)} PE, "
                f"Sell {int(sig.spot // 50 * 50) - 200} PE (nearest expiry)\n"
                f"           OR: Buy {int(sig.spot // 50 * 50)} PE outright (IV cheap)"
            )
        else:
            opts_play = (
                f"Bear put spread — Buy {int(sig.spot // 50 * 50)} PE, "
                f"Sell {int(sig.spot // 50 * 50) - 200} PE (spread; IV elevated)"
            )

    print(f"\n{sep}")
    print(f"Signal   : {sig.label} {sig.direction}  ({sig.confidence:.0f}% confluence)")
    print(f"Symbol   : {symbol.upper()}  |  Spot: {sig.spot:,.0f}")
    print(f"{sep}")
    print(f"Scoring breakdown:")
    print(f"  Technical   : {sig.tech_score:>3} / 65")
    print(f"  Institutional: {sig.inst_score:>2} / 20  "
          f"(FII+DII net: ₹{sig.fiidii['combined_net_cr']:+,.0f} Cr)")
    print(f"  Options     : {sig.opt_score:>3} / 15  "
          f"(PCR={pcr:.2f}, MaxPain={mp:,.0f}, IVpct={iv:.0f})")
    print(f"  TOTAL       : {sig.total_score:>3} / 100")
    print(f"\nTrade Plan:")
    print(f"  Entry    : {sig.entry_low:,.0f} – {sig.entry_high:,.0f}")
    print(f"  Stop loss: {sig.stop_loss:,.0f}")
    print(f"  Target 1 : {sig.target1:,.0f}")
    print(f"  Target 2 : {sig.target2:,.0f}")
    print(f"  R:R      : 1:{sig.rr}")
    print(f"  Risk     : 0.5–1% of capital")
    print(f"\nOptions  : {opts_play}")
    print(f"\nVIX      : {sig.vix:.1f}")
    print(f"\nKey technicals:")
    for note in sig.tech_notes[:10]:
        print(f"  • {note}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Advisor — BUY/SELL Signal Engine")
    parser.add_argument("--symbol",    default="NIFTY",
                        help="NIFTY | SENSEX | BANKNIFTY (default: NIFTY)")
    parser.add_argument("--direction", default="both",
                        choices=["long", "short", "both"],
                        help="Signal direction (default: both)")
    args = parser.parse_args()

    print(f"\n[stock_advisor] Analysing {args.symbol.upper()} — direction: {args.direction}")

    result = compute_signal(args.symbol, args.direction)

    if isinstance(result, tuple):
        sig_l, sig_s = result
        print_signal(sig_l, args.symbol)
        print_signal(sig_s, args.symbol)
        # Highlight the stronger side
        winner = sig_l if sig_l.confidence >= sig_s.confidence else sig_s
        print(f"\n>>> Active signal: {winner.label} {winner.direction} "
              f"({winner.confidence:.0f}% confluence)")
    else:
        print_signal(result, args.symbol)