"""
options_feed.py
---------------
Parses the NSE F&O options chain for Nifty 50 / Sensex.

Computes:
  - PCR (Put-Call Ratio based on Open Interest)
  - Max pain level
  - Call vs Put OI buildup at key strikes
  - IV percentile
  - IV skew (put IV minus call IV)

Scores each metric for LONG and SHORT confluence.
"""

from __future__ import annotations

import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# NSE session helper
# ---------------------------------------------------------------------------
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

OPTIONS_URL = "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
VIX_TICKER  = "^INDIAVIX"


def _nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
    except Exception:
        pass
    return session


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_options_data(symbol: str = "NIFTY", expiry_index: int = 0) -> dict:
    """
    Fetch and analyse the options chain for *symbol*.

    Parameters
    ----------
    symbol       : "NIFTY" or "BANKNIFTY"
    expiry_index : 0 = nearest weekly/monthly expiry, 1 = next, etc.

    Returns
    -------
    dict with keys:
        symbol            : str
        expiry            : str   (DD-Mon-YYYY)
        spot_price        : float
        pcr_oi            : float
        max_pain          : float
        max_pain_vs_spot  : float  (max_pain - spot; positive → expiry pull-up)
        iv_percentile     : float  (0–100)
        iv_skew           : float  (avg put IV - avg call IV)
        top_call_oi_strike: float
        top_put_oi_strike : float
        institutional_bias: str   ("BULLISH" | "BEARISH" | "NEUTRAL")
        score_long        : int   (0–15)
        score_short       : int   (0–15)
        chain_df          : pd.DataFrame
    """
    session = _nse_session()
    url = OPTIONS_URL.format(symbol=symbol.upper())
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[options_feed] WARNING: Could not fetch live chain — {exc}")
        return _fallback_result(symbol)

    try:
        result = _analyse_chain(data, expiry_index)
        result["symbol"] = symbol
        return result
    except Exception as exc:
        print(f"[options_feed] WARNING: Analysis error — {exc}")
        return _fallback_result(symbol)


# ---------------------------------------------------------------------------
# Chain analysis
# ---------------------------------------------------------------------------

def _analyse_chain(data: dict, expiry_index: int) -> dict:
    records     = data["records"]
    expiry_dates = sorted(records["expiryDates"])
    expiry       = expiry_dates[min(expiry_index, len(expiry_dates) - 1)]
    spot_price   = float(records["underlyingValue"])

    rows = []
    for entry in records["data"]:
        if entry.get("expiryDate") != expiry:
            continue
        strike = float(entry["strikePrice"])
        ce = entry.get("CE", {})
        pe = entry.get("PE", {})
        rows.append({
            "strike":    strike,
            "call_oi":   float(ce.get("openInterest", 0)),
            "call_iv":   float(ce.get("impliedVolatility", 0)),
            "call_ltp":  float(ce.get("lastPrice", 0)),
            "put_oi":    float(pe.get("openInterest", 0)),
            "put_iv":    float(pe.get("impliedVolatility", 0)),
            "put_ltp":   float(pe.get("lastPrice", 0)),
        })

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    if df.empty:
        raise ValueError("Empty chain after filtering expiry.")

    # ── PCR (OI-based) ──────────────────────────────────────────────────────
    total_call_oi = df["call_oi"].sum()
    total_put_oi  = df["put_oi"].sum()
    pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

    # ── Max pain ────────────────────────────────────────────────────────────
    max_pain = _compute_max_pain(df)

    # ── IV percentile & skew ────────────────────────────────────────────────
    call_ivs = df.loc[df["call_iv"] > 0, "call_iv"].values
    put_ivs  = df.loc[df["put_iv"]  > 0, "put_iv"].values
    all_ivs  = np.concatenate([call_ivs, put_ivs])
    # Simple cross-sectional percentile of ATM-ish strikes
    atm_mask = (df["strike"] >= spot_price * 0.95) & (df["strike"] <= spot_price * 1.05)
    atm_call_iv = df.loc[atm_mask, "call_iv"].mean() if atm_mask.any() else 0
    atm_put_iv  = df.loc[atm_mask, "put_iv"].mean()  if atm_mask.any() else 0
    avg_atm_iv  = (atm_call_iv + atm_put_iv) / 2

    iv_percentile = (
        float(np.percentile(all_ivs, [25, 50, 75, 100], method="linear")[1])
        if len(all_ivs) >= 4 else 50.0
    )
    # Re-express as percentile rank of avg_atm_iv within all_ivs
    if len(all_ivs) > 0:
        iv_pct_rank = float(np.mean(all_ivs <= avg_atm_iv) * 100)
    else:
        iv_pct_rank = 50.0

    iv_skew = float(atm_put_iv - atm_call_iv)  # positive → put IV > call IV

    # ── Top OI strikes ──────────────────────────────────────────────────────
    top_call_strike = float(df.loc[df["call_oi"].idxmax(), "strike"])
    top_put_strike  = float(df.loc[df["put_oi"].idxmax(),  "strike"])

    # ── Scoring ─────────────────────────────────────────────────────────────
    score_long, score_short = _score(
        pcr, max_pain, spot_price, iv_pct_rank, iv_skew
    )

    bias = _classify_options_bias(pcr, max_pain, spot_price)

    return {
        "expiry":             expiry,
        "spot_price":         spot_price,
        "pcr_oi":             round(pcr, 3),
        "max_pain":           max_pain,
        "max_pain_vs_spot":   round(max_pain - spot_price, 2),
        "iv_percentile":      round(iv_pct_rank, 1),
        "iv_skew":            round(iv_skew, 2),
        "top_call_oi_strike": top_call_strike,
        "top_put_oi_strike":  top_put_strike,
        "institutional_bias": bias,
        "score_long":         score_long,
        "score_short":        score_short,
        "chain_df":           df,
    }


# ---------------------------------------------------------------------------
# Max pain calculation
# ---------------------------------------------------------------------------

def _compute_max_pain(df: pd.DataFrame) -> float:
    """
    For each strike S, compute total $ loss to option writers if price expires at S.
    Max pain = strike where writer losses are minimised (i.e. min total OI pain).
    """
    strikes = df["strike"].values
    pain = []
    for s in strikes:
        # Call writer loss: sum over all strikes K < s of (s - K) * call_OI(K)
        call_loss = np.sum(
            np.where(strikes < s, (s - strikes) * df["call_oi"].values, 0)
        )
        # Put writer loss: sum over all strikes K > s of (K - s) * put_OI(K)
        put_loss = np.sum(
            np.where(strikes > s, (strikes - s) * df["put_oi"].values, 0)
        )
        pain.append(call_loss + put_loss)

    return float(strikes[np.argmin(pain)])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(
    pcr: float,
    max_pain: float,
    spot: float,
    iv_pct: float,
    iv_skew: float,
) -> tuple[int, int]:
    """Returns (score_long, score_short) each 0–15."""

    sl = ss = 0

    # PCR (0–10 pts each side)
    if pcr > 1.4:
        sl += 10; ss += 0
    elif pcr > 1.2:
        sl += 7;  ss += 0
    elif pcr > 1.0:
        sl += 3;  ss += 3
    elif pcr > 0.8:
        sl += 0;  ss += 7
    else:
        sl += 0;  ss += 10

    # Max pain vs spot (+3 pts)
    mp_diff = max_pain - spot
    if mp_diff > 50:
        sl += 3
    elif mp_diff < -50:
        ss += 3

    # IV skew (put IV > call IV → bearish hedge → shorts loading → +2 short)
    if iv_skew > 2:
        ss += 2
    elif iv_skew < -2:
        sl += 2

    return min(sl, 15), min(ss, 15)


def _classify_options_bias(pcr: float, max_pain: float, spot: float) -> str:
    if pcr > 1.2 and max_pain > spot:
        return "BULLISH"
    if pcr < 0.8 and max_pain < spot:
        return "BEARISH"
    return "NEUTRAL"


def _fallback_result(symbol: str) -> dict:
    return {
        "symbol":             symbol,
        "expiry":             "N/A",
        "spot_price":         0.0,
        "pcr_oi":             1.0,
        "max_pain":           0.0,
        "max_pain_vs_spot":   0.0,
        "iv_percentile":      50.0,
        "iv_skew":            0.0,
        "top_call_oi_strike": 0.0,
        "top_put_oi_strike":  0.0,
        "institutional_bias": "NEUTRAL",
        "score_long":         0,
        "score_short":        0,
        "chain_df":           pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# CLI quick-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    res = fetch_options_data("NIFTY")
    print("\n=== Options Feed — NIFTY ===")
    for k, v in res.items():
        if k != "chain_df":
            print(f"  {k:<25}: {v}")