"""
fii_dii_feed.py
---------------
Fetches FII / DII cash-equity and F&O netflow data from NSE India.
Returns a structured dict used by the confluence scorer.

Data sources
  - NSE FII/DII activity:
    https://www.nseindia.com/api/fiidiiTradeReact

Requirements
  pip install curl_cffi pandas
  (curl_cffi spoofs TLS fingerprints to bypass NSE/Akamai bot detection)
"""

from __future__ import annotations

import time
import pandas as pd
from datetime import datetime

try:
    from curl_cffi import requests as curl_requests
    _USE_CURL_CFFI = True
except ImportError:
    import requests as curl_requests          # type: ignore[no-redef]
    _USE_CURL_CFFI = False
    print(
        "[fii_dii_feed] WARNING: curl_cffi not installed. "
        "Falling back to requests — NSE may block the connection.\n"
        "  Install with:  pip install curl_cffi"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

NSE_HOME_URL   = "https://www.nseindia.com"
NSE_MARKET_URL = "https://www.nseindia.com/market-data/live-equity-market"
FIIDII_URL     = "https://www.nseindia.com/api/fiidiiTradeReact"


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------

def _nse_session():
    """
    Return a warmed-up session that NSE's Akamai bot-detection will accept.

    With curl_cffi  → impersonates Chrome 124 at the TLS layer (most reliable).
    Without it      → plain requests with NSE headers (may still be blocked).
    """
    if _USE_CURL_CFFI:
        session = curl_requests.Session(impersonate="chrome124")
    else:
        session = curl_requests.Session()

    session.headers.update(NSE_HEADERS)

    # Warm-up step 1 – main page (sets initial cookies)
    try:
        r = session.get(NSE_HOME_URL, timeout=12)
        print(f"[fii_dii_feed] Warm-up 1 → {r.status_code}")
        time.sleep(2)
    except Exception as e:
        print(f"[fii_dii_feed] Warm-up 1 failed: {e}")

    # Warm-up step 2 – market-data page (closer to the API route)
    try:
        r = session.get(NSE_MARKET_URL, timeout=12)
        print(f"[fii_dii_feed] Warm-up 2 → {r.status_code}")
        time.sleep(1)
    except Exception as e:
        print(f"[fii_dii_feed] Warm-up 2 failed: {e}")

    return session


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_fiidii_data(days: int = 5) -> dict:
    """
    Fetch FII / DII cash-equity netflow for the last *days* trading sessions.

    Returns
    -------
    dict with keys:
        latest_date          : str   (YYYY-MM-DD)
        fii_cash_net_cr      : float (₹ Crore, negative = net selling)
        dii_cash_net_cr      : float
        combined_net_cr      : float
        fii_5d_avg_cr        : float (rolling average over last *days* sessions)
        dii_5d_avg_cr        : float
        institutional_bias   : str   ("BULLISH" | "BEARISH" | "NEUTRAL")
        score_long           : int   (0–20)
        score_short          : int   (0–20)
        raw_df               : pd.DataFrame
    """
    session = _nse_session()

    # ── Fetch ────────────────────────────────────────────────────────────────
    try:
        resp = session.get(FIIDII_URL, timeout=20)

        # Debug info
        print(f"[fii_dii_feed] API status  : {resp.status_code}")
        print(f"[fii_dii_feed] Content-Type: {resp.headers.get('Content-Type', 'n/a')}")
        preview = resp.text[:300].strip() if resp.text else "<empty>"
        print(f"[fii_dii_feed] Response    : {preview}")

        if not resp.text.strip():
            print("[fii_dii_feed] Empty response body — bot-detection likely triggered.")
            return _fallback_result()

        resp.raise_for_status()
        data = resp.json()

    except Exception as exc:
        print(f"[fii_dii_feed] Fetch error: {exc}")
        return _fallback_result()

    # ── Parse ────────────────────────────────────────────────────────────────
    try:
        df = _parse_fiidii_json(data)
    except Exception as exc:
        print(f"[fii_dii_feed] Parse error: {exc}")
        return _fallback_result()

    if df.empty:
        print("[fii_dii_feed] Parsed DataFrame is empty.")
        return _fallback_result()

    # ── Compute ──────────────────────────────────────────────────────────────
    df = df.tail(days).copy()

    latest      = df.iloc[-1]
    fii_net     = float(latest.get("fii_net", 0))
    dii_net     = float(latest.get("dii_net", 0))
    combined    = fii_net + dii_net

    fii_avg     = float(df["fii_net"].mean())
    dii_avg     = float(df["dii_net"].mean())

    bias                  = _classify_bias(combined, fii_avg)
    score_long, score_short = _score(combined, fii_avg)

    return {
        "latest_date"       : str(latest.get("date", datetime.today().date())),
        "fii_cash_net_cr"   : fii_net,
        "dii_cash_net_cr"   : dii_net,
        "combined_net_cr"   : combined,
        "fii_5d_avg_cr"     : fii_avg,
        "dii_5d_avg_cr"     : dii_avg,
        "institutional_bias": bias,
        "score_long"        : score_long,
        "score_short"       : score_short,
        "raw_df"            : df,
    }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_fiidii_json(data: list | dict) -> pd.DataFrame:
    """
    Normalises NSE JSON where FII and DII appear as separate row entries
    and pivots them into one row per date.

    Handles both shapes NSE has been observed to return:
      • A bare list  → [ {category, date, netValue, …}, … ]
      • A dict       → { "data": [ … ] }  or  { "category": [ … ] }
    """
    # 1. Unwrap dict wrapper if present
    if isinstance(data, dict):
        data = data.get("data", data.get("category", list(data.values())[0] if data else []))

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=["date", "fii_net", "dii_net"])

    # 2. Build raw DataFrame
    raw = pd.DataFrame(data)
    print(f"[fii_dii_feed] Raw columns: {list(raw.columns)}")
    print(f"[fii_dii_feed] Raw shape  : {raw.shape}")

    # 3. Date column — handle casing variants
    date_col = next((c for c in raw.columns if c.lower() == "date"), None)
    if date_col is None:
        raise ValueError("No date column found in NSE response")
    raw["date"] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")

    # 4. Category column — normalise to uppercase, strip whitespace
    cat_col = next((c for c in raw.columns if c.lower() == "category"), None)
    if cat_col is None:
        raise ValueError("No category column found in NSE response")
    raw["category"] = raw[cat_col].str.strip().str.upper()

    # 5. Net-value column — try several known names
    for candidate in ("netValue", "net_value", "NetValue", "NET_VALUE", "netvalue"):
        if candidate in raw.columns:
            val_col = candidate
            break
    else:
        raise ValueError(f"No netValue column found. Available: {list(raw.columns)}")

    raw["net_float"] = pd.to_numeric(raw[val_col], errors="coerce").fillna(0.0)

    # 6. Pivot: one row per date, columns = category names
    pivoted = (
        raw.pivot_table(index="date", columns="category", values="net_float", aggfunc="sum")
        .reset_index()
    )
    print(f"[fii_dii_feed] Pivoted columns: {list(pivoted.columns)}")

    # 7. Robust column mapping  (FII / FII/FPI / FPI  →  fii_net)
    result = {"date": pivoted["date"]}

    fii_src = [c for c in pivoted.columns if "FII" in str(c).upper() or "FPI" in str(c).upper()]
    result["fii_net"] = pivoted[fii_src[0]] if fii_src else 0.0

    dii_src = [c for c in pivoted.columns if "DII" in str(c).upper()]
    result["dii_net"] = pivoted[dii_src[0]] if dii_src else 0.0

    df_final = pd.DataFrame(result)
    return df_final.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scoring / classification
# ---------------------------------------------------------------------------

def _classify_bias(combined: float, fii_avg: float) -> str:
    if combined > 3000 or fii_avg > 1000:
        return "BULLISH"
    if combined < -3000 or fii_avg < -1000:
        return "BEARISH"
    return "NEUTRAL"


def _score(combined: float, fii_avg: float) -> tuple[int, int]:
    """
    Returns (score_long, score_short) each in range 0–20.

    Full 20 pts when combined > +3 000 Cr (long) or < -3 000 Cr (short).
    Partial credit in between.
    """
    # Long score
    if combined >= 3000:
        sl = 20
    elif combined >= 1500:
        sl = 12
    elif combined >= 0:
        sl = 5
    else:
        sl = 0

    # Short score
    if combined <= -3000:
        ss = 20
    elif combined <= -1500:
        ss = 12
    elif combined <= 0:
        ss = 5
    else:
        ss = 0

    return sl, ss


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fallback_result() -> dict:
    """Neutral zeroed-out result returned when the live feed is unavailable."""
    return {
        "latest_date"       : str(datetime.today().date()),
        "fii_cash_net_cr"   : 0.0,
        "dii_cash_net_cr"   : 0.0,
        "combined_net_cr"   : 0.0,
        "fii_5d_avg_cr"     : 0.0,
        "dii_5d_avg_cr"     : 0.0,
        "institutional_bias": "NEUTRAL",
        "score_long"        : 0,
        "score_short"       : 0,
        "raw_df"            : pd.DataFrame(columns=["date", "fii_net", "dii_net"]),
    }


# ---------------------------------------------------------------------------
# CLI quick-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = fetch_fiidii_data()
    print("\n=== FII / DII Feed ===")
    for k, v in result.items():
        if k != "raw_df":
            print(f"  {k:<25}: {v}")
    print("\nLast 5 days raw data:")
    print(result["raw_df"].to_string(index=False))