"""
backtester.py
-------------
Walk-forward backtester for the Stock Advisor signal engine.

For each window it:
  1. Scores technical techniques on in-sample data.
  2. Fires a signal if total_score >= threshold.
  3. Checks out-of-sample result (next N bars).
  4. Logs trade P&L and builds an equity curve.

NOTE: FII/DII and Options layers cannot be back-tested against historical data
      from NSE (live-only feeds).  The backtester uses the technical layer only
      (max 65 pts) and scales the threshold proportionally.

Usage
-----
    python backtester.py --symbol NIFTY --threshold 55 --hold 5 --windows 50
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from stock_advisor import (
    SYMBOLS, download_ohlcv, score_technical, _atr, _ema
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD = 55   # out of 65 (technical layer only)
DEFAULT_HOLD      = 5    # bars to hold after signal
DEFAULT_WINDOWS   = 60   # number of walk-forward windows
TRAIN_BARS        = 252  # in-sample window (1 year of daily bars)
RISK_PCT          = 0.01 # 1% risk per trade


@dataclass
class Trade:
    date:      str
    direction: str
    entry:     float
    stop:      float
    target:    float
    exit:      float
    bars_held: int
    pnl_pct:   float
    outcome:   str   # "WIN" | "LOSS" | "FLAT"


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def run_backtest(
    symbol:    str   = "NIFTY",
    threshold: int   = DEFAULT_THRESHOLD,
    hold:      int   = DEFAULT_HOLD,
    windows:   int   = DEFAULT_WINDOWS,
) -> dict:

    ticker = SYMBOLS.get(symbol.upper(), symbol)
    df     = download_ohlcv(ticker, period="5y", interval="1d")
    htf_df = download_ohlcv(ticker, period="10y", interval="1wk")

    total_bars = len(df)
    if total_bars < TRAIN_BARS + hold + 10:
        raise ValueError(f"Not enough data: {total_bars} bars, need ≥ {TRAIN_BARS + hold + 10}")

    trades:     list[Trade] = []
    equity_pts: list[float] = [1.0]
    equity      = 1.0

    step = max(1, (total_bars - TRAIN_BARS - hold) // windows)

    for i in range(TRAIN_BARS, total_bars - hold, step):
        train_df     = df.iloc[i - TRAIN_BARS : i].copy()
        train_htf    = htf_df.iloc[: i // 5].copy() if len(htf_df) > i // 5 else htf_df.copy()
        oos_df       = df.iloc[i : i + hold].copy()

        if len(train_df) < 60 or len(oos_df) < 1:
            continue

        tech = score_technical(train_df, train_htf)
        date = str(df.index[i])[:10]

        for dirn, score in [("LONG", tech.long), ("SHORT", tech.short)]:
            if score < threshold:
                continue

            entry = float(train_df["close"].iloc[-1])
            atr14 = float(_atr(train_df["high"], train_df["low"], train_df["close"], 14).iloc[-1])

            if dirn == "LONG":
                stop   = entry - 1.5 * atr14
                target = entry + 3.0 * atr14
            else:
                stop   = entry + 1.5 * atr14
                target = entry - 3.0 * atr14

            # Simulate OOS bars
            exit_price = float(oos_df["close"].iloc[-1])
            bars_held  = len(oos_df)
            outcome    = "FLAT"

            for bar_idx, (_, bar) in enumerate(oos_df.iterrows()):
                if dirn == "LONG":
                    if float(bar["low"]) <= stop:
                        exit_price = stop; bars_held = bar_idx + 1; outcome = "LOSS"; break
                    if float(bar["high"]) >= target:
                        exit_price = target; bars_held = bar_idx + 1; outcome = "WIN"; break
                else:
                    if float(bar["high"]) >= stop:
                        exit_price = stop; bars_held = bar_idx + 1; outcome = "LOSS"; break
                    if float(bar["low"]) <= target:
                        exit_price = target; bars_held = bar_idx + 1; outcome = "WIN"; break

            if outcome == "FLAT":
                outcome = "WIN" if (
                    (dirn == "LONG"  and exit_price > entry) or
                    (dirn == "SHORT" and exit_price < entry)
                ) else "LOSS"

            risk    = abs(entry - stop) / entry
            pnl_pct = (
                (exit_price - entry) / entry if dirn == "LONG"
                else (entry - exit_price) / entry
            )
            # Size position so risk = RISK_PCT of equity
            if risk > 0:
                position_size = RISK_PCT / risk
                trade_return  = pnl_pct * position_size
            else:
                trade_return  = 0.0

            equity *= (1 + trade_return)
            equity_pts.append(equity)

            trades.append(Trade(
                date      = date,
                direction = dirn,
                entry     = entry,
                stop      = stop,
                target    = target,
                exit      = exit_price,
                bars_held = bars_held,
                pnl_pct   = round(pnl_pct * 100, 2),
                outcome   = outcome,
            ))

    # ── Summary stats ────────────────────────────────────────────────────────
    if not trades:
        print("[backtester] No trades fired — try lowering --threshold.")
        return {}

    tdf      = pd.DataFrame([t.__dict__ for t in trades])
    wins     = tdf[tdf["outcome"] == "WIN"]
    losses   = tdf[tdf["outcome"] == "LOSS"]
    win_rate = len(wins) / len(tdf) * 100

    avg_win  = wins["pnl_pct"].mean()   if len(wins)   > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    profit_factor = (
        abs(wins["pnl_pct"].sum() / losses["pnl_pct"].sum())
        if losses["pnl_pct"].sum() != 0 else float("inf")
    )

    # Max drawdown on equity curve
    eq   = pd.Series(equity_pts)
    roll = eq.cummax()
    dd   = (eq - roll) / roll
    max_dd = float(dd.min() * 100)

    print("\n" + "=" * 55)
    print(f"  BACKTEST RESULTS — {symbol.upper()}")
    print("=" * 55)
    print(f"  Total trades     : {len(tdf)}")
    print(f"  Win rate         : {win_rate:.1f}%")
    print(f"  Avg win          : +{avg_win:.2f}%")
    print(f"  Avg loss         : {avg_loss:.2f}%")
    print(f"  Profit factor    : {profit_factor:.2f}")
    print(f"  Final equity     : ×{equity:.3f}")
    print(f"  Max drawdown     : {max_dd:.2f}%")
    print(f"  Threshold used   : {threshold} / 65 pts")
    print("=" * 55)

    # Plot equity curve
    _plot_equity(equity_pts, symbol, tdf)

    return {
        "trades":         tdf,
        "equity":         equity_pts,
        "win_rate":       win_rate,
        "profit_factor":  profit_factor,
        "max_drawdown":   max_dd,
        "final_equity":   equity,
    }


def _plot_equity(equity_pts: list, symbol: str, tdf: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Equity curve
    axes[0].plot(equity_pts, color="steelblue", linewidth=1.5)
    axes[0].set_title(f"{symbol.upper()} — Walk-Forward Equity Curve")
    axes[0].set_ylabel("Equity (×)")
    axes[0].axhline(1.0, linestyle="--", color="grey", linewidth=0.8)
    axes[0].grid(alpha=0.3)

    # Trade distribution
    wins  = tdf[tdf["outcome"] == "WIN"]["pnl_pct"]
    loss  = tdf[tdf["outcome"] == "LOSS"]["pnl_pct"]
    axes[1].hist(wins, bins=20, color="green", alpha=0.6, label=f"Wins ({len(wins)})")
    axes[1].hist(loss, bins=20, color="red",   alpha=0.6, label=f"Losses ({len(loss)})")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Trade PnL Distribution (%)")
    axes[1].set_xlabel("PnL %")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = f"equity_curve_{symbol.lower()}.png"
    plt.savefig(out, dpi=150)
    print(f"\n  Equity curve saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward backtester")
    parser.add_argument("--symbol",    default="NIFTY")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"Technical score threshold 0–65 (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--hold",      type=int, default=DEFAULT_HOLD,
                        help=f"Bars to hold position (default {DEFAULT_HOLD})")
    parser.add_argument("--windows",   type=int, default=DEFAULT_WINDOWS,
                        help=f"Number of walk-forward windows (default {DEFAULT_WINDOWS})")
    args = parser.parse_args()

    run_backtest(
        symbol    = args.symbol,
        threshold = args.threshold,
        hold      = args.hold,
        windows   = args.windows,
    )