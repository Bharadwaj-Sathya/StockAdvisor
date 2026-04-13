# Stock Advisor — BUY & SELL Signals via Technical Confluence + FII/DII + Options Flow

A fully rule-based trading dashboard for Nifty 50 and Sensex that fires
high-confidence BUY or SELL signals when all three layers align:
22 technical techniques + FII/DII institutional netflow + Options flow (PCR, OI, max pain).
No ML training required. Signals update in real time.

---

## Project structure

```
stock_advisor/
├── stock_advisor.py      # Core pipeline — indicators + confluence + signal
├── fii_dii_feed.py       # Real-time FII/DII netflow fetcher (NSE/SEBI)
├── options_feed.py       # Options chain parser — PCR, OI, max pain, IV skew
├── dashboard.py          # Full live dashboard (Streamlit) — BUY + SELL + options
├── backtester.py         # Walk-forward backtester + equity curve
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the advisor (terminal signal)

```bash
python stock_advisor.py --direction both
```

Flags:
- `--direction long` — only score BUY setups
- `--direction short` — only score SELL setups
- `--direction both` — score both, print the stronger one

### 3. Run the live dashboard

```bash
streamlit run dashboard.py
```

Toggle LONG / SHORT in the dashboard to switch between BUY and SELL scoring.
All three layers recalculate instantly — institutional flow, options bias, and all 22 techniques.

### 4. Run the backtester

```bash
python backtester.py
```

---

## Architecture

```
OHLCV (yfinance)     FII/DII Netflow (NSE)     Options Chain (NSE F&O)
      │                      │                          │
      ▼                      ▼                          ▼
22 Techniques          Institutional            Options Flow Layer
  (see below)          Flow Layer               ├─ PCR (OI-based)
                    ├─ FII cash equity          ├─ Max pain level
                    ├─ FII F&O (futures)        ├─ Call vs Put OI buildup
                    ├─ DII (MF + insurance)     ├─ IV percentile
                    └─ Sector-wise FII          └─ IV skew (put/call)
      │                      │                          │
      └──────────────┬────────────────────────────────┘
                     ▼
       3-Layer Confluence Scoring (100 pts total)
       ├─ Technical layer      : 65 pts max (22 techniques weighted)
       ├─ Institutional layer  : 20 pts max (FII/DII combined netflow)
       └─ Options layer        : 15 pts max (PCR + OI structure + IV)

       Scored separately for LONG and SHORT.
       The side with higher score is the active signal.
                     │
                     ▼
       Confidence Threshold
       ├─ ≥ 95%  → HIGH-CONFIDENCE BUY or SELL + full trade plan
       ├─ 75–94% → WATCH — setup forming, wait for trigger candle
       ├─ 50–74% → PARTIAL — do not trade, monitor only
       └─ < 50%  → NO SIGNAL — stay flat
                     │
                     ▼
       Trade Plan (BUY or SELL)
       Entry zone · Stop loss · Target 1 · Target 2 · R:R · Options play · Expiry
```

---

## BUY signal conditions (LONG)

| Layer | Bullish condition | Points |
|---|---|---|
| FII/DII | Combined netflow > +₹3,000 Cr | 20 |
| Options | PCR > 1.2 (put writers defending) | 10–15 |
| Options | Max pain above spot (expiry pull-up) | +3 |
| Technical | BOS up + HTF trend up | 15 |
| Technical | Price at key support / S&D zone | 13 |
| Technical | Fibonacci 0.618 retracement hold | 7 |
| Technical | RSI divergence bullish | 6 |
| Technical | Bullish candle (engulfing / pin bar) | 4 |
| Technical | Volume spike at support | 3 |

---

## SELL signal conditions (SHORT)

| Layer | Bearish condition | Points |
|---|---|---|
| FII/DII | Combined netflow < -₹3,000 Cr | 20 |
| Options | PCR < 0.8 (call writers defending resistance) | 10–15 |
| Options | Max pain below spot (expiry pull-down) | +3 |
| Technical | BOS down + HTF trend down | 15 |
| Technical | Price at key resistance / supply zone | 13 |
| Technical | Fibonacci 0.618 retracement rejection | 7 |
| Technical | RSI divergence bearish | 6 |
| Technical | Bearish candle (engulfing / pin bar) | 4 |
| Technical | Volume spike at resistance | 3 |

---

## Options flow signals

| Metric | Bullish reading | Bearish reading |
|---|---|---|
| PCR (OI) | > 1.2 — heavy put writing, floor in place | < 0.8 — heavy call writing, ceiling in place |
| Max pain | Above spot — price likely to rise into expiry | Below spot — price likely to fall into expiry |
| IV percentile | < 30% — buy options (cheap premium) | > 70% — sell options (expensive premium) |
| IV skew | Put IV > Call IV — institutions hedging downside | Call IV > Put IV — fear of rally, shorts loading |
| OI buildup | Large put OI at round number = strong support | Large call OI at round number = strong resistance |

---

## The 22 techniques (with LONG and SHORT interpretation)

| Technique | BUY signal | SELL signal |
|---|---|---|
| BOS / CHOCH | Bullish BOS, demand structure | Bearish BOS, supply structure |
| Higher timeframe trend | HTF uptrend aligned | HTF downtrend aligned |
| Key S/R | Price bouncing off support | Price rejecting resistance |
| Dynamic S/R (EMA 20/50) | Price above EMA, bounce | Price below EMA, rejection |
| Trendlines | Bouncing off ascending trendline | Breaking below ascending trendline |
| Gann angles | Price above 1x1 Gann angle | Price below 1x1 Gann angle |
| Fibonacci retracements | 0.618–0.786 retracement hold | 0.618–0.786 retracement rejection |
| Harmonic patterns | Bullish Gartley / Bat / Crab at PRZ | Bearish Gartley / Bat / Crab at PRZ |
| Elliott Wave | Wave 3 or Wave C bounce | Wave 3 down or Wave C drop |
| Supply & demand zones | Price entering demand zone | Price entering supply zone |
| Fair value gap (FVG) | Price filling bullish FVG | Price filling bearish FVG |
| Breakouts | Breakout above range with volume | Breakdown below range with volume |
| Momentum indicators | Momentum turning up | Momentum turning down |
| Oscillators | RSI < 35, stoch oversold | RSI > 65, stoch overbought |
| Divergence | Bullish divergence confirmed | Bearish divergence confirmed |
| Reversal signals | Reversal at support | Reversal at resistance |
| Candlesticks | Engulfing / pin bar / hammer | Engulfing / shooting star / doji |
| Heikin Ashi | Green HA candles, no lower shadows | Red HA candles, no upper shadows |
| Renko | Renko box flipped up | Renko box flipped down |
| Volume | Spike on bounce = buying pressure | Spike on rejection = selling pressure |
| OBV trend | OBV rising with price | OBV falling with price |
| India VIX | VIX falling — risk-on | VIX rising — risk-off, avoid longs |

---

## Trade plan output format

```
Signal   : HIGH-CONFIDENCE SELL  (92% confluence)
Direction: SHORT — Nifty 50 (Weekly / Daily)

Entry    : 22,600 – 22,650 (supply zone / call OI wall)
Stop loss: 22,750 (above supply zone)
Target 1 : 22,300 (put OI support / Fib 0.5 level)
Target 2 : 22,100 (Fib 1.272 extension / demand zone)
R:R      : 1:3
Risk     : 0.5 – 1% of capital

Options  : Bear put spread — Buy 22,500 PE, Sell 22,200 PE (weekly expiry)
           OR: Buy 22,500 PE outright if IV percentile < 30%

Trail SL : Move to breakeven after Target 1 hit
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| BUY + SELL both scored | Markets move both ways — a sell signal is as valuable as a buy |
| Options layer added | PCR and max pain reveal where smart money is positioned |
| Options play in trade plan | Gives a defined-risk alternative to pure futures trading |
| FII as institutional tide | Retail setups against FII selling historically underperform |
| 95% threshold unchanged | Quality over quantity — fewer but higher-conviction trades |
| IV filter for options entry | Never buy expensive options; never short cheap ones naked |
| VIX filter | VIX > 18 means avoid naked longs; use spreads instead |

---

## Data sources

| Data | Source | Refresh |
|---|---|---|
| OHLCV price data | yfinance (NSE) | Intraday / daily |
| FII/DII cash segment | NSE India portal | End of day |
| FII/DII F&O positions | NSE participant OI data | End of day |
| Options chain (PCR, OI) | NSE F&O bhavcopy | Intraday (every 30s) |
| India VIX | NSE (^INDIAVIX via yfinance) | Real-time |
| Sector FII flow | NSE sector index data | Intraday estimate |

---

## Disclaimer

This project is for **educational purposes only**.
Past performance does not guarantee future results.
A 95% confluence score means a technically strong setup — not a 95% win rate.
Options trading carries unlimited loss risk on naked positions — always use spreads.
Do not use this as the sole basis for real trading decisions.
