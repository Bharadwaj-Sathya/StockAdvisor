# 📈 AI Option Trading Signal Model
### High-Probability Nifty Options Signal Engine

> **Goal:** Build an AI-powered system that analyzes real-time market data and generates high-confidence option trading signals for Nifty 50, specifically for CE/PE buying strategies.

---

## 🧠 Project Philosophy

| Principle | Description |
|---|---|
| **Data-Driven** | Every signal backed by multi-indicator confluence |
| **Risk-First** | No trade without defined risk parameters |
| **High Probability** | Target 80%+ win rate through strict filters |
| **Transparent** | Every signal comes with reasoning & confidence score |

> ⚠️ **Important:** No model achieves 100% accuracy in live markets. This system aims for the **highest probability** setups by combining multiple confirmation layers before generating a signal.

---

## 📁 Project Structure

```
ai-option-signal-model/
│
├── README.md                        # This file
├── .env                             # API keys (never commit)
├── requirements.txt                 # Python dependencies
├── main.py                          # Entry point — starts all async loops
│
├── data/                            # Data ingestion layer
│   ├── ws_kite.py                   # Zerodha Kite WebSocket client
│   ├── ws_upstox.py                 # Upstox WebSocket client (fallback)
│   ├── poller_options_chain.py      # NSE options chain — HTTP poll 5s
│   ├── poller_vix.py                # India VIX — HTTP poll 30s
│   ├── fetcher_fii.py               # FII/DII EOD downloader
│   ├── fetcher_historical.py        # Kite historical OHLCV sync
│   └── event_bus.py                 # Async event bus (asyncio)
│
├── store/                           # In-memory state layer
│   ├── tick_store.py                # Latest tick per instrument (Redis/dict)
│   ├── oi_buffer.py                 # Rolling 20-snapshot OI buffer
│   └── candle_builder.py            # Live 1m / 5m / 15m candle builder
│
├── indicators/                      # Technical indicator engine
│   ├── pcr_calculator.py            # PCR — fires every 5s on OI update
│   ├── oi_analysis.py               # OI buildup & unwinding delta
│   ├── vix_filter.py                # VIX gate — fires every 30s
│   ├── ema_engine.py                # EMA 9/21/50 — fires on 1m close
│   ├── supertrend.py                # Supertrend — fires on 1m close
│   ├── support_resistance.py        # S/R levels — fires on 5m close
│   └── trend_engine.py              # Trend & momentum aggregator
│
├── ai_model/                        # Core AI/ML engine
│   ├── signal_generator.py          # Main signal generation
│   ├── confidence_scorer.py         # Signal confidence (0-100%)
│   ├── pattern_recognizer.py        # Candlestick & chart patterns
│   ├── ml_model.py                  # Trained ML classifier
│   └── model_trainer.py             # Model training pipeline
│
├── risk/                            # Risk management layer
│   ├── position_sizer.py            # Lot size calculator
│   ├── stop_loss_engine.py          # Dynamic SL calculator
│   └── reward_ratio.py              # Risk:Reward validator
│
├── alerts/                          # Signal delivery system
│   ├── telegram_bot.py              # Telegram alert sender
│   ├── dashboard.py                 # Web dashboard (Streamlit)
│   └── signal_logger.py             # Trade log & journal
│
├── backtester/                      # Strategy backtesting
│   ├── backtest_engine.py           # Historical signal tester
│   ├── performance_report.py        # Win rate, P&L analysis
│   └── historical_data/             # Stored historical data
│
└── config/
    ├── settings.py                  # Global config
    └── strategy_params.py           # Strategy parameters
```

---

## 🔄 System Architecture & Signal Flow

```
ZERODHA KITE WEBSOCKET (~50ms)
UPSTOX WEBSOCKET (~50ms)          ──► ASYNC EVENT BUS (asyncio)
NSE OPTIONS CHAIN POLL (5s)       ──►   Normalise | Deduplicate | Align
NSE VIX REST POLL (30s)           ──►
      │
      ▼
┌──────────────────────────────────────────┐
│        IN-MEMORY STATE (Redis)           │
│  Latest tick | Rolling OI buffer         │
│  Live candle builder (1m / 5m / 15m)    │
└──────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────┐
│         INDICATOR ENGINE                 │
│  PCR (5s) | EMA/Supertrend (1m close)   │
│  VIX gate (30s) | S/R levels (5m close) │
└──────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────┐
│         AI SIGNAL ENGINE                 │
│  XGBoost + LSTM ensemble inference       │
│  Confidence Scorer (0–100%)              │
└──────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────┐
│       RISK FILTER (MANDATORY)            │
│  VIX Check | SL Defined | R:R ≥ 1:2     │
└──────────────────────────────────────────┘
      │
      ▼  (Only if ALL filters pass)
┌──────────────────────────────────────────┐
│           SIGNAL OUTPUT                  │
│  Strike | Direction | Entry | SL | TGT  │
└──────────────────────────────────────────┘
      │
      ▼
  📱 Telegram Alert (<1s)  +  📊 Streamlit Dashboard

  ⚡ Tick → Signal latency: < 100ms
```

---

## 📊 Signal Parameters (Example)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🟢 SIGNAL GENERATED — CE BUY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Underlying   : NIFTY 50
Spot Price   : 24,000
Strike       : 24,200 CE
Premium      : ₹85
Entry Range  : ₹82 – ₹88
Target 1     : ₹130 (+₹45)
Target 2     : ₹160 (+₹75)
Stop Loss    : ₹55 (–₹30)
Risk:Reward  : 1 : 2.5
Confidence   : 78%
VIX          : 20.1 (✅ Safe)
PCR          : 1.18 (✅ Bullish)
Time Window  : 10:15 – 11:30 AM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📡 Data Sources & Real-Time Architecture

> ⚡ **Why 3 minutes is too slow:** Options premiums can move 30–50% in under 60 seconds during a breakout. By the time a 3-min poll returns, the signal is already gone or the trade is already against you. Every layer must be as close to real-time as possible.

### Connection Types

| Data Type | Source | Method | Latency |
|---|---|---|---|
| Live Nifty tick price | **Zerodha Kite Connect** | WebSocket (persistent) | ~50ms |
| Live Options OI per strike | **Zerodha Kite / Upstox v2** | WebSocket (subscribed strikes) | ~50ms |
| Full Options Chain snapshot | NSE Options Chain REST | HTTP poll every **5 seconds** | 5s |
| India VIX | NSE REST API | HTTP poll every **30 seconds** | 30s |
| FII/DII Activity | NSE Bhav Copy | Daily EOD download | 1 day |
| Historical OHLCV | Zerodha Kite Historical API | On-demand / daily sync | — |

### Why WebSocket Over Polling

```
Polling (OLD — 3 min):        WebSocket (NEW — real-time):
  You → NSE every 3 min         Server pushes to you on every tick
  Miss breakout moves ❌         Receive data in ~50ms ✅
  Stale OI data ❌               Live OI updates ✅
  High API rate-limit risk ❌    Single persistent connection ✅
```

### Real-Time Data Flow

```
Zerodha Kite WebSocket ──────────────────────────────┐
Upstox WebSocket ─────────────────────────────────── │
                                                      ▼
NSE Options Chain (5s poll) ──────► Async Event Bus (asyncio + aiohttp)
NSE VIX REST (30s poll) ──────────►   Normalise | Deduplicate | Align
                                                      │
                                                      ▼
                                      In-Memory State (Redis / Python dict)
                                      ┌───────────────────────────────────┐
                                      │  Latest tick per instrument        │
                                      │  Rolling 20-snapshot OI buffer     │
                                      │  Live candle builder (1m/5m/15m)  │
                                      └───────────────────────────────────┘
                                                      │
                                                      ▼
                                         Indicator Engine (real-time)
                                      ┌───────────────────────────────────┐
                                      │  PCR + OI delta  → every 5s       │
                                      │  EMA / Supertrend → each 1m close  │
                                      │  VIX gate         → every 30s      │
                                      │  S/R levels       → each 5m close  │
                                      └───────────────────────────────────┘
                                                      │
                                                      ▼
                                         AI Signal Engine (<50ms inference)
                                                      │
                                                      ▼
                                    Tick → Signal end-to-end: < 100ms
```

### Broker API Recommendation

**Zerodha Kite Connect** is the primary choice:
- Full WebSocket tick feed for any instrument ✅
- Options chain + Greeks access ✅
- Historical OHLCV API (intraday + daily) ✅
- Mature Python SDK: `kiteconnect` ✅
- Cost: ₹2,000/month for API access

**Upstox API v2** as a free fallback:
- WebSocket available ✅
- Good for options Greeks data ✅
- Free tier available ✅

### Indicator Compute Triggers

| Indicator | Trigger | Frequency |
|---|---|---|
| PCR + OI delta | Every OI snapshot | Every 5s |
| EMA (9, 21, 50) | Each 1-min candle close | Every 1 min |
| Supertrend | Each 1-min candle close | Every 1 min |
| VIX gate check | Every VIX refresh | Every 30s |
| Support / Resistance | Each 5-min candle close | Every 5 min |
| AI model inference | Each 1-min candle close | Every 1 min |

---

## 🤖 AI Model — What It Learns

### Training Features (Inputs)
```
Market Features:
  - Nifty spot price & % change
  - Distance from key S/R levels
  - 9 EMA, 21 EMA, 50 EMA values
  - RSI (14), MACD, Supertrend

Options Features:
  - ATM, OTM-1, OTM-2 IV
  - PCR (OI based)
  - OI change % in last 30 min
  - Max Pain level

Macro Features:
  - India VIX
  - FII net buy/sell (prev day)
  - SGX Nifty gap
  - Time of day (session factor)
```

### Model Output (Labels)
```
0 → No Trade (wait)
1 → CE Buy Signal
2 → PE Buy Signal
```

### Model Type
- **Primary:** XGBoost Classifier (fast, accurate for tabular data)
- **Secondary:** LSTM Neural Network (for time-series pattern learning)
- **Ensemble:** Both models vote → signal only if both agree

---

## ✅ Signal Generation Rules (Confluence Filters)

A signal is generated **ONLY** when ALL of these conditions pass:

### Filter 1 — VIX Gate
```
VIX < 15       → ✅ Excellent (options cheap, clean moves)
VIX 15–20      → ✅ Good (trade with normal SL)
VIX 20–24      → ⚠️  Caution (widen SL, reduce quantity)
VIX > 24       → ❌ No Trade (too expensive, unpredictable)
```

### Filter 2 — Trend Confirmation
```
For CE Buy:
  - Nifty above 21 EMA on 15-min chart
  - Higher highs forming in last 3 candles
  - Supertrend on BUY mode

For PE Buy:
  - Nifty below 21 EMA on 15-min chart
  - Lower lows forming in last 3 candles
  - Supertrend on SELL mode
```

### Filter 3 — OI & PCR Check
```
For CE Buy:
  - PCR > 1.0 (more puts = bullish floor support)
  - CE OI at target strike not very high (resistance check)
  - PE writing happening at lower strikes (support building)

For PE Buy:
  - PCR < 0.8 (more calls = bearish ceiling pressure)
  - PE OI at target strike not very high
  - CE writing happening at higher strikes
```

### Filter 4 — Time Filter
```
✅ Best:  10:00 AM – 11:30 AM  (direction established)
✅ Good:  1:30 PM  – 2:30 PM   (post-lunch momentum)
❌ Avoid: 9:15 AM – 9:45 AM    (volatile open)
❌ Avoid: 2:45 PM – 3:30 PM    (expiry manipulation)
```

### Filter 5 — Event Filter
```
❌ No trade on:
  - RBI Policy day
  - Budget day
  - US Fed announcement day
  - Monthly/Weekly expiry day (unless expert mode)
```

---

## 📉 Risk Management Module

```
MANDATORY RULES — Cannot be overridden by AI:

1. Stop Loss    = Always defined BEFORE entry
2. Max Loss/Day = 2% of total capital
3. Max Trades   = 3 per day
4. R:R Minimum  = 1:1.5 (ideally 1:2.5)
5. Trailing SL  = Activated when Target 1 is hit

POSITION SIZING:
  Capital     = ₹1,00,000 (example)
  Risk/Trade  = 1% = ₹1,000
  SL Distance = ₹30 per unit
  Lot Size    = ₹1,000 / ₹30 = 33 units (~1 lot Nifty)
```

---

## 🧪 Backtesting Framework

```
Test Period   : Jan 2022 – Dec 2024 (3 years)
Data          : 5-min & 15-min OHLCV + OI data

Metrics Tracked:
  - Win Rate %
  - Average P&L per trade
  - Max Drawdown
  - Sharpe Ratio
  - Profit Factor
  - Average holding time

Target Benchmark:
  Win Rate    ≥ 65%
  Avg P&L     ≥ ₹500/trade
  Max DD      ≤ 15%
  Sharpe      ≥ 1.5
```

---

## 🚀 Development Phases

### Phase 1 — Real-Time Data Foundation (Week 1–2)
- [ ] Set up Zerodha Kite WebSocket client (`ws_kite.py`)
- [ ] Set up Upstox WebSocket as fallback (`ws_upstox.py`)
- [ ] Build async event bus to normalise all feeds
- [ ] NSE Options Chain HTTP poller — every **5 seconds**
- [ ] India VIX HTTP poller — every **30 seconds**
- [ ] Redis / in-memory tick store with rolling OI buffer
- [ ] Live candle builder (1m, 5m, 15m) from tick stream
- [ ] Verify end-to-end tick → store latency < 100ms

### Phase 2 — Indicator Engine (Week 2–3)
- [ ] Build all technical indicators
- [ ] Support & resistance level detection
- [ ] Trend identification module
- [ ] All confluence filters coded

### Phase 3 — AI Model (Week 3–5)
- [ ] Prepare training dataset (label past signals)
- [ ] Train XGBoost classifier
- [ ] Train LSTM time-series model
- [ ] Ensemble logic: both must agree
- [ ] Confidence scoring system

### Phase 4 — Backtesting (Week 5–6)
- [ ] Run backtest on 3 years of data
- [ ] Optimize parameters
- [ ] Validate win rate target ≥ 65%
- [ ] Fix false signal leakages

### Phase 5 — Live Alerts (Week 6–7)
- [ ] Telegram bot for real-time signals
- [ ] Streamlit dashboard (live view)
- [ ] Signal log & trade journal
- [ ] Paper trading (2 weeks before going live)

### Phase 6 — Live Deployment (Week 8+)
- [ ] Paper trade validation complete
- [ ] Deploy on cloud (AWS / Heroku)
- [ ] Monitor signal quality
- [ ] Monthly model retraining

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Real-Time Feed | Zerodha Kite WebSocket / Upstox WebSocket |
| Async Engine | `asyncio` + `aiohttp` |
| In-Memory Store | Redis (prod) / Python dict (dev) |
| Data Processing | Pandas, NumPy |
| ML Model | XGBoost, Scikit-learn, TensorFlow/Keras |
| Technical Analysis | TA-Lib, Pandas-TA |
| Database | PostgreSQL (historical + signal log) |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Alerts | Python-Telegram-Bot |
| Scheduling | APScheduler |
| Deployment | AWS EC2 / Railway.app |

---

## 🔑 API Keys Required

```
# .env file

# Zerodha Kite Connect (primary — WebSocket + historical data)
ZERODHA_API_KEY=your_kite_api_key
ZERODHA_API_SECRET=your_kite_secret
ZERODHA_ACCESS_TOKEN=generated_at_login   # Refresh daily

# Upstox API v2 (fallback WebSocket)
UPSTOX_API_KEY=your_upstox_key
UPSTOX_API_SECRET=your_upstox_secret
UPSTOX_ACCESS_TOKEN=generated_at_login

# Redis (in-memory tick store)
REDIS_HOST=localhost
REDIS_PORT=6379

# Telegram alerts
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## ⚠️ Honest Disclaimer

```
📌 This model aims for HIGHEST PROBABILITY — not 100% accuracy.
📌 Markets are unpredictable — always use Stop Loss.
📌 Past performance does not guarantee future results.
📌 Never trade money you cannot afford to lose.
📌 This is a decision-support tool, not a guaranteed profit machine.
```

---

## 📌 Current Market Context

```
Nifty Spot     : ~24,000
Target Strike  : 24,200 CE (OTM by 200 pts)
VIX Status     : 20 (Falling from 24 — Positive sign ✅)
VIX Trend      : Bearish on VIX = Bullish for market
Implication    : Premium costs reducing, good time to plan CE buys
                 Wait for VIX to stabilize below 18 for cleaner trades
```

---

## 👨‍💻 Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/ai-option-signal-model.git
cd ai-option-signal-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Redis (required for in-memory tick store)
redis-server

# Setup environment variables
cp .env.example .env
# Edit .env with your Kite API key, Telegram token, Redis config

# Authenticate with Zerodha Kite (run once daily — generates access token)
python scripts/kite_auth.py

# Run the signal engine (starts WebSocket + pollers + AI engine)
python main.py
```

---

*Built for Nifty 50 Options | Intraday Focus | India Markets*
 