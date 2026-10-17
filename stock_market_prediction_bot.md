# Product Requirements Document (PRD): Single-Stock Intraday Prediction Desktop App

**Document Version:** 1.0  
**Status:** Draft / Ready for Engineering  
**Target Market:** NSE India (Equity Cash Segment — Unleveraged Intraday)  
**Infrastructure Cost Target:** ₹0 / month (100% Local Execution & Free Data Tiers)

---

## 1. Executive Summary & Objective
Retail traders operating with small capital (₹5,000 – ₹25,000) in the Indian Equity Cash segment face two primary hurdles: emotional overtrading and fixed infrastructure costs (paid APIs, cloud servers) that erode small-budget returns. 

This project builds a **local, zero-cost desktop decision-support application** focused on predicting short-term intraday price breakouts for a **single high-liquidity NSE stock** (e.g., `SBIN.NS`, `TATAMOTORS.NS`, `BEL.NS`). Instead of noisy regression price-guessing, the system uses a supervised Machine Learning classifier (`XGBoost`) combined with pre-market news sentiment to output actionable probability scores, exact position sizing, and net-of-tax breakeven levels.

---

## 2. Scope & Boundaries

### 2.1 In-Scope (MVP - v1.0)
* **Account-Free Data Pipeline:** Automated fetching of 5-minute and 15-minute OHLCV candles for a target NSE stock and its benchmark sector index via `yfinance` and `nsepython`.
* **Local Persistence:** SQLite database storing historical intraday candles, institutional delivery percentages, and past prediction logs so datasets grow past `yfinance`'s 60-day intraday rolling window.
* **Feature Engineering Pipeline:** Automated calculation of 12+ technical, volatility, and sector-correlation indicators.
* **Probability Classifier Engine:** `XGBoost` binary classification model predicting whether the stock will hit a `+0.75%` target before a `-0.40%` stop-loss within the next 3 to 6 candles (45–90 minutes).
* **Net-of-Tax Risk Calculator:** Real-time calculation of Indian statutory charges (STT, NSE turnover tax, SEBI fees, Stamp Duty, GST, and brokerage) for exact small-budget breakeven validation.
* **Desktop GUI:** Interactive TradingView-style candlestick chart (`lightweight-charts-python` or `Streamlit` local view) with audio/Telegram alerts.

### 2.2 Out-of-Scope (v1.0 — Reserved for v2.0)
* Fully autonomous API order execution (bypasses SEBI Static IP and Algo-ID requirements in v1.0).
* Multi-stock market-wide scanning (50+ tickers simultaneously).
* Options/Derivatives (F&O) Greeks calculation or leveraged MIS margin trading.

---

## 3. System Architecture & Tech Stack

| Component Layer | Technology Choice | Rationale |
| :--- | :--- | :--- |
| **Core Runtime** | Python 3.11+ | Ecosystem dominance for financial ML and data pipelines. |
| **Data Ingestion (Phase 1)** | `yfinance`, `nsepython`, `feedparser` | Zero-cost, no Demat login required for initial development. |
| **Data Ingestion (Phase 2)** | Fyers API v3 / Angel One SmartAPI | Free live WebSocket tick data once Demat account is linked. |
| **Local Storage** | `SQLite3` / `DuckDB` | Serverless, single-file local database stored on user's PC. |
| **Feature Engineering** | `pandas`, `numpy`, `pandas-ta` | Fast vectorized calculation of VWAP, ATR, RSI, and Volume profiles. |
| **Machine Learning** | `xgboost`, `scikit-learn`, `joblib` | Outperforms deep learning (LSTM) on tabular OHLCV data with small sample sizes. |
| **Sentiment Layer** | `Ollama` (Local LLM) or Gemini Flash Free Tier | Parses daily Google News RSS headlines into a `-1.0` to `+1.0` sentiment score. |
| **Desktop Interface** | `lightweight-charts-python` / `Streamlit` | Native TradingView HTML5 canvas rendering with zero frontend overhead. |

---

## 4. Functional Requirements

### 4.1 Module 1: Data Ingestion & Local Storage (`data_engine.py`)
* **FR-1.1 (Historical Backfill):** On initial setup, download the maximum available 5m and 15m historical candles (60 days via `yfinance`) for the primary stock (e.g., `SBIN.NS`), Nifty 50 (`^NSEI`), and the relevant sector index (e.g., `^NSEBANK`).
* **FR-1.2 (Incremental Append):** Every market day at 9:00 AM and 3:35 PM, append newly closed candles into the local `market_data.db` SQLite table (`candles_15m`), preventing duplicates via a composite primary key `(ticker, timestamp)`.
* **FR-1.3 (Bhavcopy Delivery Fetch):** Pull previous-day institutional delivery percentage (`DELIV_PER`) from NSE Bhavcopy via `nsepython`.
* **FR-1.4 (Market Hours Scheduler):** Poll live candle updates strictly between `09:15:00 IST` and `15:30:00 IST`, Monday through Friday (excluding NSE holidays).

### 4.2 Module 2: Feature Engineering Pipeline (`features.py`)
Every closed candle must generate a feature vector $X_t$ containing:
* **FR-2.1 (Trend & VWAP):** Percentage distance of Close price from intraday VWAP:
  $$\text{VWAP\_Dist} = \frac{\text{Close} - \text{VWAP}}{\text{VWAP}} \times 100$$
* **FR-2.2 (Opening Range Context):** Binary flags indicating whether current price is above the 9:15–9:30 AM Opening Range High (`OR_High`) or below Opening Range Low (`OR_Low`).
* **FR-2.3 (Momentum & Volatility):** `RSI(14)`, `ATR(14)` normalized as a percentage of Close price, and `EMA(9)` vs `EMA(21)` spread.
* **FR-2.4 (Relative Volume - RVOL):** Current candle volume divided by the rolling 20-day mean volume for that *exact time-of-day bucket* (avoids 9:15 AM volume skew).
* **FR-2.5 (Sector Relative Strength):** Difference between the stock's 15-minute percentage return and its sector index's 15-minute return:
  $$\text{Rel\_Strength} = R_{\text{Stock}, 15m} - R_{\text{Sector}, 15m}$$
* **FR-2.6 (Time-of-Day Encoding):** Cyclical encoding ($\sin/\cos$) of minutes elapsed since 9:15 AM to capture morning vs. afternoon volatility regimes.

### 4.3 Module 3: Target Labeling & ML Classifier (`ml_engine.py`)
* **FR-3.1 (Triple-Barrier Labeling):** Historical training rows are labeled `1` (Win) ONLY IF the price touches the Upper Profit Barrier ($P_{\text{entry}} \times 1.0075$ or $+1.5 \times \text{ATR}$) *before* touching the Lower Stop-Loss Barrier ($P_{\text{entry}} \times 0.9960$ or $-1.0 \times \text{ATR}$) within the next $N = 4$ candles (60 minutes). Otherwise, label `0`.
* **FR-3.2 (Walk-Forward Validation):** Prohibit random `train_test_split` (which causes data leakage in time-series). Use chronological `TimeSeriesSplit` (e.g., train on Days 1–45, validate on Days 46–60).
* **FR-3.3 (Probability Calibration):** Output calibrated probability via `model.predict_proba(X_live)[:, 1]`.
* **FR-3.4 (Signal Thresholds):**
  * `< 0.65`: **WAIT / NO TRADE** (Grey)
  * `0.65 – 0.74`: **WATCHLIST / MODERATE** (Amber)
  * `>= 0.75`: **HIGH-CONVICTION LONG SIGNAL** (Green)

### 4.4 Module 4: Small-Budget Risk & Indian Tax Calculator (`risk_manager.py`)
Given user capital $C$ (e.g., ₹10,000) and max daily risk $R_{\max} = 1\%$ (₹100):
* **FR-4.1 (Unleveraged Position Sizing):**
  $$\text{Qty} = \min \left( \left\lfloor \frac{C}{\text{Entry Price}} \right\rfloor, \left\lfloor \frac{R_{\max}}{\text{Entry Price} - \text{Stop Loss Price}} \right\rfloor \right)$$
* **FR-4.2 (Exact Statutory Charge Deduction):** Automatically compute round-trip costs for the calculated quantity:
  * **Brokerage:** $\min(0.03\% \times \text{Turnover}, ₹20)$ per leg.
  * **STT (Intraday):** $0.025\%$ on the Sell leg only.
  * **Exchange Transaction Charge (NSE):** $0.00297\%$ on total turnover.
  * **SEBI Turnover Fee:** $0.0001\%$ on total turnover.
  * **Stamp Duty:** $0.003\%$ on the Buy leg only.
  * **GST:** $18\%$ on $(\text{Brokerage} + \text{NSE Charge} + \text{SEBI Fee})$.
* **FR-4.3 (Net Reward Filter):** Suppress any `BUY` signal where estimated Net Profit (Gross Target Profit minus Total Taxes) is less than $1.5 \times \text{Net Risk}$.

### 4.5 Module 5: Desktop UI & Alerting (`app_ui.py`)
* **FR-5.1 (Chart Canvas):** Render live 5m/15m candlesticks with overlaid VWAP line (yellow), Opening Range High/Low dashed lines, and automatic Green/Red arrow markers on signal candles.
* **FR-5.2 (HUD Metrics Panel):** Display current Prediction Confidence (%), Pre-Market News Sentiment Score, Recommended Share Quantity, Entry Price, Stop-Loss, Target, and Net Post-Tax Profit in ₹.
* **FR-5.3 (Instant Notifications):** Trigger a local desktop sound (`winsound` / `plyer`) and an optional Telegram Bot webhook message when probability crosses `0.75`.

---

## 5. Database Schema (`market_data.db`)

```sql
-- Table 1: Historical & Live Intraday Candles
CREATE TABLE IF NOT EXISTS candles_15m (
    ticker TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    vwap REAL,
    PRIMARY KEY (ticker, timestamp)
);

-- Table 2: Model Predictions & Paper-Trade Audit Log
CREATE TABLE IF NOT EXISTS signal_logs (
    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    ticker TEXT NOT NULL,
    close_price REAL NOT NULL,
    pred_probability REAL NOT NULL,
    sentiment_score REAL DEFAULT 0.0,
    recommended_qty INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    target_price REAL NOT NULL,
    est_taxes_inr REAL NOT NULL,
    actual_outcome INTEGER DEFAULT NULL -- Populated after 60 mins (1=Hit Target, 0=Stopped/Expired)
);
```

---

## 6. Project Directory Structure

```text
stock_predictor_app/
├── config.yaml              # Target ticker (e.g., SBIN.NS), capital (10000), risk_pct (0.01)
├── requirements.txt         # yfinance, pandas, pandas-ta, xgboost, scikit-learn, lightweight-charts
├── data/
│   └── market_data.db       # Local SQLite database
├── models/
│   └── sbin_xgb_v1.joblib   # Trained XGBoost model artifact
├── src/
│   ├── data_engine.py       # yfinance / nsepython fetchers & SQLite sync
│   ├── features.py          # Technical indicators & sector correlation math
│   ├── sentiment.py         # Google News RSS parser + LLM scorer
│   ├── ml_engine.py         # Walk-forward training & live probability inference
│   └── risk_manager.py      # Position sizing & Indian brokerage/tax calculator
└── main_desktop.py          # Desktop GUI launcher & 5-minute live polling loop
```

---

## 7. Phased Implementation Roadmap

| Phase | Milestone | Deliverable | Est. Effort |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Data & Feature Pipeline** | Script that builds `market_data.db`, fetches 60 days of 15m data for `SBIN.NS` + `^NSEBANK`, and computes VWAP, RSI, ATR, and RVOL. | Days 1–2 |
| **Phase 2** | **ML Training & Backtest** | `ml_engine.py` with Triple-Barrier labeling, chronological split, and Precision/Win-Rate evaluation report. | Days 3–4 |
| **Phase 3** | **Tax & Risk Engine** | `risk_manager.py` verifying net take-home profit in ₹ after STT, GST, and brokerage on a ₹5k–₹10k budget. | Day 5 |
| **Phase 4** | **Desktop GUI & Alerts** | Native window showing live chart, probability gauge, and Telegram notification integration. | Days 6–7 |
| **Phase 5** | **15-Day Forward Paper Test** | Run the app live during market hours (9:15 AM–3:15 PM) logging virtual trades in `signal_logs` before risking real capital. | Weeks 2–3 |

---

## 8. Success Metrics & Go-Live Gate
Do **not** deploy real capital until the bot achieves the following benchmarks during Phase 5 (15-Day Live Paper Trading):
1. **Precision Score (at $\ge 0.75$ threshold):** Minimum **58% win rate** on triggered signals.
2. **Net Profit Factor:** Gross Net Profits (after deducting simulated STT, brokerage, and GST) divided by Gross Net Losses must be **greater than 1.4**.
3. **Signal Frequency:** Between **1 and 3 signals per week** for a single stock (avoids overtrading and brokerage bleed).
4. **Inference Latency:** Less than **2.0 seconds** from candle close (e.g., 9:30:00 AM) to desktop alert generation (9:30:02 AM).