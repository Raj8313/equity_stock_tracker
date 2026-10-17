# Project Memory & Context: Stock Prediction App

**Last Updated:** 2026-10-17  
**Location:** `/Users/raja/my_projects/stock_prediction_app/MEMORY.md` (Local project-level memory)

---

## 1. Operating Rules & User Preferences
* **Memory Storage:** Store and update project memory in this workspace directory (`MEMORY.md`), never in global config. Update periodically when key decisions or architectural choices are made.
* **Mentoring Mode & Role:** User is a beginner with no prior experience building quantitative trading systems or ML pipelines. Agent acts as a Senior Quant/Mentor: explain concepts in plain English, discuss and align on the "why" and mechanics first before writing any code.
* **No Unsolicited Code Changes:** Strictly discuss and plan; only write or modify application code when explicitly instructed.
* **Target Environment:** macOS, Python 3.11+, local execution (₹0/month recurring cost).

---

## 2. Project Overview & Core Objectives
* **Product:** Single-Stock Intraday Prediction Desktop App (PRD v1.0 in `stock_market_prediction_bot.md`).
* **Market:** NSE India (Equity Cash Segment, unleveraged intraday, strictly Long-only in v1.0).
* **Segment Definition:** Equity Cash Intraday (MIS/same-day square-off; buying actual shares with 0 time decay, squared off before 15:15 IST).
* **Target Universe:** **Strictly NIFTY 50 constituents ONLY** (Large Cap). Explicitly prohibited: Mid Cap and Small Cap stocks (due to illiquidity, circuit limit risks, and wide bid-ask spreads).
* **Preferred Ticker Selection:**
  - **Active Selection for 5-Day Test:** `TMPV.NS` (Tata Motors Passenger Vehicles Limited — post-demerger active NSE ticker for Tata Motors passenger car/EV division, ~₹285, 1,497 candles backfilled).
  - **Alternate Target:** `SBIN.NS` (State Bank of India — banking sector benchmark, ~₹800).
  - **Deprioritized for Small Capital:** `RELIANCE.NS` (~₹2,800, too expensive for ₹5k-10k sizing) and `HDFCBANK.NS` (~₹1,650, slower percentage moves).
* **Validation Strategy:** 5-Day Live Observation / Forward Paper Test (no real capital risked; system monitors live market, news, predicts probabilities, and logs virtual trade outcomes).
* **Target User Capital:** ₹5,000 – ₹25,000.
* **Core Philosophy:** Classification over regression. Focus on probability of hitting an asymmetric target before a stop-loss under strict statutory transaction fee constraints.
* **Current Implementation Status (Completed & Verified):**
  - LEGO Block 1: `config.yaml` — Central trading, tax, and model parameters.
  - LEGO Block 2: `src/data_engine.py` — SQLite ingestion, 1,497 15m candles backfilled into `market_data.db`.
  - LEGO Block 3: `src/features.py` — Vectorized VWAP, ORB, normalized ATR, RSI, time-bucketed RVOL, sector relative strength.
  - LEGO Block 4: `src/sentiment.py` — Google News RSS feedparser + financial sentiment lexicon + veto filter.
  - LEGO Block 5: `src/risk_manager.py` — Exact Indian statutory taxes (STT, GST, SEBI fee, NSE turnover, stamp duty, brokerage) + integer position sizing.
  - LEGO Block 6: `src/ml_engine.py` — Triple-barrier labeling, chronological TimeSeriesSplit, calibrated gradient boosting.
  - LEGO Block 7: `main_desktop.py` — Native desktop station (`lightweight-charts-python`) with TradingView canvas, VWAP line, decision HUD table, and macOS audio alerts.
  - Test Harness: `test_pipeline.py` — Verified all 7 blocks end-to-end with 100% test pass rate.

---

## 3. Architecture & Tech Stack Summary
* **Data Ingestion:**
  * Phase 1: `yfinance`, `nsepython`, `feedparser` (Google News RSS).
  * Phase 2: Broker APIs (Fyers v3 / Angel One SmartAPI / Dhan) for live zero-latency tick data via Demat.
* **Database:** Local `SQLite3` (`market_data.db`) storing `candles_15m` (composite PK `(ticker, timestamp)`) and `signal_logs`.
* **Feature Engineering (`pandas`, `pandas-ta`):**
  * VWAP distance (`(Close - VWAP) / VWAP * 100`)
  * Opening Range Breakout (ORB: 9:15–9:30 AM High/Low flags)
  * Normalized ATR(14), RSI(14), EMA(9/21) spread
  * Time-of-day bucketed RVOL (20-day baseline per exact candle slot)
  * Sector relative strength (`Stock 15m return - Sector 15m return`)
  * Cyclical time encoding (`sin/cos` of minutes since 9:15 AM)
* **ML Model:** `XGBoost` binary classifier with `TimeSeriesSplit` (walk-forward validation).
* **Labeling:** Triple-Barrier Method (+0.75% profit vs -0.40% stop loss within $N=4$ candles / 60 mins).
* **Risk & Indian Tax Engine:**
  * Exact calculation of STT (0.025% on sell), NSE charges (0.00297%), SEBI turnover (0.0001%), Stamp duty (0.003% on buy), Brokerage (min 0.03% or ₹20), GST (18%).
  * Minimum Net Reward filter: Net Profit $\ge 1.5 \times$ Net Risk.
* **UI & Alerts:** `lightweight-charts-python` / `Streamlit`, audio alerts + Telegram Bot webhooks.

---

## 4. Key Engineering & Quantitative Decisions to Preserve
1. **No Data Leakage:** Strict ban on random splits; use chronological walk-forward cross-validation.
2. **Realistic Entry Execution:** Signals computed at Candle $T$ close must simulate entry at Candle $T+1$ Open.
3. **Trading Cut-Off Time:** No new entry signals after 14:15 IST to prevent broker auto square-off conflicts (MIS cutoff at 15:15–15:20 IST).
4. **Discrete Share Rounding:** Always compute integer shares via `floor(min(C / Entry, R_max / (Entry - SL)))` and verify post-tax viability on the actual integer quantity.
5. **News & Web Information Architecture:**
   * **Role:** News acts as a **Gatekeeper / Veto Filter** (circuit breaker), not a standalone blind entry trigger.
   * **Time Horizons:**
     - *Past Month:* Macro/regime bias (e.g. quarterly earnings result, regulatory backdrop).
     - *Past 24 Hours / Overnight (Primary):* Pre-market catalyst. If negative news broke overnight, suppress Longs for the day.
     - *Past 1 Hour (Real-Time Shock Filter):* Emergency brake. If negative news hits mid-day, cancel pending signals or warn.
   * **Data Sources (₹0):** Google News RSS by ticker/company name (`feedparser`) and NSE corporate announcements.
   * **Scoring:** Local LLM (`Ollama`) or Gemini Flash free tier to assign a normalized sentiment score $(-1.0 \text{ to } +1.0)$.
6. **Pandas 3.0 Datetime Resolution (Bug Prevention):**
   * Pandas 3.0 defaults timestamps to microseconds (`datetime64[us]`). `lightweight-charts` divides integer timestamps by $10^9$ assuming nanoseconds (`datetime64[ns]`).
   * This previously caused timestamps to collapse to the year 1970 (`1784798`), rendering a blank chart.
   * Rule: Always explicitly cast timestamps to `.astype('datetime64[ns]')` before passing dataframes to `chart.set()` or `line.set()`.
7. **Chart Mode Toggle (Candlestick ⇄ Line Graph):**
   * Added dual-mode visualizer in `main_desktop.py`.
   * Can be toggled interactively by clicking the top row in the HUD table or by pressing hotkey **`V`** or **`G`** on the keyboard.
8. **Hotkey Callback Signature Handling:**
   * `lightweight-charts` passes the triggered key string to hotkey handlers.
   * `toggle_view(*args, **kwargs)` accepts variable arguments to cleanly support hotkeys, table cell clicks, and direct invocation.
9. **Git Repository Setup:**
   * Initialized repository on branch `main` with `.gitignore` (protecting `.venv`, `data/market_data.db`, and caches).
   * Remote configured: `origin -> https://github.com/Raj8313/equity_stock_tracker.git`.
