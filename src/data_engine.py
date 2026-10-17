"""
Module: data_engine.py
Responsibility: Historical and live data ingestion via yfinance with local SQLite persistence.
Manages market_data.db and ensures no data duplicates.
"""

from datetime import datetime, timezone
import logging
import os
import sqlite3
from typing import Dict, Any, Optional, List
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ticker = config.get("ticker", "TATAMOTORS.NS")
        self.sector_ticker = config.get("sector_ticker", "^CNXAUTO")
        self.timeframe = config.get("timeframe", "15m")
        self.lookback_days = config.get("lookback_days", 60)
        
        db_cfg = config.get("database", {})
        self.db_path = db_cfg.get("path", "data/market_data.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self._init_sqlite()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self) -> None:
        """
        Initializes schema for candles_15m and signal_logs.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles_15m (
                    ticker TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    vwap REAL,
                    PRIMARY KEY (ticker, timestamp)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_logs (
                    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    close_price REAL NOT NULL,
                    pred_probability REAL NOT NULL,
                    sentiment_score REAL DEFAULT 0.0,
                    recommended_qty INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    target_price REAL NOT NULL,
                    est_taxes_inr REAL NOT NULL,
                    actual_outcome INTEGER DEFAULT NULL
                )
            """)
            conn.commit()
            logger.info(f"Initialized SQLite database at {self.db_path}")

    def fetch_yfinance_history(self, symbol: str, days: int = 60, interval: str = "15m") -> pd.DataFrame:
        """
        Fetches historical intraday candles via yfinance (up to 60 days for 15m).
        """
        logger.info(f"Downloading {days}d of {interval} candles for {symbol}...")
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period=f"{days}d", interval=interval)
        
        if df.empty:
            logger.warning(f"No data returned from yfinance for {symbol}")
            return pd.DataFrame()

        # Clean and standardize columns
        df = df.reset_index()
        # Rename Datetime / Date column to timestamp
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(columns={
            date_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        
        # Convert timestamp to ISO string without tz offsets
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['ticker'] = symbol
        
        cols = ['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        return df[cols].dropna()

    def save_candles(self, df: pd.DataFrame) -> int:
        """
        Upserts candles into SQLite candles_15m table.
        """
        if df.empty:
            return 0
        
        records = df[['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO candles_15m (ticker, timestamp, open, high, low, close, volume)
                VALUES (:ticker, :timestamp, :open, :high, :low, :close, :volume)
                ON CONFLICT(ticker, timestamp) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume
            """, records)
            conn.commit()
            return len(records)

    def backfill(self) -> Dict[str, int]:
        """
        Backfills historical data for both target ticker and sector benchmark.
        """
        stock_df = self.fetch_yfinance_history(self.ticker, days=self.lookback_days, interval=self.timeframe)
        stock_count = self.save_candles(stock_df)

        sector_df = self.fetch_yfinance_history(self.sector_ticker, days=self.lookback_days, interval=self.timeframe)
        sector_count = self.save_candles(sector_df)

        return {self.ticker: stock_count, self.sector_ticker: sector_count}

    def load_candles(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Loads candles from SQLite as a clean sorted DataFrame.
        """
        sym = symbol or self.ticker
        with self._get_connection() as conn:
            query = "SELECT * FROM candles_15m WHERE ticker = ? ORDER BY timestamp ASC"
            df = pd.read_sql_query(query, conn, params=(sym,))
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp']).astype('datetime64[ns]')
            return df

    def log_signal(self, log_dict: Dict[str, Any]) -> int:
        """
        Logs a generated trade signal or paper trade into signal_logs.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO signal_logs (
                    timestamp, ticker, close_price, pred_probability, sentiment_score,
                    recommended_qty, entry_price, stop_loss, target_price, est_taxes_inr
                ) VALUES (
                    :timestamp, :ticker, :close_price, :pred_probability, :sentiment_score,
                    :recommended_qty, :entry_price, :stop_loss, :target_price, :est_taxes_inr
                )
            """, log_dict)
            conn.commit()
            return cursor.lastrowid
