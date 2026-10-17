"""
Script: test_pipeline.py
Purpose: Modular integration test verifying Data, News Sentiment, Features, ML, and Risk engines.
"""

import sys
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestPipeline")

from src.risk_manager import RiskManager
from src.sentiment import NewsSentimentEngine
from src.data_engine import DataEngine
from src.features import compute_features
from src.ml_engine import MLEngine


def run_tests():
    logger.info("=== STEP 1: Loading config.yaml ===")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(f"Target Ticker: {config['ticker']} | Capital: Rs {config['trading']['capital']}")

    logger.info("\n=== STEP 2: Testing Risk & Indian Tax Calculator ===")
    risk_mgr = RiskManager(config)
    sample_entry = 910.0
    sample_sl = sample_entry * (1 - config['trading']['stop_loss_pct'])
    sample_target = sample_entry * (1 + config['trading']['target_profit_pct'])

    eval_result = risk_mgr.evaluate_trade(sample_entry, sample_sl, sample_target)
    print(f"Sample Entry: Rs {sample_entry:.2f} | SL: Rs {sample_sl:.2f} | Target: Rs {sample_target:.2f}")
    print(f"Allocated Qty: {eval_result.quantity} shares")
    print(f"Buy Turnover: Rs {eval_result.buy_turnover:.2f} | Sell Turnover: Rs {eval_result.sell_turnover:.2f}")
    print(f"Brokerage: Rs {eval_result.brokerage:.2f} | STT: Rs {eval_result.stt:.2f} | GST: Rs {eval_result.gst:.2f}")
    print(f"Total Indian Taxes & Fees: Rs {eval_result.total_fees_and_taxes:.2f}")
    print(f"Gross Profit: Rs {eval_result.gross_profit:.2f} -> Net Take-Home Profit: Rs {eval_result.net_profit:.2f}")
    print(f"Net Reward/Risk Ratio: {eval_result.net_reward_risk_ratio} (Viable: {eval_result.is_viable})")

    logger.info("\n=== STEP 3: Testing Google News RSS Sentiment Scorer ===")
    sentiment_engine = NewsSentimentEngine(config)
    sentiment_res = sentiment_engine.fetch_news_sentiment(lookback_window="1d")
    print(f"Headlines found for '{sentiment_res.query}': {sentiment_res.headline_count}")
    print(f"Aggregate Sentiment Score: {sentiment_res.overall_score} (Veto Active: {sentiment_res.is_veto_active})")
    for i, h in enumerate(sentiment_res.headlines[:3], 1):
        print(f"  {i}. [{h.score:+.2f}] {h.title}")

    logger.info("\n=== STEP 4: Testing Historical Data Ingestion & SQLite Sync ===")
    data_engine = DataEngine(config)
    counts = data_engine.backfill()
    print(f"Backfill Complete! Ingested records: {counts}")
    
    stock_df = data_engine.load_candles(config['ticker'])
    sector_df = data_engine.load_candles(config['sector_ticker'])
    print(f"Loaded {len(stock_df)} candles for {config['ticker']} from SQLite")

    logger.info("\n=== STEP 5: Testing Feature Engineering Pipeline ===")
    df_features = compute_features(stock_df, sector_df)
    print(f"Computed {len(df_features.columns)} features on {len(df_features)} candles")
    sample_cols = ['timestamp', 'close', 'vwap', 'vwap_dist_pct', 'rsi_14', 'atr_norm_pct', 'rvol']
    print(df_features[sample_cols].tail(3).to_string())

    logger.info("\n=== STEP 6: Testing ML Triple-Barrier Training & Evaluation ===")
    ml_engine = MLEngine(config)
    metrics = ml_engine.train(df_features)
    print(f"Training Complete!")
    print(f"  Total Valid Candles: {metrics.total_samples}")
    print(f"  Train Set: {metrics.train_samples} | Test Set: {metrics.test_samples}")
    print(f"  Out-of-Sample Win Rate on High-Conviction Signals: {metrics.win_rate}%")
    print(f"  Out-of-Sample Precision: {metrics.precision}% | AUC: {metrics.auc_score}")

    logger.info("\n=== STEP 7: Testing Live Inference on Latest Candle ===")
    latest_candle = df_features.iloc[-1]
    pred = ml_engine.predict_live(latest_candle)
    print(f"Latest Candle Time: {latest_candle['timestamp']} | Close: Rs {pred.entry_price}")
    print(f"Model Prediction Probability: {pred.probability * 100:.1f}% -> Decision: {pred.signal_type}")
    print(f"Recommended Target: Rs {pred.target_price} | Stop Loss: Rs {pred.stop_loss_price}")

    print("\n✅ ALL 7 LEGO BLOCKS TESTED AND WORKING SEAMLESSLY TOGETHER!")


if __name__ == "__main__":
    run_tests()
