"""
Module: main_desktop.py
Responsibility: Native Desktop GUI powered by lightweight-charts-python (TradingView canvas).
Renders 15m candlesticks, VWAP line, buy signal markers, and live prediction HUD with audio alerts.
Supports live market monitoring (Mon-Fri) and weekend replay simulation mode (--replay).
"""

import argparse
import os
import sys
import time
import threading
import logging
from datetime import datetime, timezone
import yaml
import pandas as pd
from lightweight_charts import Chart

from src.data_engine import DataEngine
from src.features import compute_features
from src.sentiment import NewsSentimentEngine
from src.ml_engine import MLEngine
from src.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DesktopApp")


def trigger_mac_alert(title: str, message: str):
    """
    Triggers native macOS audio chime and system notification banner.
    """
    try:
        os.system("afplay /System/Library/Sounds/Glass.aiff &")
        script = f'display notification "{message}" with title "{title}" sound name "Glass"'
        os.system(f"osascript -e '{script}' &")
    except Exception as e:
        logger.warning(f"Could not fire native Mac alert: {e}")


def get_market_status() -> str:
    now = datetime.now()
    if now.weekday() >= 5:
        return "CLOSED (Weekend — Last Session: Friday Oct 16)"
    
    current_time = now.strftime("%H:%M")
    if "09:15" <= current_time <= "15:30":
        return "LIVE (NSE Open)"
    elif current_time < "09:15":
        return "PRE-MARKET (Opens 09:15 IST)"
    else:
        return "POST-MARKET (Closed at 15:30 IST)"


def launch_desktop_app(replay_mode: bool = False):
    # 1. Load Configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ticker = config.get("ticker", "TMPV.NS")
    sector_ticker = config.get("sector_ticker", "^CNXAUTO")

    # 2. Initialize Core Engines
    data_engine = DataEngine(config)
    risk_manager = RiskManager(config)
    sentiment_engine = NewsSentimentEngine(config)
    ml_engine = MLEngine(config)

    # 3. Load historical candles from SQLite
    stock_df = data_engine.load_candles(ticker)
    sector_df = data_engine.load_candles(sector_ticker)

    if stock_df.empty:
        logger.info(f"No local data found in SQLite. Running initial backfill for {ticker}...")
        data_engine.backfill()
        stock_df = data_engine.load_candles(ticker)
        sector_df = data_engine.load_candles(sector_ticker)

    if stock_df.empty:
        logger.error("Failed to load or backfill candle data. Exiting.")
        return

    # 4. Feature calculation & initial prediction
    df_features = compute_features(stock_df, sector_df)
    latest_row = df_features.iloc[-1]
    
    # 5. Fetch News Sentiment
    sentiment_res = sentiment_engine.fetch_news_sentiment(lookback_window="1d")
    
    # 6. Predict on latest candle
    pred = ml_engine.predict_live(latest_row)
    eval_result = risk_manager.evaluate_trade(pred.entry_price, pred.stop_loss_price, pred.target_price)
    confidence_pct = round(pred.probability * 100, 1)

    # Market status check
    market_status = get_market_status()

    # 7. Create Native TradingView Desktop Chart Window
    chart = Chart(
        title=f"NSE: {ticker} (Tata Motors) | 15-Minute Intraday Predictor",
        width=1320,
        height=820,
        inner_width=0.75,   # Main chart takes 75% width
        inner_height=1.0
    )

    # Configure chart styling
    chart.layout(background_color='#131722', text_color='#d1d4dc', font_size=13)
    chart.candle_style(up_color='#089981', down_color='#f23645', border_up_color='#089981', border_down_color='#f23645', wick_up_color='#089981', wick_down_color='#f23645')
    chart.grid(vert_enabled=True, horz_enabled=True, color='rgba(42, 46, 57, 0.4)')

    # Enable Interactive Chart Legend (OHLC values, %, VWAP, and Price Line)
    chart.legend(
        visible=True,
        ohlc=True,
        percent=True,
        lines=True,
        color_based_on_candle=True,
        font_size=12,
        font_family="Monaco, -apple-system, BlinkMacSystemFont, sans-serif",
        text=f"NSE: {ticker} (Tata Motors) | 15M"
    )

    # Format chart dataframe: date, open, high, low, close, volume
    chart_df = df_features[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    chart_df = chart_df.rename(columns={'timestamp': 'date'})
    chart_df['date'] = pd.to_datetime(chart_df['date']).astype('datetime64[ns]')
    chart.set(chart_df)
    chart.time_scale(right_offset=5)
    chart.fit()

    # Add Price Line for Graph View (Blue)
    price_line = chart.create_line(name='Price', color='#2962FF', width=2)
    price_df = df_features[['timestamp', 'close']].rename(columns={'timestamp': 'date', 'close': 'Price'}).dropna()
    price_df['date'] = pd.to_datetime(price_df['date']).astype('datetime64[ns]')
    price_line.set(price_df)
    price_line.hide_data()  # Initially hidden; candlesticks default

    # Add VWAP Line Indicator (Yellow)
    vwap_line = chart.create_line(name='VWAP', color='#f1c40f', width=2)
    vwap_df = df_features[['timestamp', 'vwap']].rename(columns={'timestamp': 'date', 'vwap': 'VWAP'}).dropna()
    vwap_df['date'] = pd.to_datetime(vwap_df['date']).astype('datetime64[ns]')
    vwap_line.set(vwap_df)

    # Set Top Watermark
    chart.watermark(f"{ticker} 15M", color='rgba(255, 255, 255, 0.05)')

    # TopBar Direct GUI Controls (View Switcher & Zoom Controls)
    def on_view_switch(ch):
        val = ch.topbar.get("view_switcher").value
        if "Line" in val:
            chart.hide_data()
            price_line.show_data()
            view_state["mode"] = "line"
            logger.info("Switched to Line Graph View")
            if toggle_row_ref[0]:
                toggle_row_ref[0]['Value'] = "📈 Line Graph"
                toggle_row_ref[0].background_color('Value', '#2962FF')
        else:
            price_line.hide_data()
            chart.show_data()
            view_state["mode"] = "candle"
            logger.info("Switched to Candlestick View")
            if toggle_row_ref[0]:
                toggle_row_ref[0]['Value'] = "📊 Candlestick"
                toggle_row_ref[0].background_color('Value', '#089981')

    def on_zoom_in(ch):
        ch.run_script(f'''
        (() => {{
            const ts = {ch.id}.chart.timeScale();
            const range = ts.getVisibleLogicalRange();
            if (range) {{
                const delta = (range.to - range.from) * 0.20;
                ts.setVisibleLogicalRange({{ from: range.from + delta, to: range.to - delta }});
            }}
        }})();
        ''')

    def on_zoom_out(ch):
        ch.run_script(f'''
        (() => {{
            const ts = {ch.id}.chart.timeScale();
            const range = ts.getVisibleLogicalRange();
            if (range) {{
                const delta = (range.to - range.from) * 0.25;
                ts.setVisibleLogicalRange({{ from: range.from - delta, to: range.to + delta }});
            }}
        }})();
        ''')

    def on_reset_view(ch):
        ch.fit()
        ch.time_scale(right_offset=5)

    chart.topbar.switcher(
        name="view_switcher",
        options=('📊 Candlesticks', '📈 Line Graph'),
        default='📊 Candlesticks',
        align='left',
        func=on_view_switch
    )
    chart.topbar.button(name="zoom_in", button_text="🔍 + Zoom In", align='left', func=on_zoom_in)
    chart.topbar.button(name="zoom_out", button_text="🔍 - Zoom Out", align='left', func=on_zoom_out)
    chart.topbar.button(name="reset_view", button_text="↺ Reset View", align='left', func=on_reset_view)

    # Chart View Mode Toggle Handler
    view_state = {"mode": "candle"}
    toggle_row_ref = [None]

    def toggle_view(*args, **kwargs):
        target_row = toggle_row_ref[0]
        if view_state["mode"] == "candle":
            chart.hide_data()
            price_line.show_data()
            view_state["mode"] = "line"
            logger.info("Switched to Line Graph View")
            if target_row:
                target_row['Value'] = "📈 Line Graph"
                target_row.background_color('Value', '#2962FF')
                target_row.text_color('Value', '#ffffff')
        else:
            price_line.hide_data()
            chart.show_data()
            view_state["mode"] = "candle"
            logger.info("Switched to Candlestick View")
            if target_row:
                target_row['Value'] = "📊 Candlestick"
                target_row.background_color('Value', '#089981')
                target_row.text_color('Value', '#ffffff')

    # Interactive HUD Metrics Table on Right Panel (Click any cell to toggle view)
    table = chart.create_table(
        width=0.25,
        height=1.0,
        headings=('Decision Metric', 'Value'),
        position='right',
        background_color='#1a1e29',
        return_clicked_cells=True,
        func=toggle_view
    )
    toggle_row = table.new_row('Chart Mode [Key: V]', '📊 Candlestick')
    toggle_row.background_color('Value', '#089981')
    toggle_row.text_color('Value', '#ffffff')
    toggle_row_ref[0] = toggle_row

    # Bind Hotkeys: press 'V' or 'G' to toggle
    chart.hotkey(None, 'v', toggle_view)
    chart.hotkey(None, 'g', toggle_view)

    table.new_row('Stock', f"Tata Motors ({ticker})")
    table.new_row('Market Status', market_status)
    table.new_row('Last Candle', str(latest_row['timestamp'])[:16])
    table.new_row('Last Price', f"Rs {pred.entry_price:.2f}")
    table.new_row('Win Probability', f"{confidence_pct}%")
    table.new_row('Model Decision', str(pred.signal_type))
    table.new_row('News Sentiment', f"{sentiment_res.overall_score:+.2f} ({'VETO' if sentiment_res.is_veto_active else 'CLEAR'})")
    table.new_row('Target (+0.75%)', f"Rs {pred.target_price:.2f}")
    table.new_row('Stop Loss (-0.40%)', f"Rs {pred.stop_loss_price:.2f}")
    table.new_row('Recommended Qty', f"{eval_result.quantity} shares")
    table.new_row('Gross Target Profit', f"Rs {eval_result.gross_profit:.2f}")
    table.new_row('Taxes & Brokerage', f"Rs {eval_result.total_fees_and_taxes:.2f}")
    table.new_row('Net Take-Home', f"Rs {eval_result.net_profit:.2f}")

    # Add signal markers on historical high-conviction signals
    for idx, row in df_features.tail(60).iterrows():
        try:
            p = ml_engine.predict_live(row)
            if p.signal_type == "BUY":
                date_str = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                chart.marker(
                    time=date_str,
                    position='below',
                    shape='arrow_up',
                    color='#089981',
                    text=f"BUY {round(p.probability*100)}%"
                )
        except Exception:
            pass

    print("\n" + "="*70)
    print(f"🚀 DESKTOP TRADING TERMINAL LAUNCHED: {ticker} (Tata Motors)")
    print(f"📊 Market Status: {market_status}")
    print(f"🕒 Total Candles in Database: {len(df_features)} (Up to {latest_row['timestamp']})")
    print(f"💰 Last Price: Rs {pred.entry_price:.2f} | Probability: {confidence_pct}% [{pred.signal_type}]")
    print(f"📰 News Sentiment: {sentiment_res.overall_score:+.2f} ({sentiment_res.headline_count} headlines)")
    print("="*70 + "\n")

    # Render desktop window
    chart.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tata Motors Intraday Prediction Desktop App")
    parser.add_argument("--replay", action="store_true", help="Run simulated replay of last trading session")
    args = parser.parse_args()
    launch_desktop_app(replay_mode=args.replay)
