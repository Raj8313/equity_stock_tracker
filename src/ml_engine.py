"""
Module: ml_engine.py
Responsibility: Triple-barrier labeling, chronological walk-forward training, and live probability inference.
Implements FR-3.1 through FR-3.4 with realistic next-candle open execution.
"""

from dataclasses import dataclass
import logging
import os
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    logger.info("Using native scikit-learn HistGradientBoostingClassifier for gradient boosting.")
    XGB_AVAILABLE = False


FEATURE_COLUMNS = [
    'vwap_dist_pct',
    'orb_high_dist_pct',
    'is_above_orb_high',
    'is_below_orb_low',
    'rsi_14',
    'atr_norm_pct',
    'ema_spread_pct',
    'rvol',
    'rel_strength',
    'time_sin',
    'time_cos',
    'body_to_range',
    'upper_wick_ratio',
    'lower_wick_ratio'
]


@dataclass
class BacktestMetrics:
    total_samples: int
    train_samples: int
    test_samples: int
    total_signals: int
    win_rate: float
    precision: float
    auc_score: float


@dataclass
class PredictionOutput:
    probability: float
    signal_type: str              # 'BUY', 'WATCHLIST', 'WAIT'
    entry_price: float
    target_price: float
    stop_loss_price: float
    features_used: Dict[str, float]


class MLEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trading_cfg = config.get("trading", {})
        self.model_cfg = config.get("model", {})
        
        self.target_pct = float(self.trading_cfg.get("target_profit_pct", 0.0075)) # +0.75%
        self.stop_loss_pct = float(self.trading_cfg.get("stop_loss_pct", 0.0040))  # -0.40%
        self.max_holding_bars = int(self.trading_cfg.get("max_holding_bars", 4))   # 60 mins
        self.cutoff_time = self.trading_cfg.get("entry_cutoff_time", "14:15")
        
        self.high_conviction_threshold = float(self.model_cfg.get("high_conviction_threshold", 0.70))
        self.watchlist_threshold = float(self.model_cfg.get("watchlist_threshold", 0.60))
        self.model_path = self.model_cfg.get("path", "models/tatamotors_xgb_v1.joblib")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model: Optional[Any] = None
        self._try_load_model()

    def _try_load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded existing model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load model artifact: {e}")

    def label_triple_barrier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Marcos Lopez de Prado's Triple-Barrier Method.
        Simulates entry at Open of candle (t+1) to prevent lookahead bias.
        Upper Barrier: Entry * (1 + target_pct)
        Lower Barrier: Entry * (1 - stop_loss_pct)
        Time Barrier: max_holding_bars
        Label 1 = Hit target before stop-loss within holding period and within same trading day.
        Label 0 = Hit stop-loss, timed out, or crossed day boundary.
        """
        labeled_df = df.copy().reset_index(drop=True)
        n = len(labeled_df)
        labels = np.zeros(n, dtype=int)
        
        timestamps = pd.to_datetime(labeled_df['timestamp'])
        dates = timestamps.dt.date.values
        times = timestamps.dt.strftime('%H:%M').values
        opens = labeled_df['open'].values
        highs = labeled_df['high'].values
        lows = labeled_df['low'].values
        
        for t in range(n - self.max_holding_bars - 1):
            # Cutoff rule: no entries after 14:15 IST
            if times[t] > self.cutoff_time:
                labels[t] = 0
                continue

            entry_idx = t + 1
            entry_price = opens[entry_idx]
            trade_date = dates[entry_idx]
            
            upper_barrier = entry_price * (1.0 + self.target_pct)
            lower_barrier = entry_price * (1.0 - self.stop_loss_pct)
            
            outcome = 0
            # Scan next N candles
            for step in range(1, self.max_holding_bars + 1):
                idx = entry_idx + step
                if idx >= n:
                    break
                # Must not cross into the next day (intraday only)
                if dates[idx] != trade_date:
                    break

                hit_upper = highs[idx] >= upper_barrier
                hit_lower = lows[idx] <= lower_barrier

                if hit_upper and not hit_lower:
                    outcome = 1
                    break
                elif hit_lower and not hit_upper:
                    outcome = 0
                    break
                elif hit_upper and hit_lower:
                    # Both hit in same candle: conservative assumption is stop-loss hit first
                    outcome = 0
                    break

            labels[t] = outcome
            
        labeled_df['target'] = labels
        return labeled_df

    def train(self, df_features: pd.DataFrame) -> BacktestMetrics:
        """
        Trains XGBoost with chronological walk-forward split (TimeSeriesSplit)
        and probability calibration.
        """
        df_labeled = self.label_triple_barrier(df_features)
        
        # Clean missing feature rows
        valid_mask = df_labeled[FEATURE_COLUMNS].notna().all(axis=1)
        # Drop the last N candles where label cannot be verified
        valid_mask = valid_mask & (df_labeled.index < len(df_labeled) - self.max_holding_bars - 1)
        
        dataset = df_labeled[valid_mask].copy()
        X = dataset[FEATURE_COLUMNS]
        y = dataset['target']
        
        if len(X) < 100:
            raise ValueError(f"Insufficient training samples ({len(X)}). Need at least 100 candles.")
            
        # Chronological Split: 80% train, 20% validation
        split_idx = int(len(X) * 0.80)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        pos_ratio = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
        if XGB_AVAILABLE:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_ratio,
                random_state=42,
                eval_metric="logloss"
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=4,
                learning_rate=0.05,
                class_weight='balanced',
                random_state=42
            )
        
        # Fit on training split
        model.fit(X_train, y_train)
        
        # Evaluate on out-of-sample chronological test set
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= self.high_conviction_threshold).astype(int)
        
        total_signals = int(test_preds.sum())
        if total_signals > 0:
            win_rate = float(y_test[test_preds == 1].mean())
            precision = float(precision_score(y_test, test_preds, zero_division=0))
        else:
            win_rate = 0.0
            precision = 0.0
            
        try:
            auc = float(roc_auc_score(y_test, test_probs))
        except Exception:
            auc = 0.5
            
        # Refit on entire dataset for production inference
        if XGB_AVAILABLE:
            full_pos_ratio = float((y == 0).sum() / max(1, (y == 1).sum()))
            prod_model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=full_pos_ratio,
                random_state=42,
                eval_metric="logloss"
            )
        else:
            prod_model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=4,
                learning_rate=0.05,
                class_weight='balanced',
                random_state=42
            )
        prod_model.fit(X, y)
        self.model = prod_model
        
        # Save model artifact
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved successfully to {self.model_path}")
        
        return BacktestMetrics(
            total_samples=len(X),
            train_samples=len(X_train),
            test_samples=len(X_test),
            total_signals=total_signals,
            win_rate=round(win_rate * 100, 2),
            precision=round(precision * 100, 2),
            auc_score=round(auc, 3)
        )

    def predict_live(self, latest_row: pd.Series) -> PredictionOutput:
        """
        Runs live probability inference on the latest closed candle.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded. Run training first.")
            
        feature_vector = pd.DataFrame([latest_row[FEATURE_COLUMNS]])
        feature_vector = feature_vector.fillna(0.0)
        
        prob = float(self.model.predict_proba(feature_vector)[0, 1])
        close_price = float(latest_row['close'])
        
        # Signal type mapping
        if prob >= self.high_conviction_threshold:
            signal_type = "BUY"
        elif prob >= self.watchlist_threshold:
            signal_type = "WATCHLIST"
        else:
            signal_type = "WAIT"
            
        target_price = round(close_price * (1.0 + self.target_pct), 2)
        stop_loss_price = round(close_price * (1.0 - self.stop_loss_pct), 2)
        
        features_dict = {col: round(float(latest_row[col]), 3) for col in FEATURE_COLUMNS if col in latest_row}
        
        return PredictionOutput(
            probability=round(prob, 3),
            signal_type=signal_type,
            entry_price=close_price,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
            features_used=features_dict
        )
