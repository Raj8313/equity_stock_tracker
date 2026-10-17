"""
Module: risk_manager.py
Responsibility: Unleveraged intraday position sizing and Indian statutory tax & charge deductions.
Enforces that net post-tax profit justifies risk before executing.
"""

from dataclasses import dataclass
import math
from typing import Dict, Any, Optional


@dataclass
class TradeCostBreakdown:
    quantity: int
    entry_price: float
    target_price: float
    stop_loss_price: float
    buy_turnover: float
    sell_turnover: float
    total_turnover: float
    brokerage: float
    stt: float
    exchange_charges: float
    sebi_fee: float
    stamp_duty: float
    gst: float
    total_fees_and_taxes: float
    gross_profit: float
    net_profit: float
    net_risk: float
    net_reward_risk_ratio: float
    is_viable: bool
    rejection_reason: Optional[str] = None


class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with config dict (from config.yaml).
        """
        self.trading_cfg = config.get("trading", {})
        self.tax_cfg = config.get("statutory_taxes", {})
        
        self.capital = float(self.trading_cfg.get("capital", 10000.0))
        self.risk_pct = float(self.trading_cfg.get("risk_per_trade_pct", 0.01))
        self.min_rr_ratio = float(self.trading_cfg.get("min_reward_risk_ratio", 1.5))
        
        # Statutory rates
        self.brokerage_pct = float(self.tax_cfg.get("brokerage_pct", 0.0003))
        self.brokerage_max = float(self.tax_cfg.get("brokerage_max_per_leg", 20.0))
        self.stt_sell_pct = float(self.tax_cfg.get("stt_sell_pct", 0.00025))
        self.nse_txn_pct = float(self.tax_cfg.get("nse_txn_charge_pct", 0.0000297))
        self.sebi_fee_pct = float(self.tax_cfg.get("sebi_fee_pct", 0.000001))
        self.stamp_duty_pct = float(self.tax_cfg.get("stamp_duty_buy_pct", 0.00003))
        self.gst_pct = float(self.tax_cfg.get("gst_pct", 0.18))

    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> int:
        """
        Calculates unleveraged integer share quantity based on max risk and total capital.
        Qty = min( floor(Capital / Entry), floor(MaxRisk / (Entry - SL)) )
        """
        if entry_price <= 0 or stop_loss_price >= entry_price:
            return 0
        
        risk_per_share = entry_price - stop_loss_price
        max_rupee_risk = self.capital * self.risk_pct
        
        qty_by_risk = math.floor(max_rupee_risk / risk_per_share)
        qty_by_capital = math.floor(self.capital / entry_price)
        
        return max(0, min(qty_by_risk, qty_by_capital))

    def calculate_roundtrip_taxes(self, qty: int, buy_price: float, sell_price: float) -> Dict[str, float]:
        """
        Calculates all Indian statutory taxes and brokerage for a round-trip intraday trade.
        """
        buy_turnover = qty * buy_price
        sell_turnover = qty * sell_price
        total_turnover = buy_turnover + sell_turnover

        # Brokerage: min(0.03% or Rs 20) per leg
        buy_brokerage = min(self.brokerage_pct * buy_turnover, self.brokerage_max)
        sell_brokerage = min(self.brokerage_pct * sell_turnover, self.brokerage_max)
        brokerage = buy_brokerage + sell_brokerage

        # STT: 0.025% on intraday sell leg only
        stt = round(self.stt_sell_pct * sell_turnover, 2)

        # NSE Exchange Transaction Charge: 0.00297% on total turnover
        nse_charges = round(self.nse_txn_pct * total_turnover, 2)

        # SEBI Turnover Fee: 0.0001% on total turnover
        sebi_fee = round(self.sebi_fee_pct * total_turnover, 2)

        # Stamp Duty: 0.003% on buy leg only
        stamp_duty = round(self.stamp_duty_pct * buy_turnover, 2)

        # GST: 18% on (Brokerage + NSE charges + SEBI fee)
        gst = round(self.gst_pct * (brokerage + nse_charges + sebi_fee), 2)

        total_fees = brokerage + stt + nse_charges + sebi_fee + stamp_duty + gst

        return {
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "total_turnover": total_turnover,
            "brokerage": brokerage,
            "stt": stt,
            "exchange_charges": nse_charges,
            "sebi_fee": sebi_fee,
            "stamp_duty": stamp_duty,
            "gst": gst,
            "total_fees": round(total_fees, 2)
        }

    def evaluate_trade(self, entry_price: float, stop_loss_price: float, target_price: float) -> TradeCostBreakdown:
        """
        Full evaluation: sizing, tax breakdown, net reward/risk ratio, and viability gate.
        """
        qty = self.calculate_position_size(entry_price, stop_loss_price)
        if qty <= 0:
            return TradeCostBreakdown(
                quantity=0,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                buy_turnover=0.0,
                sell_turnover=0.0,
                total_turnover=0.0,
                brokerage=0.0,
                stt=0.0,
                exchange_charges=0.0,
                sebi_fee=0.0,
                stamp_duty=0.0,
                gst=0.0,
                total_fees_and_taxes=0.0,
                gross_profit=0.0,
                net_profit=0.0,
                net_risk=0.0,
                net_reward_risk_ratio=0.0,
                is_viable=False,
                rejection_reason="Insufficient capital or risk allocation for even 1 share"
            )

        costs = self.calculate_roundtrip_taxes(qty, entry_price, target_price)
        gross_profit = qty * (target_price - entry_price)
        net_profit = gross_profit - costs["total_fees"]

        # Risk on stop loss leg
        sl_costs = self.calculate_roundtrip_taxes(qty, entry_price, stop_loss_price)
        gross_loss = qty * (entry_price - stop_loss_price)
        net_risk = gross_loss + sl_costs["total_fees"]

        net_rr = (net_profit / net_risk) if net_risk > 0 else 0.0

        is_viable = (net_profit > 0) and (net_rr >= self.min_rr_ratio)
        rejection_reason = None
        if not is_viable:
            if net_profit <= 0:
                rejection_reason = f"Net profit after taxes (Rs {net_profit:.2f}) is negative"
            else:
                rejection_reason = f"Net reward/risk ({net_rr:.2f}) below required minimum ({self.min_rr_ratio:.2f})"

        return TradeCostBreakdown(
            quantity=qty,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
            buy_turnover=costs["buy_turnover"],
            sell_turnover=costs["sell_turnover"],
            total_turnover=costs["total_turnover"],
            brokerage=costs["brokerage"],
            stt=costs["stt"],
            exchange_charges=costs["exchange_charges"],
            sebi_fee=costs["sebi_fee"],
            stamp_duty=costs["stamp_duty"],
            gst=costs["gst"],
            total_fees_and_taxes=costs["total_fees"],
            gross_profit=round(gross_profit, 2),
            net_profit=round(net_profit, 2),
            net_risk=round(net_risk, 2),
            net_reward_risk_ratio=round(net_rr, 2),
            is_viable=is_viable,
            rejection_reason=rejection_reason
        )
