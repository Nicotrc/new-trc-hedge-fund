"""SectorRotation strategy scanner — L2 Signal Detection.

Identifies tickers benefiting from sector rotation capital flows.
Upgrade of the flat sector_momentum detector.

Factors:
  1. RS rank improvement (sector moving up)  weight 0.30
  2. Breadth expansion                       weight 0.25
  3. Flow momentum (sector ETF acceleration) weight 0.25
  4. Macro alignment                         weight 0.20
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

STRATEGY_NAME = "sector_rotation"
QUALITY_GATE = 50.0

# Sector ETF map for rotation tracking
SECTOR_ETF_MAP: dict[str, str] = {
    "XLK": "technology",
    "XLV": "healthcare",
    "XLF": "financials",
    "XLE": "energy",
    "XLI": "industrials",
    "XLB": "materials",
    "XLU": "utilities",
    "XLRE": "real_estate",
    "XLP": "consumer_staples",
    "XLY": "consumer_discretionary",
    "XLC": "communication",
    "IBB": "biotech",
}


def _sector_for_ticker(ticker: str) -> Optional[str]:
    """Infer sector ETF from environment-configured sector map."""
    sector_map_raw = os.environ.get("SECTOR_MAP", "")
    for pair in sector_map_raw.split(","):
        if ":" in pair:
            t, etf = pair.strip().split(":", 1)
            if t.strip().upper() == ticker.upper():
                return etf.strip().upper()
    return None


def _score_rs_improvement(
    ticker_5d: float,
    ticker_20d: float,
    sector_5d: float,
    sector_20d: float,
) -> float:
    """Improvement in relative strength = was lagging, now leading."""
    rs_recent = ticker_5d - sector_5d
    rs_prior = ticker_20d - sector_20d
    improvement = rs_recent - rs_prior
    if improvement >= 0.05:
        return 95.0
    if improvement >= 0.02:
        return 75.0
    if improvement >= 0.0:
        return 55.0
    return 20.0


def _score_breadth(sector_bars: list[dict[str, Any]]) -> float:
    """Sector ETF volume expanding = broad participation."""
    if len(sector_bars) < 6:
        return 50.0
    volumes = [int(b.get("volume", 0) or 0) for b in sector_bars]
    recent_avg = float(np.mean(volumes[-3:]))
    base_avg = float(np.mean(volumes[-6:-3]))
    if base_avg <= 0:
        return 50.0
    ratio = recent_avg / base_avg
    if ratio >= 1.5:
        return 90.0
    if ratio >= 1.2:
        return 70.0
    if ratio >= 1.0:
        return 50.0
    return 25.0


def _score_flow_momentum(sector_bars: list[dict[str, Any]]) -> float:
    """Sector ETF price momentum accelerating."""
    if len(sector_bars) < 10:
        return 40.0
    closes = [float(b.get("close", 0) or 0) for b in sector_bars]
    if closes[-1] <= 0 or closes[-6] <= 0 or closes[-10] <= 0:
        return 40.0
    return_5d = (closes[-1] - closes[-6]) / closes[-6]
    return_5d_10 = (closes[-6] - closes[-10]) / closes[-10]
    acceleration = return_5d - return_5d_10
    if acceleration >= 0.02:
        return 90.0
    if acceleration >= 0.01:
        return 70.0
    if acceleration >= 0:
        return 50.0
    return 20.0


def _score_macro_alignment(macro_data: Optional[dict[str, Any]], sector_etf: Optional[str]) -> float:
    """Does macro environment support this sector?"""
    if not macro_data:
        return 50.0
    vix = float(macro_data.get("VIX", 20.0))
    dxy_change = float(macro_data.get("DXY_change_5d", 0.0))

    # Biotech/growth: benefits from risk-on, falling yields
    if sector_etf in ("IBB", "XLK", "XLY"):
        if vix < 18 and dxy_change < 0:
            return 85.0
        if vix > 25:
            return 20.0
        return 55.0

    # Energy: benefits from commodity cycle
    if sector_etf == "XLE":
        return 65.0

    # Utilities/staples: defensive in risk-off
    if sector_etf in ("XLU", "XLP"):
        if vix > 25:
            return 80.0
        return 40.0

    return 55.0  # default neutral


def scan(
    ticker: str,
    ticker_bars: list[dict[str, Any]],
    sector_bars: Optional[list[dict[str, Any]]] = None,
    macro_snapshot: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Run SectorRotation scan for a single ticker."""
    if len(ticker_bars) < 21:
        return None

    closes = [float(b.get("close", 0) or 0) for b in ticker_bars]
    if not closes or closes[-1] <= 0:
        return None

    current_close = closes[-1]

    # Compute returns
    ticker_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] > 0 else 0.0
    ticker_20d = (closes[-1] - closes[-21]) / closes[-21] if len(closes) >= 21 and closes[-21] > 0 else 0.0

    # Sector bars
    sector_5d = 0.0
    sector_20d = 0.0
    if sector_bars and len(sector_bars) >= 21:
        sc = [float(b.get("close", 0) or 0) for b in sector_bars]
        sector_5d = (sc[-1] - sc[-6]) / sc[-6] if sc[-6] > 0 else 0.0
        sector_20d = (sc[-1] - sc[-21]) / sc[-21] if sc[-21] > 0 else 0.0

    sector_etf = _sector_for_ticker(ticker)

    f1 = _score_rs_improvement(ticker_5d, ticker_20d, sector_5d, sector_20d)
    f2 = _score_breadth(sector_bars) if sector_bars else 40.0
    f3 = _score_flow_momentum(sector_bars) if sector_bars else 40.0
    f4 = _score_macro_alignment(macro_snapshot, sector_etf)

    score = f1 * 0.30 + f2 * 0.25 + f3 * 0.25 + f4 * 0.20

    if score < QUALITY_GATE:
        return None

    expected_move_pct = round(ticker_5d * 2 * 100, 2)  # project forward

    logger.info(
        "SectorRotation signal: %s score=%.1f RS_improvement=%.1f sector=%s",
        ticker, score, f1, sector_etf or "unknown",
    )

    return {
        "ticker": ticker,
        "strategy": STRATEGY_NAME,
        "score": round(score, 2),
        "confidence": round(min(score / 100.0, 1.0), 3),
        "expected_move_pct": expected_move_pct,
        "time_horizon_days": 15,
        "catalyst": f"Sector rotation into {sector_etf or 'sector'}",
        "entry_zone_low": round(current_close, 2),
        "entry_zone_high": round(current_close * 1.02, 2),
        "invalidation_price": round(current_close * 0.94, 2),
        "binary_event": False,
        "detail": {
            "ticker_5d_return": round(ticker_5d * 100, 2),
            "sector_5d_return": round(sector_5d * 100, 2),
            "rs_score": round(f1, 1),
            "breadth_score": round(f2, 1),
            "flow_score": round(f3, 1),
            "macro_score": round(f4, 1),
            "sector_etf": sector_etf,
        },
    }
