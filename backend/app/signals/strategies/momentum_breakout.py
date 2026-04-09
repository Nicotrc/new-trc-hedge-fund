"""MomentumBreakout strategy scanner — L2 Signal Detection.

Replaces the flat volume_spike + price_breakout detectors.
Score = weighted sum of 5 factors (0-100 scale).

Factors:
  1. Breakout magnitude vs ATR    weight 0.35
  2. Volume ratio vs 20d avg      weight 0.25
  3. ATR expansion                weight 0.20
  4. Consolidation quality        weight 0.10
  5. Sector relative strength     weight 0.10

Quality gate: score >= 55 (replaces static 0.35 threshold).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

STRATEGY_NAME = "momentum_breakout"
QUALITY_GATE = 55.0


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    """Compute Average True Range."""
    if len(highs) < period + 1:
        return (max(highs) - min(lows)) / len(highs) if highs else 0.01
    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return float(np.mean(trs[-period:]))


def _score_breakout_magnitude(
    close: float,
    high_20d: float,
    atr: float,
) -> float:
    """Factor 1: Breakout magnitude normalised by ATR."""
    if atr <= 0:
        return 0.0
    magnitude = (close - high_20d) / atr
    if magnitude <= 0:
        return 0.0
    if magnitude >= 2.0:
        return 95.0
    if magnitude >= 1.0:
        return 85.0
    if magnitude >= 0.5:
        return 65.0
    return 40.0


def _score_volume(current_volume: int, avg_volume_20d: float) -> float:
    """Factor 2: Volume ratio vs 20-day average."""
    if avg_volume_20d <= 0:
        return 30.0
    ratio = current_volume / avg_volume_20d
    if ratio >= 3.0:
        return 95.0
    if ratio >= 2.0:
        return 80.0
    if ratio >= 1.0:
        return 55.0
    return 10.0


def _score_atr_expansion(recent_atr: float, base_atr: float) -> float:
    """Factor 3: ATR expanding = volatility increasing into breakout."""
    if base_atr <= 0:
        return 50.0
    expansion = recent_atr / base_atr
    if expansion >= 1.5:
        return 90.0
    if expansion >= 1.2:
        return 70.0
    if expansion >= 0.9:
        return 50.0
    return 20.0  # contracting


def _score_consolidation(closes: list[float], lookback: int = 10) -> float:
    """Factor 4: Tight consolidation before breakout (low range)."""
    if len(closes) < lookback:
        return 40.0
    segment = closes[-lookback - 1 : -1]
    if not segment:
        return 40.0
    rng = (max(segment) - min(segment)) / (np.mean(segment) or 1.0)
    if rng <= 0.03:
        return 95.0  # very tight base
    if rng <= 0.06:
        return 75.0
    if rng <= 0.10:
        return 50.0
    return 20.0  # wide, volatile — not a clean base


def _score_sector_rs(ticker_5d_return: float, sector_median_5d: float) -> float:
    """Factor 5: Ticker return vs sector median over 5 days."""
    rs = ticker_5d_return - sector_median_5d
    if rs >= 0.06:
        return 95.0
    if rs >= 0.03:
        return 80.0
    if rs >= 0.0:
        return 60.0
    if rs >= -0.02:
        return 40.0
    return 10.0


def scan(
    ticker: str,
    price_bars: list[dict[str, Any]],
    sector_median_5d_return: float = 0.0,
) -> Optional[dict[str, Any]]:
    """Run MomentumBreakout scan for a single ticker.

    Args:
        ticker: Asset ticker symbol.
        price_bars: List of OHLCV dicts (sorted oldest→newest).
        sector_median_5d_return: Sector median 5-day return (fraction).

    Returns:
        StrategySignal dict if score >= QUALITY_GATE, else None.
    """
    if len(price_bars) < 21:
        return None

    closes = [float(b.get("close", 0) or 0) for b in price_bars]
    highs = [float(b.get("high", 0) or 0) for b in price_bars]
    lows = [float(b.get("low", 0) or 0) for b in price_bars]
    volumes = [int(b.get("volume", 0) or 0) for b in price_bars]

    if not closes or closes[-1] <= 0:
        return None

    current_close = closes[-1]
    current_volume = volumes[-1]

    high_20d = max(highs[-21:-1])  # exclude current bar
    avg_vol_20d = float(np.mean(volumes[-21:-1])) if len(volumes) >= 21 else 0.0
    atr_14 = _atr(highs[-15:], lows[-15:], closes[-15:], 14)
    base_atr = _atr(highs[-30:-15], lows[-30:-15], closes[-30:-15], 14) if len(highs) >= 30 else atr_14

    # 5-day return
    ticker_5d = (current_close - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] > 0 else 0.0

    f1 = _score_breakout_magnitude(current_close, high_20d, atr_14)
    f2 = _score_volume(current_volume, avg_vol_20d)
    f3 = _score_atr_expansion(atr_14, base_atr)
    f4 = _score_consolidation(closes)
    f5 = _score_sector_rs(ticker_5d, sector_median_5d_return)

    score = (
        f1 * 0.35
        + f2 * 0.25
        + f3 * 0.20
        + f4 * 0.10
        + f5 * 0.10
    )

    if score < QUALITY_GATE:
        return None

    # Expected move: ATR-based
    expected_move_pct = round((atr_14 * 2.5 / current_close) * 100, 2)
    entry_zone_low = round(current_close * 1.002, 2)
    entry_zone_high = round(current_close * 1.008, 2)
    invalidation = round(high_20d - atr_14, 2)

    logger.info(
        "MomentumBreakout signal: %s score=%.1f (f1=%.0f f2=%.0f f3=%.0f f4=%.0f f5=%.0f)",
        ticker, score, f1, f2, f3, f4, f5,
    )

    return {
        "ticker": ticker,
        "strategy": STRATEGY_NAME,
        "score": round(score, 2),
        "confidence": round(min(score / 100.0, 1.0), 3),
        "expected_move_pct": expected_move_pct,
        "time_horizon_days": 10,
        "catalyst": None,
        "entry_zone_low": entry_zone_low,
        "entry_zone_high": entry_zone_high,
        "invalidation_price": invalidation,
        "binary_event": False,
        "detail": {
            "breakout_magnitude": round(f1, 1),
            "volume_ratio": round(f2, 1),
            "atr_expansion": round(f3, 1),
            "consolidation": round(f4, 1),
            "sector_rs": round(f5, 1),
            "atr_14": round(atr_14, 4),
            "high_20d": round(high_20d, 2),
        },
    }
