"""SqueezeDetector strategy scanner — L2 Signal Detection.

Identifies short squeeze setup candidates.
Score = weighted sum of 4 factors (0-100 scale).

Factors:
  1. Short interest intensity (SI%)     weight 0.30
  2. Covering pressure (days to cover)  weight 0.25
  3. DTC tightness                      weight 0.25
  4. Borrow cost                        weight 0.20

WARNING: Position sizing reduced 50% for squeeze trades (ambiguous signal).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

STRATEGY_NAME = "squeeze_detector"
QUALITY_GATE = 55.0
POSITION_SIZE_MULTIPLIER = 0.5  # CIO uses this to halve squeeze positions


def _score_si_intensity(si_pct: Optional[float]) -> float:
    if si_pct is None:
        return 20.0
    if si_pct >= 30:
        return 95.0
    if si_pct >= 20:
        return 80.0
    if si_pct >= 15:
        return 65.0
    if si_pct >= 10:
        return 45.0
    return 10.0


def _score_covering_pressure(price_bars: list[dict[str, Any]]) -> float:
    """Score based on recent price momentum (shorts underwater)."""
    if len(price_bars) < 4:
        return 30.0
    closes = [float(b.get("close", 0) or 0) for b in price_bars]
    if closes[-1] <= 0 or closes[-4] <= 0:
        return 30.0
    three_day_return = (closes[-1] - closes[-4]) / closes[-4]
    if three_day_return >= 0.10:
        return 90.0
    if three_day_return >= 0.05:
        return 70.0
    if three_day_return >= 0.02:
        return 55.0
    if three_day_return >= 0:
        return 40.0
    return 10.0  # price falling — shorts winning


def _score_dtc(days_to_cover: Optional[float]) -> float:
    if days_to_cover is None:
        return 30.0
    if days_to_cover <= 1.0:
        return 90.0
    if days_to_cover <= 2.5:
        return 75.0
    if days_to_cover <= 5.0:
        return 55.0
    if days_to_cover <= 10.0:
        return 35.0
    return 10.0


def _score_borrow_cost(borrow_rate: Optional[float]) -> float:
    """High borrow rate = painful for shorts = squeeze potential."""
    if borrow_rate is None:
        return 40.0
    if borrow_rate >= 50.0:
        return 95.0
    if borrow_rate >= 20.0:
        return 75.0
    if borrow_rate >= 10.0:
        return 55.0
    if borrow_rate >= 5.0:
        return 40.0
    return 25.0


def scan(
    ticker: str,
    price_bars: list[dict[str, Any]],
    short_interest: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Run SqueezeDetector scan for a single ticker."""
    if len(price_bars) < 5:
        return None

    si_pct = float(short_interest["si_pct"]) if short_interest and short_interest.get("si_pct") else None
    days_to_cover = float(short_interest["days_to_cover"]) if short_interest and short_interest.get("days_to_cover") else None
    borrow_rate = float(short_interest["borrow_rate"]) if short_interest and short_interest.get("borrow_rate") else None

    # Minimum SI threshold — no squeeze without significant short interest
    if si_pct is not None and si_pct < 8.0:
        return None

    # Volume confirmation: current volume > 2x 20d avg
    if len(price_bars) >= 21:
        volumes = [int(b.get("volume", 0) or 0) for b in price_bars]
        avg_vol = float(np.mean(volumes[-21:-1]))
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
        if vol_ratio < 1.5:
            return None  # No volume — not a squeeze

    f1 = _score_si_intensity(si_pct)
    f2 = _score_covering_pressure(price_bars)
    f3 = _score_dtc(days_to_cover)
    f4 = _score_borrow_cost(borrow_rate)

    score = f1 * 0.30 + f2 * 0.25 + f3 * 0.25 + f4 * 0.20

    if score < QUALITY_GATE:
        return None

    closes = [float(b.get("close", 0) or 0) for b in price_bars]
    current_close = closes[-1]

    logger.info(
        "SqueezeDetector signal: %s score=%.1f SI=%.1f%% DTC=%.1f borrow=%.1f%%",
        ticker, score, si_pct or 0, days_to_cover or 0, borrow_rate or 0,
    )

    return {
        "ticker": ticker,
        "strategy": STRATEGY_NAME,
        "score": round(score, 2),
        "confidence": round(min(score / 100.0, 1.0), 3),
        "expected_move_pct": round(min(si_pct or 15.0, 50.0), 1),
        "time_horizon_days": 3,
        "catalyst": "Short squeeze setup",
        "entry_zone_low": round(current_close * 1.005, 2),
        "entry_zone_high": round(current_close * 1.02, 2),
        "invalidation_price": round(current_close * 0.93, 2),
        "binary_event": False,
        "position_size_multiplier": POSITION_SIZE_MULTIPLIER,
        "detail": {
            "si_pct": round(si_pct, 2) if si_pct else None,
            "days_to_cover": round(days_to_cover, 2) if days_to_cover else None,
            "borrow_rate": round(borrow_rate, 2) if borrow_rate else None,
            "si_score": round(f1, 1),
            "covering_score": round(f2, 1),
            "dtc_score": round(f3, 1),
            "borrow_score": round(f4, 1),
        },
    }
