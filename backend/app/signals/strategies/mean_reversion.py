"""MeanReversion strategy scanner — L2 Signal Detection.

Targets deeply oversold stocks with fundamental support.
Score = weighted sum of 4 factors (0-100 scale).

Factors:
  1. Oversold depth (RSI + z-score)  weight 0.30
  2. Z-score extremity               weight 0.25
  3. Fundamental support             weight 0.25
  4. Volatility compression          weight 0.20
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

STRATEGY_NAME = "mean_reversion"
QUALITY_GATE = 55.0


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _z_score(closes: list[float], window: int = 20) -> float:
    if len(closes) < window:
        return 0.0
    segment = closes[-window:]
    mean = float(np.mean(segment))
    std = float(np.std(segment))
    if std == 0:
        return 0.0
    return (closes[-1] - mean) / std


def _score_oversold(rsi: float) -> float:
    if rsi < 20:
        return 95.0
    if rsi < 25:
        return 85.0
    if rsi < 30:
        return 70.0
    if rsi < 35:
        return 50.0
    return 10.0


def _score_zscore(z: float) -> float:
    if z <= -2.5:
        return 95.0
    if z <= -2.0:
        return 80.0
    if z <= -1.5:
        return 60.0
    if z <= -1.0:
        return 40.0
    return 10.0


def _score_fundamental_support(
    fcf_yield: Optional[float],
    sector_avg_fcf_yield: float = 0.04,
) -> float:
    if fcf_yield is None:
        return 30.0  # unknown — neutral
    if fcf_yield < 0:
        return 10.0  # burning cash
    if fcf_yield >= sector_avg_fcf_yield * 1.5:
        return 90.0
    if fcf_yield >= sector_avg_fcf_yield:
        return 65.0
    if fcf_yield >= sector_avg_fcf_yield * 0.5:
        return 45.0
    return 25.0


def _score_vol_compression(closes: list[float]) -> float:
    """Low recent volatility vs historical = compression before expansion."""
    if len(closes) < 30:
        return 40.0
    recent_std = float(np.std(closes[-10:])) / (np.mean(closes[-10:]) or 1.0)
    base_std = float(np.std(closes[-30:-10])) / (np.mean(closes[-30:-10]) or 1.0)
    if base_std == 0:
        return 40.0
    ratio = recent_std / base_std
    if ratio <= 0.5:
        return 90.0
    if ratio <= 0.7:
        return 70.0
    if ratio <= 1.0:
        return 50.0
    return 20.0


def scan(
    ticker: str,
    price_bars: list[dict[str, Any]],
    fundamentals: Optional[dict[str, Any]] = None,
    sector_avg_fcf_yield: float = 0.04,
) -> Optional[dict[str, Any]]:
    """Run MeanReversion scan for a single ticker."""
    if len(price_bars) < 21:
        return None

    closes = [float(b.get("close", 0) or 0) for b in price_bars]
    if not closes or closes[-1] <= 0:
        return None

    current_close = closes[-1]
    rsi_14 = _rsi(closes)
    z = _z_score(closes, 20)

    # FCF yield from fundamentals (optional)
    fcf_yield: Optional[float] = None
    if fundamentals:
        fcf = fundamentals.get("free_cash_flow")
        mkt_cap = fundamentals.get("market_cap")
        if fcf and mkt_cap and mkt_cap > 0:
            try:
                fcf_yield = float(fcf) / float(mkt_cap)
            except Exception:
                pass

    f1 = _score_oversold(rsi_14)
    f2 = _score_zscore(z)
    f3 = _score_fundamental_support(fcf_yield, sector_avg_fcf_yield)
    f4 = _score_vol_compression(closes)

    score = f1 * 0.30 + f2 * 0.25 + f3 * 0.25 + f4 * 0.20

    if score < QUALITY_GATE:
        return None

    # Mean reversion target: return to 20d mean
    mean_20d = float(np.mean(closes[-20:]))
    expected_move_pct = round(((mean_20d - current_close) / current_close) * 100, 2)
    if expected_move_pct < 0:
        return None  # Not oversold vs mean

    entry = round(current_close * 1.005, 2)
    invalidation = round(current_close * 0.92, 2)  # 8% stop for MR

    logger.info("MeanReversion signal: %s score=%.1f RSI=%.1f z=%.2f", ticker, score, rsi_14, z)

    return {
        "ticker": ticker,
        "strategy": STRATEGY_NAME,
        "score": round(score, 2),
        "confidence": round(min(score / 100.0, 1.0), 3),
        "expected_move_pct": expected_move_pct,
        "time_horizon_days": 7,
        "catalyst": None,
        "entry_zone_low": entry,
        "entry_zone_high": round(current_close * 1.02, 2),
        "invalidation_price": invalidation,
        "binary_event": False,
        "detail": {
            "rsi_14": round(rsi_14, 1),
            "z_score_20d": round(z, 3),
            "mean_20d": round(mean_20d, 2),
            "fcf_yield": round(fcf_yield, 4) if fcf_yield else None,
            "oversold_score": round(f1, 1),
            "zscore_score": round(f2, 1),
            "fundamental_score": round(f3, 1),
            "compression_score": round(f4, 1),
        },
    }
