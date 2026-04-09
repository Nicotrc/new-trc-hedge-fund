"""EventDrivenBiotech strategy scanner — L2 Signal Detection.

Highest priority scanner for biotech catalyst opportunities.
Score = weighted sum of 5 factors (0-100 scale).

Factors:
  1. Catalyst proximity (days to event)    weight 0.25
  2. IV cheapness (IV rank)               weight 0.20
  3. Historical success rate              weight 0.20
  4. Asymmetric R/R (binary EV)           weight 0.20
  5. Setup quality (unusual activity)     weight 0.15

Hard rule: NO catalyst within 45 days → score = 0 → not emitted.

Sets binary_event = True → MC uses jump-diffusion model.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

STRATEGY_NAME = "event_driven_bio"
QUALITY_GATE = 50.0

# Historical base rates by event type
_BASE_RATES: dict[str, dict[str, float]] = {
    "fda_pdufa":      {"p": 0.85, "upside": 0.27, "downside": -0.50},
    "phase3_readout": {"p": 0.58, "upside": 0.55, "downside": -0.35},
    "phase2_readout": {"p": 0.29, "upside": 0.90, "downside": -0.40},
    "phase1_readout": {"p": 0.52, "upside": 0.45, "downside": -0.27},
    "fda_adcom":      {"p": 0.68, "upside": 0.35, "downside": -0.30},
    "earnings":       {"p": 0.62, "upside": 0.10, "downside": -0.14},
    "conference":     {"p": 0.55, "upside": 0.20, "downside": -0.15},
}


def _days_until(event_date: datetime) -> int:
    today = date.today()
    event_d = event_date.astimezone(timezone.utc).date() if event_date.tzinfo else event_date.date()
    return max(0, (event_d - today).days)


def _score_proximity(days: int) -> float:
    if days <= 3:
        return 95.0
    if days <= 7:
        return 85.0
    if days <= 15:
        return 70.0
    if days <= 30:
        return 50.0
    if days <= 45:
        return 25.0
    return 0.0  # No catalyst → hard zero


def _score_iv(iv_rank: Optional[float]) -> float:
    if iv_rank is None:
        return 50.0
    if iv_rank < 20:
        return 90.0
    if iv_rank < 40:
        return 70.0
    if iv_rank < 60:
        return 50.0
    if iv_rank < 80:
        return 30.0
    return 10.0


def _score_historical_rate(p_success: float) -> float:
    if p_success > 0.70:
        return 85.0
    if p_success > 0.55:
        return 65.0
    if p_success > 0.40:
        return 45.0
    if p_success > 0.25:
        return 30.0
    return 10.0


def _compute_ev(p_success: float, upside: float, downside: float) -> float:
    return p_success * upside + (1 - p_success) * downside


def _score_ev(ev: float, upside: float, downside: float) -> float:
    if ev < -0.05:
        return 0.0  # Negative EV → hard zero
    if ev > 0.25:
        score = 95.0
    elif ev > 0.15:
        score = 85.0
    elif ev > 0.05:
        score = 65.0
    else:
        score = 50.0
    # Asymmetry bonus
    if upside > abs(downside) * 3:
        score = min(100.0, score + 10.0)
    return score


def _score_setup(iv_rank: Optional[float], pc_ratio: Optional[float]) -> float:
    score = 50.0
    if pc_ratio is not None and pc_ratio < 0.5:
        score += 20.0  # call-heavy unusual activity
    if pc_ratio is not None and pc_ratio < 0.3:
        score += 15.0  # very call-heavy
    if iv_rank is not None and iv_rank < 30:
        score += 10.0
    return min(100.0, score)


def scan(
    ticker: str,
    events: list[dict[str, Any]],
    options_flow: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Run EventDrivenBiotech scan for a single ticker.

    Args:
        ticker: Asset ticker.
        events: List of event_calendar dicts for this ticker.
        options_flow: Latest options flow dict for this ticker.

    Returns:
        StrategySignal dict if score >= QUALITY_GATE, else None.
    """
    # Find nearest upcoming binary event
    upcoming = [
        e for e in events
        if e.get("binary_outcome") and e.get("event_date")
        and _days_until(e["event_date"]) <= 45
    ]
    if not upcoming:
        return None

    upcoming.sort(key=lambda e: _days_until(e["event_date"]))
    nearest_event = upcoming[0]
    days = _days_until(nearest_event["event_date"])
    event_type = nearest_event.get("event_type", "conference")

    f1 = _score_proximity(days)
    if f1 == 0.0:
        return None

    # Base rate
    rates = _BASE_RATES.get(event_type, _BASE_RATES["conference"])
    p_success = rates["p"]
    upside = rates["upside"]
    downside = rates["downside"]

    iv_rank = float(options_flow.get("iv_rank", 50)) if options_flow else None
    pc_ratio = float(options_flow.get("put_call_ratio", 1.0)) if options_flow else None

    ev = _compute_ev(p_success, upside, downside)

    f2 = _score_iv(iv_rank)
    f3 = _score_historical_rate(p_success)
    f4 = _score_ev(ev, upside, downside)
    f5 = _score_setup(iv_rank, pc_ratio)

    # Hard rule: negative EV → no trade
    if ev < -0.05:
        logger.debug("EventDrivenBio: %s EV=%.2f < -5%% — skipping", ticker, ev)
        return None

    score = f1 * 0.25 + f2 * 0.20 + f3 * 0.20 + f4 * 0.20 + f5 * 0.15

    if score < QUALITY_GATE:
        return None

    event_date_str = nearest_event["event_date"].isoformat()

    logger.info(
        "EventDrivenBio signal: %s score=%.1f event=%s days=%d EV=%.1f%%",
        ticker, score, event_type, days, ev * 100,
    )

    return {
        "ticker": ticker,
        "strategy": STRATEGY_NAME,
        "score": round(score, 2),
        "confidence": round(min(score / 100.0, 1.0), 3),
        "expected_move_pct": round(ev * 100, 2),
        "time_horizon_days": max(days, 1),
        "catalyst": nearest_event.get("description", event_type),
        "catalyst_date": event_date_str,
        "entry_zone_low": None,
        "entry_zone_high": None,
        "invalidation_price": None,  # binary events can't be stop-lossed
        "binary_event": True,
        "detail": {
            "event_type": event_type,
            "days_to_event": days,
            "p_success": round(p_success, 3),
            "upside_move": round(upside * 100, 1),
            "downside_move": round(downside * 100, 1),
            "binary_ev": round(ev * 100, 2),
            "iv_rank": round(iv_rank, 1) if iv_rank else None,
            "put_call_ratio": round(pc_ratio, 3) if pc_ratio else None,
        },
    }
