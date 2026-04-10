"""Opportunity Ranker — L2 Signal Detection final stage.

Merges multi-strategy signals for the same ticker (confluence bonus),
applies regime-adaptive weights, and ranks all opportunities cross-ticker.

Output: ranked list of opportunities ready for the opportunity queue.

Ranking formula:
  final_rank = 0.30 * best_signal_score
             + 0.25 * (expected_move / risk_ratio)
             + 0.20 * catalyst_proximity_factor
             + 0.15 * regime_fit
             + 0.10 * novelty_factor

Quality gate V2: final_rank >= 55 (0-100 scale).
Confluence bonus: same ticker in multiple strategies → +15 points.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

QUALITY_GATE_V2 = 55.0
CONFLUENCE_BONUS = 15.0

# Regime-adaptive strategy weights
_REGIME_STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    "momentum": {
        "momentum_breakout": 0.35,
        "event_driven_bio": 0.20,
        "squeeze_detector": 0.25,
        "mean_reversion": 0.10,
        "sector_rotation": 0.10,
    },
    "event": {
        "momentum_breakout": 0.15,
        "event_driven_bio": 0.40,
        "squeeze_detector": 0.15,
        "mean_reversion": 0.15,
        "sector_rotation": 0.15,
    },
    "risk_off": {
        "momentum_breakout": 0.10,
        "event_driven_bio": 0.20,
        "squeeze_detector": 0.10,
        "mean_reversion": 0.25,
        "sector_rotation": 0.35,
    },
    "default": {
        "momentum_breakout": 0.25,
        "event_driven_bio": 0.25,
        "squeeze_detector": 0.20,
        "mean_reversion": 0.15,
        "sector_rotation": 0.15,
    },
}


def _infer_regime(macro_snapshot: Optional[dict[str, Any]]) -> str:
    """Determine market regime from macro data snapshot."""
    if not macro_snapshot:
        return "default"
    vix = float(macro_snapshot.get("VIX", 20.0))
    if vix > 28:
        return "risk_off"
    # Event regime: no reliable macro signal, use default
    return "momentum" if vix < 16 else "default"


def _catalyst_proximity_factor(signals: list[dict[str, Any]]) -> float:
    """Factor for how soon a catalyst is (0-1 scale)."""
    for sig in signals:
        if sig.get("binary_event"):
            days = sig.get("detail", {}).get("days_to_event", 30)
            if days <= 7:
                return 1.0
            if days <= 15:
                return 0.75
            if days <= 30:
                return 0.50
            return 0.25
    return 0.0


def _regime_fit_factor(
    signals: list[dict[str, Any]],
    regime: str,
    weights: dict[str, float],
) -> float:
    """Average regime fit weight for this ticker's strategies."""
    if not signals:
        return 0.5
    strategy_weights = [weights.get(s["strategy"], 0.20) for s in signals]
    return min(1.0, sum(strategy_weights) / len(strategy_weights) / 0.35)


def _novelty_factor(ticker: str, seen_tickers_24h: set[str]) -> float:
    """Penalise re-emitting the same ticker within 24 hours."""
    return 0.3 if ticker in seen_tickers_24h else 1.0


def rank_opportunities(
    all_signals: list[dict[str, Any]],
    macro_snapshot: Optional[dict[str, Any]] = None,
    seen_tickers_24h: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Rank all strategy signals cross-ticker and apply quality gate.

    Args:
        all_signals: List of raw strategy signal dicts from all scanners.
        macro_snapshot: Current macro indicator snapshot for regime detection.
        seen_tickers_24h: Set of tickers already queued in last 24 hours.

    Returns:
        Sorted list of opportunity dicts (highest rank first),
        filtered to final_rank >= QUALITY_GATE_V2.
    """
    if not all_signals:
        return []

    seen_tickers_24h = seen_tickers_24h or set()
    regime = _infer_regime(macro_snapshot)
    strategy_weights = _REGIME_STRATEGY_WEIGHTS.get(regime, _REGIME_STRATEGY_WEIGHTS["default"])

    # Group signals by ticker
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for sig in all_signals:
        ticker = sig["ticker"]
        by_ticker.setdefault(ticker, []).append(sig)

    ranked: list[dict[str, Any]] = []

    for ticker, signals in by_ticker.items():
        # Best signal score across strategies
        best_score = max(s["score"] for s in signals)

        # Confluence: multiple strategies → +15 bonus
        confluence_bonus = CONFLUENCE_BONUS if len(signals) > 1 else 0.0

        # Best expected move / risk ratio
        best_ev = max(s.get("expected_move_pct", 0.0) for s in signals)
        ev_factor = min(100.0, best_ev * 2.0)  # scale to 0-100

        # Catalyst proximity
        cat_factor = _catalyst_proximity_factor(signals) * 100.0

        # Regime fit
        regime_factor = _regime_fit_factor(signals, regime, strategy_weights) * 100.0

        # Novelty
        novelty = _novelty_factor(ticker, seen_tickers_24h) * 100.0

        final_rank = (
            0.30 * (best_score + confluence_bonus)
            + 0.25 * ev_factor
            + 0.20 * cat_factor
            + 0.15 * regime_factor
            + 0.10 * novelty
        )

        if final_rank < QUALITY_GATE_V2:
            continue

        # Merge all signals into one opportunity
        best_signal = max(signals, key=lambda s: s["score"])
        opportunity = {
            "ticker": ticker,
            "strategy": best_signal["strategy"],
            "all_strategies": [s["strategy"] for s in signals],
            "final_rank": round(final_rank, 2),
            "best_signal_score": round(best_score, 2),
            "confluence": len(signals) > 1,
            "confluence_bonus": confluence_bonus,
            "expected_move_pct": round(best_ev, 2),
            "binary_event": any(s.get("binary_event") for s in signals),
            "catalyst_date": next(
                (s.get("catalyst_date") for s in signals if s.get("catalyst_date")),
                None,
            ),
            "catalyst": best_signal.get("catalyst"),
            "entry_zone_low": best_signal.get("entry_zone_low"),
            "entry_zone_high": best_signal.get("entry_zone_high"),
            "invalidation_price": best_signal.get("invalidation_price"),
            "signals": signals,
            "regime": regime,
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "composite_score": round(final_rank / 100.0, 4),  # v1 compat
        }
        ranked.append(opportunity)
        logger.info(
            "Ranked opportunity: %s rank=%.1f score=%.1f confluence=%s binary=%s",
            ticker,
            final_rank,
            best_score,
            len(signals) > 1,
            opportunity["binary_event"],
        )

    ranked.sort(key=lambda x: x["final_rank"], reverse=True)
    return ranked
