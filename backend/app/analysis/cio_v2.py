"""CIO 2.0 Probabilistic Decision Engine — Layer 5.

Replaces the deterministic CIO (conviction tiers → allocation) with a
probabilistic engine using Monte Carlo simulation and Kelly Criterion.

Pipeline:
  1. Weighted aggregation of agent scores/returns/risks
  2. Monte Carlo simulation (10,000 paths)
  3. Kelly Criterion position sizing
  4. Risk adjustment from Meta-Agent
  5. Portfolio constraints (max 10%, sector < 30%)
  6. Level generation from MC percentiles
  7. Veto check from Risk Agent
  8. Final decision BUY/SELL/MONITOR/PASS

Hard rules:
  - Risk per trade <= 2% of capital (non-overridable)
  - No single position > 10% of portfolio
  - Risk Agent veto: score < 25 AND conviction HIGH → forced PASS
  - Binary events: position <= 5% regardless of Kelly
  - Direction consensus < 0.4 → MONITOR
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

from app.agents.schemas import (
    CIODecisionV2,
    MetaAgentReport,
    MonteCarloResult,
    QuantAgentVerdict,
    Scenario,
)
from app.analysis.meta_agent import run_meta_agent
from app.analysis.monte_carlo import mc_params_from_verdicts, run_monte_carlo

logger = logging.getLogger(__name__)

# Portfolio constraints
MAX_POSITION_PCT = 10.0       # max single position as % of portfolio
MAX_BINARY_POSITION_PCT = 5.0  # max binary event position
MAX_RISK_PER_TRADE_PCT = 2.0  # max capital at risk per trade
SECTOR_CONCENTRATION_CAP = 30.0  # max sector exposure

# Decision thresholds
BUY_SCORE_THRESHOLD = 65.0
MONITOR_SCORE_THRESHOLD = 45.0
MIN_POSITION_PCT = 0.5         # don't size below 0.5%


def _weighted_aggregate(
    verdicts: list[QuantAgentVerdict],
    agent_weights: dict[str, float],
) -> dict[str, float]:
    """Compute weighted aggregation of key verdict metrics.

    Returns:
        Dict with: weighted_score, weighted_expected_return, weighted_max_loss,
        avg_time_horizon, direction_majority, binary_event.
    """
    weight_map = {v.agent_id: agent_weights.get(v.agent_id, 0.20) for v in verdicts}
    total_w = sum(weight_map.get(v.agent_id, 0.20) for v in verdicts)

    w_score = sum(v.score * weight_map[v.agent_id] for v in verdicts) / (total_w or 1.0)
    w_return = sum(v.expected_return_pct * weight_map[v.agent_id] for v in verdicts) / (total_w or 1.0)
    w_max_loss = sum(v.max_loss_pct * weight_map[v.agent_id] for v in verdicts) / (total_w or 1.0)
    avg_horizon = sum(v.time_horizon_days * weight_map[v.agent_id] for v in verdicts) / (total_w or 1.0)

    # Direction: use weighted voting
    direction_votes: dict[str, float] = {"LONG": 0.0, "SHORT": 0.0, "FLAT": 0.0}
    for v in verdicts:
        direction_votes[v.direction] = direction_votes.get(v.direction, 0.0) + weight_map[v.agent_id]
    direction_majority = max(direction_votes, key=lambda k: direction_votes[k])

    binary_event = any(v.binary_event for v in verdicts)

    return {
        "weighted_score": w_score,
        "weighted_expected_return": w_return,
        "weighted_max_loss": w_max_loss,
        "avg_time_horizon": avg_horizon,
        "direction_majority": direction_majority,
        "binary_event": binary_event,
    }


def _compute_kelly(
    prob_profit: float,
    avg_win_pct: float,
    avg_loss_pct: float,
) -> float:
    """Compute half-Kelly position size (as % of portfolio).

    Kelly fraction = (p * b - (1-p)) / b
      where b = avg_win / avg_loss (win/loss ratio)

    Returns half-Kelly capped at MAX_POSITION_PCT.
    """
    if avg_win_pct <= 0 or avg_loss_pct >= 0:
        return 0.0

    avg_win = avg_win_pct / 100.0
    avg_loss = abs(avg_loss_pct) / 100.0
    if avg_loss == 0:
        return 0.0

    b = avg_win / avg_loss  # win/loss ratio
    kelly = (prob_profit * b - (1 - prob_profit)) / b
    half_kelly = max(0.0, kelly * 0.5)
    return min(half_kelly * 100.0, MAX_POSITION_PCT)  # convert to pct


def _generate_levels(
    verdicts: list[QuantAgentVerdict],
    agent_weights: dict[str, float],
    mc_result: MonteCarloResult,
    current_price: float,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Generate entry, stop, and target levels.

    Entry = weighted average of agent entry levels.
    Stop = min(MC VaR_95 implied price, tightest agent stop).
    Target = MC p75 percentile implied price.

    Returns: (entry, stop, target)
    """
    weight_map = {v.agent_id: agent_weights.get(v.agent_id, 0.20) for v in verdicts}
    total_w = sum(weight_map.values())

    # Entry: weighted average of agent entries
    entries = [
        (v.key_levels.entry, weight_map[v.agent_id])
        for v in verdicts
        if v.key_levels.entry is not None
    ]
    if entries:
        entry = sum(e * w for e, w in entries) / sum(w for _, w in entries)
    else:
        entry = current_price

    # Stop: tightest of MC VaR_95 and agent stops
    mc_stop_price = current_price * (1 + mc_result.var_95)  # var_95 is negative
    agent_stops = [v.key_levels.stop for v in verdicts if v.key_levels.stop is not None]
    if agent_stops:
        tightest_agent_stop = max(agent_stops)  # tightest = highest stop price for long
        stop = max(mc_stop_price, tightest_agent_stop)  # use the tighter (higher) stop
    else:
        stop = mc_stop_price

    # Target: MC p75 percentile return applied to entry
    target = entry * (1 + mc_result.percentiles.get("p75", 0.15))

    return (
        round(entry, 2) if entry else None,
        round(stop, 2) if stop else None,
        round(target, 2) if target else None,
    )


def _build_scenarios(
    verdicts: list[QuantAgentVerdict],
    mc_result: MonteCarloResult,
    current_price: float,
) -> list[Scenario]:
    """Build named scenarios from MC percentiles."""
    scenarios = []

    # Bull case (p95)
    p95_ret = mc_result.percentiles.get("p95", 0.40)
    bull_drivers = []
    for v in verdicts:
        if v.direction == "LONG" and v.bull_factors:
            bull_drivers.extend(v.bull_factors[:1])
    scenarios.append(Scenario(
        name="Bull case",
        probability=0.05,  # 5th percentile from top
        target_price=round(current_price * (1 + p95_ret), 2),
        return_pct=round(p95_ret * 100, 1),
        narrative="Upside surprise — all catalysts materialise",
        key_driver=bull_drivers[0] if bull_drivers else "Momentum continuation",
    ))

    # Base case (p50)
    p50_ret = mc_result.percentiles.get("p50", 0.10)
    scenarios.append(Scenario(
        name="Base case",
        probability=0.50,
        target_price=round(current_price * (1 + p50_ret), 2),
        return_pct=round(p50_ret * 100, 1),
        narrative="Expected outcome — thesis plays out as modelled",
        key_driver="Weighted agent consensus",
    ))

    # Bear case (p5)
    p5_ret = mc_result.percentiles.get("p5", -0.20)
    bear_drivers = []
    for v in verdicts:
        if v.agent_id == "risk" and v.bear_factors:
            bear_drivers.extend(v.bear_factors[:1])
    scenarios.append(Scenario(
        name="Bear case",
        probability=0.05,
        target_price=round(current_price * (1 + p5_ret), 2),
        return_pct=round(p5_ret * 100, 1),
        narrative="Downside risk — key risks materialise",
        key_driver=bear_drivers[0] if bear_drivers else "Risk factor cascade",
    ))

    return scenarios


def make_cio_decision_v2(
    opportunity_id: str,
    ticker: str,
    verdicts: list[QuantAgentVerdict],
    current_price: float,
    realized_vol_20d: float = 0.40,
    accuracy_lookup: Optional[dict[str, float]] = None,
    portfolio_context: Optional[dict[str, Any]] = None,
) -> CIODecisionV2:
    """Produce a probabilistic CIO decision from quant agent verdicts.

    Args:
        opportunity_id: Compound key ticker:detected_at.
        ticker: Asset ticker symbol.
        verdicts: List of 5 QuantAgentVerdict objects from quant agents.
        current_price: Current market price (for MC simulation).
        realized_vol_20d: 20-day realized volatility (annualized fraction, e.g. 0.40).
        accuracy_lookup: Optional {agent_id: hit_rate} from Learning System.
        portfolio_context: Optional dict with current portfolio state for
            concentration checks.

    Returns:
        CIODecisionV2 with full MC distribution, Kelly sizing, and levels.
    """
    if not verdicts:
        logger.error("make_cio_decision_v2: no verdicts for %s", opportunity_id)
        raise ValueError(f"No verdicts provided for {opportunity_id}")

    # ------------------------------------------------------------------
    # Step 0: Check for Risk Agent veto before anything else
    # ------------------------------------------------------------------
    risk_verdict = next((v for v in verdicts if v.agent_id == "risk"), None)
    veto_triggered = False
    veto_reason: Optional[str] = None

    if risk_verdict and risk_verdict.score < 25 and risk_verdict.conviction == "HIGH":
        veto_triggered = True
        veto_reason = (
            risk_verdict.bear_factors[0]
            if risk_verdict.bear_factors
            else f"Risk score {risk_verdict.score} below veto threshold"
        )
        logger.warning(
            "RISK VETO: opportunity=%s ticker=%s reason=%s",
            opportunity_id, ticker, veto_reason,
        )

    # ------------------------------------------------------------------
    # Step 1: Meta-Agent analysis (bias detection + dynamic weights)
    # ------------------------------------------------------------------
    meta_report = run_meta_agent(opportunity_id, verdicts, accuracy_lookup)

    # ------------------------------------------------------------------
    # Step 2: Weighted aggregation
    # ------------------------------------------------------------------
    agg = _weighted_aggregate(verdicts, meta_report.agent_weights)
    weighted_score = agg["weighted_score"]
    binary_event = agg["binary_event"]

    # ------------------------------------------------------------------
    # Step 3: Monte Carlo simulation
    # ------------------------------------------------------------------
    mc_params = mc_params_from_verdicts(
        verdicts=verdicts,
        agent_weights=meta_report.agent_weights,
        current_price=current_price,
        realized_vol_20d=realized_vol_20d,
    )
    mc_result = run_monte_carlo(**mc_params)

    # ------------------------------------------------------------------
    # Step 4: Kelly Criterion position sizing
    # ------------------------------------------------------------------
    kelly_size = _compute_kelly(
        prob_profit=mc_result.prob_profit,
        avg_win_pct=mc_result.avg_profit_pct,
        avg_loss_pct=mc_result.avg_loss_pct,
    )

    # ------------------------------------------------------------------
    # Step 5: Risk adjustment + portfolio constraints
    # ------------------------------------------------------------------
    position_size = kelly_size * meta_report.risk_adjustment

    # Binary event cap
    if binary_event:
        position_size = min(position_size, MAX_BINARY_POSITION_PCT)

    # Hard cap
    position_size = min(position_size, MAX_POSITION_PCT)

    # Risk-per-trade constraint (2% max capital at risk)
    if risk_verdict and risk_verdict.max_loss_pct > 0 and position_size > 0:
        max_risk_pct = risk_verdict.max_loss_pct / 100.0
        if max_risk_pct > 0:
            max_position_by_risk = MAX_RISK_PER_TRADE_PCT / max_risk_pct
            position_size = min(position_size, max_position_by_risk)

    position_size = round(max(0.0, position_size), 2)

    # ------------------------------------------------------------------
    # Step 6: Level generation
    # ------------------------------------------------------------------
    entry, stop, target = _generate_levels(
        verdicts, meta_report.agent_weights, mc_result, current_price
    )

    # Risk/reward ratio
    if entry and stop and target and stop < entry:
        rr = (target - entry) / (entry - stop) if (entry - stop) > 0 else 0.0
    else:
        rr = mc_result.avg_profit_pct / abs(mc_result.avg_loss_pct) if mc_result.avg_loss_pct != 0 else 0.0

    # ------------------------------------------------------------------
    # Step 7: Final decision logic
    # ------------------------------------------------------------------
    if veto_triggered:
        decision = "PASS"
        position_size = 0.0
    elif meta_report.direction_consensus < 0.4:
        decision = "MONITOR"
        position_size = 0.0
    elif weighted_score >= BUY_SCORE_THRESHOLD and mc_result.prob_profit > 0.5 and position_size >= MIN_POSITION_PCT:
        if agg["direction_majority"] == "LONG":
            decision = "BUY"
        elif agg["direction_majority"] == "SHORT":
            decision = "SELL"
        else:
            decision = "MONITOR"
            position_size = 0.0
    elif weighted_score >= MONITOR_SCORE_THRESHOLD and mc_result.prob_profit > 0.4:
        decision = "MONITOR"
        position_size = 0.0
    else:
        decision = "PASS"
        position_size = 0.0

    # Conflict resolution: score spread > 30 → cap at MONITOR
    if meta_report.score_spread > 30 and decision == "BUY":
        decision = "MONITOR"
        position_size = 0.0

    # Overconfidence flag → half position
    if meta_report.overconfidence_flag and decision in ("BUY", "SELL"):
        position_size = round(position_size * 0.5, 2)

    # ------------------------------------------------------------------
    # Step 8: Build scenarios
    # ------------------------------------------------------------------
    scenarios = _build_scenarios(verdicts, mc_result, current_price)

    cio_decision = CIODecisionV2(
        opportunity_id=opportunity_id,
        ticker=ticker,
        decision=decision,  # type: ignore[arg-type]
        weighted_score=round(weighted_score, 2),
        position_size_pct=position_size,
        entry_price=entry,
        stop_loss=stop if not binary_event else None,
        take_profit=target,
        risk_reward_ratio=round(rr, 2),
        kelly_fraction=round(kelly_size / 100.0, 4),
        monte_carlo=mc_result,
        scenarios=scenarios,
        agent_weights_used=meta_report.agent_weights,
        veto_triggered=veto_triggered,
        veto_reason=veto_reason,
        risk_agent_score=risk_verdict.score if risk_verdict else None,
        meta_risk_adjustment=meta_report.risk_adjustment,
    )

    logger.info(
        "CIO v2 decision for %s: decision=%s score=%.1f size=%.2f%% "
        "P(profit)=%.2f veto=%s binary=%s",
        opportunity_id,
        decision,
        weighted_score,
        position_size,
        mc_result.prob_profit,
        veto_triggered,
        binary_event,
    )
    return cio_decision
