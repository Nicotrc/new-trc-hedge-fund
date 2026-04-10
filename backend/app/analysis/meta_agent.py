"""Meta-Agent — Layer 4.

Supervisory layer that evaluates agent output quality, detects bias,
and dynamically reweights agent influence before the CIO aggregates.

Analysis steps:
  1. Agreement/Disagreement Analysis
  2. Overconfidence Detection
  3. Anchoring Bias Detection
  4. Historical Accuracy Weighting (from Learning System)
  5. Regime-Adjusted Weights
  6. Dissent Value (lone correct contrarian amplification)

The meta-agent does NOT make trade decisions — it calibrates the signal.
"""

from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING, Any

from app.agents.schemas import MetaAgentReport, QuantAgentVerdict

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime-adjusted base weights (before historical accuracy adjustment)
# ---------------------------------------------------------------------------

_REGIME_BASE_WEIGHTS: dict[str, dict[str, float]] = {
    "momentum": {
        "momentum": 0.35,
        "value": 0.10,
        "event": 0.20,
        "macro": 0.20,
        "risk": 0.15,
    },
    "event": {
        "momentum": 0.10,
        "value": 0.10,
        "event": 0.40,
        "macro": 0.20,
        "risk": 0.20,
    },
    "risk_off": {
        "momentum": 0.10,
        "value": 0.20,
        "event": 0.15,
        "macro": 0.35,
        "risk": 0.20,
    },
    "fundamental": {
        "momentum": 0.10,
        "value": 0.35,
        "event": 0.15,
        "macro": 0.20,
        "risk": 0.20,
    },
    "default": {
        "momentum": 0.20,
        "value": 0.20,
        "event": 0.20,
        "macro": 0.20,
        "risk": 0.20,
    },
}


def _detect_regime_from_verdicts(verdicts: list[QuantAgentVerdict]) -> str:
    """Infer dominant regime from agent verdicts and signals.

    Uses the event agent's binary_event flag and macro agent's direction
    as regime hints.
    """
    event_v = next((v for v in verdicts if v.agent_id == "event"), None)
    macro_v = next((v for v in verdicts if v.agent_id == "macro"), None)
    momentum_v = next((v for v in verdicts if v.agent_id == "momentum"), None)

    # Binary event → event regime
    if event_v and event_v.binary_event and event_v.score >= 50:
        return "event"

    # Macro FLAT or very low score → risk-off
    if macro_v and macro_v.direction == "FLAT" and macro_v.score < 45:
        return "risk_off"

    # Strong momentum signal
    if momentum_v and momentum_v.score >= 70:
        return "momentum"

    return "default"


def _compute_direction_consensus(verdicts: list[QuantAgentVerdict]) -> float:
    """Fraction of agents agreeing on the plurality direction."""
    if not verdicts:
        return 0.0
    directions = [v.direction for v in verdicts]
    counts: dict[str, int] = {}
    for d in directions:
        counts[d] = counts.get(d, 0) + 1
    plurality_count = max(counts.values())
    return plurality_count / len(verdicts)


def _detect_overconfidence(verdicts: list[QuantAgentVerdict]) -> bool:
    """Flag when all agents have high confidence but low data sufficiency."""
    if not verdicts:
        return False
    avg_confidence = statistics.mean(v.confidence for v in verdicts)
    avg_sufficiency = statistics.mean(v.data_sufficiency for v in verdicts)
    return avg_sufficiency < 0.5 and avg_confidence > 0.8


def _detect_anchoring(verdicts: list[QuantAgentVerdict]) -> str | None:
    """Detect if 3+ agents anchor on the same catalyst/thesis in bull_factors."""
    if not verdicts:
        return None

    # Count mention frequency across all bull_factors
    mention_counts: dict[str, int] = {}
    for v in verdicts:
        # Simple keyword deduplication within an agent's factors
        seen_in_verdict: set[str] = set()
        for factor in v.bull_factors:
            # Extract key nouns/phrases (simplified: use first word token)
            key = factor.lower().split()[0] if factor.strip() else ""
            if key and key not in seen_in_verdict:
                mention_counts[key] = mention_counts.get(key, 0) + 1
                seen_in_verdict.add(key)

    # Find any keyword mentioned by 3+ different agents
    for keyword, count in mention_counts.items():
        if count >= 3:
            return f"{count} agents anchor on '{keyword}' catalyst"
    return None


def _apply_historical_accuracy(
    base_weights: dict[str, float],
    accuracy_lookup: dict[str, float] | None,
) -> dict[str, float]:
    """Bayesian update of weights based on historical hit rates.

    accuracy_lookup: {agent_id: hit_rate (0-1)} from Learning System.
    Agents with > 60% hit rate get upweighted; < 40% get downweighted.
    Minimum weight floor: 0.05.
    """
    if not accuracy_lookup:
        return base_weights

    adjusted: dict[str, float] = {}
    for agent_id, base_w in base_weights.items():
        accuracy = accuracy_lookup.get(agent_id, 0.5)
        # Multiplier: 0.4 accuracy → 0.8x, 0.6 → 1.2x, 0.5 → 1.0x
        multiplier = 0.6 + accuracy * 0.8
        adjusted[agent_id] = max(0.05, base_w * multiplier)

    # Re-normalise to sum to 1.0
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()}


def _compute_dissent_bonus(
    verdicts: list[QuantAgentVerdict],
    plurality_direction: str,
    accuracy_lookup: dict[str, float] | None,
) -> dict[str, float]:
    """Boost historically-correct dissenting agents.

    A dissenter who is historically accurate is a strong signal.
    """
    bonus: dict[str, float] = {}
    if not accuracy_lookup:
        return bonus
    for v in verdicts:
        if v.direction != plurality_direction:
            accuracy = accuracy_lookup.get(v.agent_id, 0.5)
            if accuracy > 0.60:
                bonus[v.agent_id] = round((accuracy - 0.50) * 0.20, 3)
    return bonus


def run_meta_agent(
    opportunity_id: str,
    verdicts: list[QuantAgentVerdict],
    accuracy_lookup: dict[str, float] | None = None,
) -> MetaAgentReport:
    """Analyse agent outputs, detect bias, and compute dynamic weights.

    Args:
        opportunity_id: Compound key ticker:detected_at.
        verdicts: List of 5 QuantAgentVerdict objects.
        accuracy_lookup: Optional dict {agent_id: rolling_hit_rate} from L7
            Learning System. If None, uses equal weights as baseline.

    Returns:
        MetaAgentReport with calibrated weights and bias flags.
    """
    if not verdicts:
        return MetaAgentReport(
            opportunity_id=opportunity_id,
            agent_weights={a: 0.20 for a in ["momentum", "value", "event", "macro", "risk"]},
            direction_consensus=0.0,
            score_spread=0.0,
            overconfidence_flag=False,
            risk_adjustment=1.0,
            regime="default",
        )

    # --- Step 1: Agreement analysis ---
    direction_consensus = _compute_direction_consensus(verdicts)
    scores = [v.score for v in verdicts]
    score_spread = statistics.stdev(scores) if len(scores) > 1 else 0.0

    # --- Step 2: Overconfidence ---
    overconfidence_flag = _detect_overconfidence(verdicts)

    # --- Step 3: Anchoring bias ---
    anchoring_bias = _detect_anchoring(verdicts)

    # --- Step 4 & 5: Regime + historical weights ---
    regime = _detect_regime_from_verdicts(verdicts)
    base_weights = _REGIME_BASE_WEIGHTS.get(regime, _REGIME_BASE_WEIGHTS["default"])
    agent_weights = _apply_historical_accuracy(base_weights, accuracy_lookup)

    # --- Step 6: Dissent bonus ---
    directions = [v.direction for v in verdicts]
    direction_counts: dict[str, int] = {}
    for d in directions:
        direction_counts[d] = direction_counts.get(d, 0) + 1
    plurality_direction = max(direction_counts, key=lambda k: direction_counts[k])
    dissent_bonus = _compute_dissent_bonus(verdicts, plurality_direction, accuracy_lookup)

    # Apply dissent bonus (re-normalise after)
    for agent_id, bonus in dissent_bonus.items():
        agent_weights[agent_id] = agent_weights.get(agent_id, 0.20) + bonus
    total = sum(agent_weights.values())
    agent_weights = {k: round(v / total, 4) for k, v in agent_weights.items()}

    # --- Risk adjustment multiplier ---
    risk_agent = next((v for v in verdicts if v.agent_id == "risk"), None)
    risk_adjustment = 1.0
    if risk_agent:
        # Linear scale: risk_score 100 → 1.0x, risk_score 50 → 0.75x, risk_score 25 → 0.5x
        risk_adjustment = max(0.3, risk_agent.score / 100.0)
        if overconfidence_flag:
            risk_adjustment *= 0.75
        if score_spread > 30:
            risk_adjustment *= 0.85
    risk_adjustment = round(min(1.0, risk_adjustment), 4)

    report = MetaAgentReport(
        opportunity_id=opportunity_id,
        agent_weights=agent_weights,
        direction_consensus=round(direction_consensus, 4),
        score_spread=round(score_spread, 2),
        overconfidence_flag=overconfidence_flag,
        anchoring_bias=anchoring_bias,
        dissent_bonus=dissent_bonus,
        risk_adjustment=risk_adjustment,
        regime=regime,
    )

    logger.info(
        "Meta-agent for %s: regime=%s consensus=%.2f spread=%.1f "
        "overconf=%s anchor=%s risk_adj=%.2f",
        opportunity_id,
        regime,
        direction_consensus,
        score_spread,
        overconfidence_flag,
        anchoring_bias is not None,
        risk_adjustment,
    )
    return report


def load_accuracy_from_redis(redis_client: Any, agent_ids: list[str]) -> dict[str, float]:
    """Load rolling hit rates from Redis (stored by Learning System).

    Keys: agent_accuracy:{agent_id} → float stored as string
    Returns: {agent_id: hit_rate} — defaults to 0.5 if not found.
    """
    result: dict[str, float] = {}
    for agent_id in agent_ids:
        raw = redis_client.get(f"agent_accuracy:{agent_id}")
        if raw:
            try:
                result[agent_id] = float(raw)
            except ValueError:
                result[agent_id] = 0.5
        else:
            result[agent_id] = 0.5
    return result
