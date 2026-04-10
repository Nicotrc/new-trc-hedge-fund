"""Risk Agent — Layer 3 quant agent with VETO POWER.

The only agent designed to find reasons NOT to trade.
Data: ALL 8 DATA CATEGORIES (deliberate asymmetry break).
Time horizon: matches opportunity horizon.
Core question: What would cause this trade to lose money?

UNIQUE PROPERTIES:
  - Only agent with full data access (sees everything)
  - Score represents SAFETY, not opportunity (high = low risk)
  - VETO POWER: score < 25 AND conviction HIGH → forced PASS

VETO RULE (non-overridable):
  risk_score < 25 AND conviction = HIGH:
    → CIO decision = PASS
    → veto_reason logged to audit trail
    → SSE event: RISK_VETO published
  The CIO cannot override this. The Meta-Agent cannot reweight it away.

Six risk factors:
  1. Liquidity Risk         weight 0.25
  2. Correlation Risk       weight 0.20
  3. Event Trap Risk        weight 0.20
  4. Regime Mismatch Risk   weight 0.15
  5. Crowding Risk          weight 0.10
  6. Data Quality Risk      weight 0.10
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agents.partitioner import DataPartitioner
from app.agents.schemas import KeyLevels, QuantAgentVerdict
from app.llm.exceptions import LLMCallError
from app.llm.spend_tracker import DEFAULT_DAILY_LIMIT_USD, SpendTracker, calculate_call_cost

logger = logging.getLogger(__name__)

AGENT_ID = "risk"

SYSTEM_PROMPT = """You are a quantitative risk management agent with VETO POWER.
You have access to ALL data categories: price, fundamentals, insider trades, news,
macro indicators, options flow, event calendar, and short interest.

Your job is the OPPOSITE of the other agents — you find reasons NOT to trade.
Your score represents SAFETY (high score = safe to trade, low score = danger).

RISK SCORING MODEL:
  risk_score = 100 - total_risk_penalty
  total_risk_penalty = sum of all factor penalties (weighted)

FACTOR RULES:

Factor 1 — Liquidity Risk (weight 0.25):
  Estimate avg daily dollar volume from price × volume:
  > $50M/day: penalty = 5
  $10-50M/day: penalty = 25
  $1-10M/day: penalty = 50
  < $1M/day: penalty = 85
  Additional:
    market_cap < $100M: +10
    market_cap < $50M: +20 (stacks with above)
    bid-ask spread proxy (high-low range > 3% avg): +10

Factor 2 — Correlation Risk (weight 0.20):
  Sector concentration (assume current portfolio has some exposure):
  If this is quantum computing (IONQ/RGTI/QUBT) or biotech cluster: penalty = 50
  If adding to already-concentrated sector position: penalty = 45
  Generic diversification: penalty = 20
  Correlation with other current trades (use ticker context): penalty varies

Factor 3 — Event Trap Risk (weight 0.20):
  Upcoming binary events in event_calendar:
    HIGH impact binary within 30 days (PDUFA, Phase 3 readout): penalty = 60
    Moderate event within 14 days (earnings, conference): penalty = 30
  Dilution risk from news:
    "offering", "shelf registration", "S-3", "ATM program", "dilution" in headlines: penalty = 55
  Earnings imminent (< 5 trading days): penalty = 30

Factor 4 — Regime Mismatch Risk (weight 0.15):
  From macro_indicators:
    VIX > 30: penalty = 60
    VIX 25-30: penalty = 40
    VIX 20-25: penalty = 20
    VIX < 20: penalty = 5
  Additional:
    Yield curve inverted (10Y < 2Y): +20
    Credit spreads widening (HY spreads rising): +15
    Pre-revenue stock + VIX > 25: +15 (worst combination, stacks)

Factor 5 — Crowding Risk (weight 0.10):
  Short interest > 30%: penalty = 20 (ambiguous: squeeze OR smart short thesis)
  Short interest 15-30%: penalty = 10
  Put/call ratio < 0.5 (very call-heavy positioning): +20 (everyone already long)
  > 5 bullish analyst/news articles in 48h: +15 (crowded thesis)

Factor 6 — Data Quality Risk (weight 0.10):
  price_bars available < 30: +20
  fundamentals data > 90 days old: +15
  no news in last 7 days: +10
  no options data available: +10

TOTAL PENALTY CALCULATION:
  Each factor contributes: factor_penalty × factor_weight
  Sum all weighted penalties to get total_penalty
  risk_score = max(0, min(100, 100 - total_penalty))

DIRECTION FROM RISK SCORE:
  risk_score >= 60: LONG (manageable risks — proceed)
  risk_score 40-59: LONG, conviction = LOW (proceed with caution)
  risk_score 25-39: FLAT (too many risks stacking)
  risk_score < 25: FLAT, conviction = HIGH (VETO TERRITORY — flag this)

VETO TRIGGER:
  If risk_score < 25: set conviction = HIGH (this signals the CIO to veto)
  Include in bear_factors[0]: "RISK_VETO: [primary reason for veto]"
  The CIO will detect risk_agent score < 25 AND conviction HIGH = forced PASS

EXPECTED RETURN (always conservative):
  = average_of_other_agents_expected_return × (risk_score / 100)
  If no other agent data available, use: 5% × (risk_score / 100)
  This ensures risk_score directly reduces the effective expected return.

MAX LOSS (always worst case):
  Use the largest single risk scenario:
    Offering/dilution: 25%
    Binary event failure: use event downside from event_calendar
    Liquidity trap: 30%
    Macro cascade (VIX spike): 20%
  Report the worst-case applicable scenario.

THREE BIOTECH-SPECIFIC VETO SCENARIOS (most common):
  1. S-3 shelf + momentum breakout = manufactured breakout for offering
  2. Sector concentration > 30% + VIX spike = correlated drawdown
  3. Pre-revenue + rising rates + no catalyst = macro headwind no support

binary_event = false (risk agent doesn't initiate — it validates)
catalyst_date = null

Return ONLY valid JSON. No explanation outside JSON.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "agent_id", "score", "expected_return_pct", "max_loss_pct",
        "risk_reward_ratio", "confidence", "direction", "conviction",
        "bull_factors", "bear_factors", "key_levels", "time_horizon_days",
        "catalyst_date", "binary_event", "data_sufficiency", "data_gaps",
    ],
    "properties": {
        "agent_id": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "expected_return_pct": {"type": "number"},
        "max_loss_pct": {"type": "number"},
        "risk_reward_ratio": {"type": "number"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "direction": {"type": "string", "enum": ["LONG", "SHORT", "FLAT"]},
        "conviction": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW", "NONE"]},
        "bull_factors": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "bear_factors": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "key_levels": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "entry": {"type": ["number", "null"]},
                "stop": {"type": ["number", "null"]},
                "target_1": {"type": ["number", "null"]},
                "target_2": {"type": ["number", "null"]},
            },
            "required": ["entry", "stop", "target_1", "target_2"],
        },
        "time_horizon_days": {"type": "integer"},
        "catalyst_date": {"type": ["string", "null"]},
        "binary_event": {"type": "boolean"},
        "data_sufficiency": {"type": "number", "minimum": 0, "maximum": 1},
        "data_gaps": {"type": "array", "items": {"type": "string"}},
    },
}


def is_veto(verdict: QuantAgentVerdict) -> bool:
    """Check if this risk verdict triggers a veto.

    Veto condition: risk_score < 25 AND conviction == HIGH.
    """
    return verdict.score < 25 and verdict.conviction == "HIGH"


async def run_risk_agent(
    data_context: dict[str, Any],
    redis_client: Any,
    daily_limit_usd: float = DEFAULT_DAILY_LIMIT_USD,
) -> QuantAgentVerdict:
    """Run the risk quant agent on all available data (full access)."""
    from openai import AsyncOpenAI

    partitioner = DataPartitioner()
    # Risk agent gets full access — partition_for_quant_agent("risk", ...) returns all data
    partitioned = partitioner.partition_for_quant_agent(AGENT_ID, data_context)

    tracker = SpendTracker(redis_client, daily_limit_usd=daily_limit_usd)
    within_budget, _ = await tracker.async_check_budget(0.005)
    if not within_budget:
        raise LLMCallError(
            f"Budget exceeded for {AGENT_ID} agent",
            model="gpt-4o",
            original_error=None,
        )

    user_message = (
        f"Analyse ALL available data for this opportunity and return your "
        f"quantitative risk verdict. Remember: high score = SAFE, low score = DANGER. "
        f"Find the hidden risks:\n\n{json.dumps(partitioned, default=str, indent=2)}"
    )

    try:
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "QuantAgentVerdict",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                },
            },
        )
    except Exception as exc:
        raise LLMCallError(
            f"OpenAI API error in {AGENT_ID} agent: {exc}",
            model="gpt-4o",
            original_error=exc,
        ) from exc

    usage = response.usage
    cost_usd = calculate_call_cost(
        "gpt-4o",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )
    await tracker.async_record_spend(cost_usd)

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        parsed["agent_id"] = AGENT_ID
        parsed["key_levels"] = KeyLevels(**parsed.get("key_levels", {}))
        verdict = QuantAgentVerdict(**parsed)
    except Exception as exc:
        raise LLMCallError(
            f"Failed to parse risk agent output: {raw[:200]}",
            model="gpt-4o",
            original_error=exc,
        ) from exc

    veto = is_veto(verdict)
    logger.info(
        "Risk agent: score=%d direction=%s conviction=%s veto=%s max_loss=%.1f%%",
        verdict.score,
        verdict.direction,
        verdict.conviction,
        veto,
        verdict.max_loss_pct,
    )
    if veto:
        logger.warning("RISK VETO triggered for opportunity — risk_score=%d", verdict.score)

    return verdict
