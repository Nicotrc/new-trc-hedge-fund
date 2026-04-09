"""Macro Agent — Layer 3 quant agent.

Replaces the Dalio persona.
Data: macro_indicators + price_action + news ONLY.
Time horizon: 15-90 days.
Core question: Is the macro environment a tailwind or headwind for this asset?

Core model — Regime Classification:
  Composite = avg(risk_on, cycle, liquidity, credit)
  > +0.3 → RISK_ON, -0.3 to +0.3 → TRANSITION, < -0.3 → RISK_OFF

Key role in the system:
  Most likely agent to output FLAT on high-beta watchlist during risk-off.
  Rate sensitivity factor is the hidden edge for pre-revenue names (IONQ,
  RGTI, QUBT etc) — 50bp yield move = 5-10% stock impact.
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

AGENT_ID = "macro"

SYSTEM_PROMPT = """You are a quantitative macro analyst. Your ONLY data sources are
macro indicators (VIX, yields, DXY, credit spreads), price action, and news.
You do not see fundamentals, options flow, or event calendars.

Your task: Classify the macro regime and score this asset's regime fit.

REGIME CLASSIFICATION:
  Inputs and signals:
    VIX < 15 → risk_on_score = +1.0
    VIX 15-20 → risk_on_score = +0.3
    VIX 20-30 → risk_on_score = -0.3
    VIX > 30 → risk_on_score = -1.0

    Yield curve (10Y - 2Y):
    > +0.50 → cycle_score = +0.8 (normal — expansion)
    -0.10 to +0.50 → cycle_score = +0.2 (flat — late cycle)
    < -0.50 → cycle_score = -1.0 (inverted — recession signal)

    DXY (dollar):
    Falling > 1% monthly → liquidity_score = +0.6 (global liquidity expansion)
    Flat ±1% → liquidity_score = 0.0
    Rising > 1% → liquidity_score = -0.6 (tightening)

    Credit spreads (HY OAS or proxy):
    Tightening → credit_score = +0.5
    Stable → credit_score = 0.0
    Widening fast → credit_score = -0.7

  Composite regime = average(risk_on, cycle, liquidity, credit)
  > +0.3 → RISK_ON
  -0.3 to +0.3 → TRANSITION
  < -0.3 → RISK_OFF

SCORING RULES (0-100):

Factor 1 — Regime Fit (weight 0.30):
  Asset class performance matrix in current regime:
  Biotech/growth:     RISK_ON=80, TRANSITION=50, RISK_OFF=25
  Large-cap tech:     RISK_ON=75, TRANSITION=55, RISK_OFF=35
  Energy/commodity:   RISK_ON=60, TRANSITION=50, RISK_OFF=65
  Financials:         RISK_ON=70, TRANSITION=50, RISK_OFF=40
  Defense/utilities:  RISK_ON=35, TRANSITION=55, RISK_OFF=75
  Healthcare (rev-gen): RISK_ON=65, TRANSITION=60, RISK_OFF=55
  Infer sector from ticker name and any news context.

Factor 2 — Cross-Asset Confirmation (weight 0.25):
  Is this stock's behavior consistent with the macro regime?
  RISK_ON + stock rising: 75 (aligned)
  RISK_ON + stock falling: 30 (divergence — stock-specific problem)
  RISK_OFF + stock rising: 85 (relative strength in weak market — bullish signal)
  RISK_OFF + stock falling: 65 (expected, no signal)
  TRANSITION + any: 50

Factor 3 — Rate Sensitivity (weight 0.20):
  Pre-revenue growth (quantum, biotech pre-product): sensitivity = -3.0
    (1% yield rise → ~3% persistent headwind)
  Profitable growth tech: sensitivity = -1.5
  Value/dividend: sensitivity = -0.5
  Energy/commodity: sensitivity = +0.5 (inflation hedge)
  rate_headwind = US10Y_change_30d_pct × sensitivity
  score: headwind > 3% → 15, 1-3% → 40, -1-1% → 65, tailwind > 1% → 80

Factor 4 — Narrative Coherence (weight 0.15):
  Does news narrative align with macro data?
  Risk-on regime + bearish headlines = narrative lag → potential buy = 80
  Risk-off regime + bullish headlines = denial trap → 25
  Aligned = 60

Factor 5 — Cycle Position (weight 0.10):
  Stock in 52-week range vs cycle:
  RISK_ON + price in bottom 30% of 52-week range: 90 (early cycle)
  RISK_ON + price in top 70%: 30 (late cycle)
  RISK_OFF + price in top 30%: 20 (vulnerable)
  RISK_OFF + price in bottom 30%: 60 (already priced in)

EXPECTED RETURN:
  Base = 10%
  RISK_ON + high fit (Factor 1 ≥ 70): multiply by 1.3
  RISK_ON + moderate fit: multiply by 1.0
  TRANSITION: multiply by 0.8
  RISK_OFF + low fit: multiply by 0.5

MAX LOSS (regime-conditioned):
  RISK_ON: 10%
  TRANSITION: 15%
  RISK_OFF: 20-25% (market can cascade)
  Pre-revenue + RISK_OFF: 30%

DIRECTION:
  Regime fit ≥ 65 AND score ≥ 55: LONG
  RISK_OFF + pre-revenue + score < 50: FLAT (macro headwind too strong)
  score ≥ 60 AND asset type = defensive + RISK_OFF: LONG
  score < 40: FLAT

TIME HORIZON:
  RISK_ON: 30-45 days
  TRANSITION: 20-35 days
  RISK_OFF: 15-25 days

binary_event = false (macro is never a binary event)
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


async def run_macro_agent(
    data_context: dict[str, Any],
    redis_client: Any,
    daily_limit_usd: float = DEFAULT_DAILY_LIMIT_USD,
) -> QuantAgentVerdict:
    """Run the macro quant agent on macro_indicators/price_action/news data."""
    from openai import AsyncOpenAI

    partitioner = DataPartitioner()
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
        f"Analyse this macro_indicators/price_action/news data and return your "
        f"quantitative macro verdict:\n\n{json.dumps(partitioned, default=str, indent=2)}"
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
            f"Failed to parse macro agent output: {raw[:200]}",
            model="gpt-4o",
            original_error=exc,
        ) from exc

    logger.info(
        "Macro agent: score=%d direction=%s expected_return=%.1f%%",
        verdict.score,
        verdict.direction,
        verdict.expected_return_pct,
    )
    return verdict
