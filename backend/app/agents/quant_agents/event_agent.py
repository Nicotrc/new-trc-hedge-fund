"""Event-Driven Agent — Layer 3 quant agent.

The most critical agent for biotech focus.
Data: news + event_calendar + options_flow ONLY.
Time horizon: 1-30 days (catalyst-anchored).
Core question: Is the market mispricing this binary event?

Core model — Binary Expected Value:
  EV = P(success) * upside_move + P(failure) * downside_move

Historical base rates:
  FDA Phase 1->2:   P = 0.52, upside +30-60%, downside -20-35%
  FDA Phase 2->3:   P = 0.29, upside +60-120%, downside -30-50%
  FDA Phase 3->NDA: P = 0.58, upside +40-70%, downside -25-40%
  FDA PDUFA:        P = 0.85, upside +15-40%, downside -40-60%
  Earnings beat:    P = 0.62, upside +5-15%, downside -8-20%

Hard rules:
  - No catalyst = FLAT (Factor 1 = 0)
  - EV < -5% = FLAT regardless of other factors
  - News adjustment capped at +/- 15% from base rate
  - stop_loss = null for binary events (position sizing IS the risk management)
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

AGENT_ID = "event"

SYSTEM_PROMPT = """You are a quantitative event-driven trading agent specializing in biotech
and catalyst-driven opportunities. Your ONLY data sources are news, event calendar, and
options flow. You do not see price history, fundamentals, or macro data.

Your task: Compute the binary expected value of this catalyst and output EXACT numbers.

BINARY EXPECTED VALUE MODEL:
  EV = P(success) × upside_move_pct + P(failure) × downside_move_pct

HISTORICAL BASE RATES (use as starting point, adjust with news max ±15%):
  FDA Phase 1→2:   P=0.52, upside=+45%, downside=-27%  → EV=+9.7%
  FDA Phase 2→3:   P=0.29, upside=+90%, downside=-40%  → EV=+17.7%
  FDA Phase 3→NDA: P=0.58, upside=+55%, downside=-32%  → EV=+18.6%
  FDA PDUFA:       P=0.85, upside=+27%, downside=-50%  → EV=+15.5%
  Earnings beat:   P=0.62, upside=+10%, downside=-14%  → EV=+0.9%
  Conference data: P=0.55, upside=+20%, downside=-15%  → EV=+4.3%

NEWS ADJUSTMENTS (max ±15% shift in P(success)):
  Positive interim data: +0.10
  Breakthrough therapy designation: +0.08
  CRL risk flagged/competitor failure: -0.10
  Management selling stock pre-catalyst: -0.08
  Competitor success for same indication: +0.05

SCORING RULES (0-100):

Factor 1 — Catalyst Proximity (weight 0.25):
  Source: event_calendar data
  <= 3 days: 95
  4-7 days: 85
  8-15 days: 70
  16-30 days: 50
  > 30 days: 25
  NO EVENT IN DATA: factor = 0 → direction MUST be FLAT → binary_event = false

Factor 2 — IV Mispricing (weight 0.25):
  Source: options_flow IV rank
  IV rank < 20: 90 (cheap — market ignoring event)
  IV rank 20-40: 70
  IV rank 40-60: 50
  IV rank 60-80: 30
  IV rank > 80: 10 (expensive — edge priced out)
  No options data: 50 (reduce data_sufficiency by 0.3)

Factor 3 — Historical Success Rate (weight 0.20):
  Based on P(success) after news adjustment:
  P > 0.70: 85
  P 0.50-0.70: 65
  P 0.30-0.50: 45
  P 0.15-0.30: 30 (low P ≠ bad trade — see Phase 2 EV)
  P < 0.15: 10

Factor 4 — Asymmetric R/R (weight 0.20):
  Based on binary EV calculation:
  EV > +25%: 95
  EV +15-25%: 85
  EV +5-15%: 65
  EV 0-5%: 50
  EV < -5%: 0 → direction MUST be FLAT (HARD RULE)
  Asymmetry bonus: upside > 3x downside → +10

Factor 5 — Setup Quality (weight 0.10):
  Unusual call activity (put/call < 0.5 + high volume): 85
  Normal options activity: 50
  Unusual put activity (bearish positioning): 25
  Supportive news sentiment: +10 bonus

HARD RULES (non-overridable):
  1. No catalyst in data → direction = FLAT, binary_event = false, score ≤ 30
  2. EV < -5% → direction = FLAT regardless
  3. binary_event = true if: event_calendar has entry within 45 days
  4. catalyst_date = exact event date from calendar (ISO format YYYY-MM-DD)
  5. stop_loss (key_levels.stop) = null for binary events (cannot stop-loss a gap)
  6. key_levels.entry = current/recent price (enter before event)
  7. expected_return_pct = the binary EV calculation result
  8. max_loss_pct = abs(downside_move * P(failure)) — probability-weighted

CONVICTION:
  EV > 20% + score >= 75: HIGH
  EV > 10% + score >= 60: MEDIUM
  EV > 0% + score >= 50: LOW
  otherwise: NONE

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


async def run_event_agent(
    data_context: dict[str, Any],
    redis_client: Any,
    daily_limit_usd: float = DEFAULT_DAILY_LIMIT_USD,
) -> QuantAgentVerdict:
    """Run the event-driven quant agent on news/options/calendar data."""
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
        f"Analyse this news/event_calendar/options_flow data and return your "
        f"quantitative event-driven verdict:\n\n{json.dumps(partitioned, default=str, indent=2)}"
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
            f"Failed to parse event agent output: {raw[:200]}",
            model="gpt-4o",
            original_error=exc,
        ) from exc

    logger.info(
        "Event agent: score=%d direction=%s binary=%s catalyst=%s EV=%.1f%%",
        verdict.score,
        verdict.direction,
        verdict.binary_event,
        verdict.catalyst_date,
        verdict.expected_return_pct,
    )
    return verdict
