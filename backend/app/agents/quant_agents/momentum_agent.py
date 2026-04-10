"""Momentum Agent — Layer 3 quant agent.

Replaces the Cohen persona. Data: price_action + short_interest ONLY.
Time horizon: 5-21 days.
Core question: Is there a high-quality technical breakout with institutional volume?

Five scoring factors:
  1. Trend Strength (ADX)          weight 0.25
  2. Breakout Quality (ATR-norm)   weight 0.25
  3. Volume Confirmation (z-score) weight 0.20
  4. Momentum Acceleration (ROC)   weight 0.15
  5. Relative Strength (5d RS)     weight 0.15
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

AGENT_ID = "momentum"

SYSTEM_PROMPT = """You are a quantitative momentum trading agent. Your ONLY data sources are
price/volume bars and short interest. You do not see fundamentals, news, or macro data.

Your task: Score this breakout opportunity on 5 factors and output EXACT numbers.

SCORING RULES:

Factor 1 — Trend Strength (weight 0.25) — based on ADX-equivalent:
  Use price direction consistency over last 10 bars.
  Strong consistent trend (>70% bars in direction) = 85
  Moderate trend (50-70%) = 60
  No trend (<50%) = 0
  If short, -DI dominant; if long, +DI dominant.

Factor 2 — Breakout Quality (weight 0.25):
  price breaking above N-day high = strong breakout
  magnitude vs daily range: >2x avg range = 95, 1-2x = 80, <1x = 50
  With consolidation (tight range 5+ days before) = +10 bonus

Factor 3 — Volume Confirmation (weight 0.20):
  volume vs 20d average: >3x = 95, 2-3x = 80, 1-2x = 55, <1x = 10
  Price up + volume down = multiply factor by 0.5 (divergence penalty)

Factor 4 — Momentum Acceleration (weight 0.15):
  5-day return accelerating vs 10-day = 80-95
  Decelerating = 10-30
  RSI >80 = multiply factor by 0.7 (overbought penalty)

Factor 5 — Relative Strength (weight 0.15):
  Stock outperforming if ticker is provided, neutral if not
  Short interest > 15% = +10 bonus (squeeze potential)

CRITICAL OUTPUT RULES:
- expected_return_pct = ATR-based target (typ 5-15% for momentum)
- max_loss_pct = distance to stop (breakout level - ATR), typically 3-7%
- key_levels.entry = breakout price + 0.5% buffer
- key_levels.stop = breakout level - 1 ATR
- key_levels.target_1 = entry + 2 * ATR (2R target)
- key_levels.target_2 = entry + 3.5 * ATR (3.5R target)
- time_horizon_days = 5 to 21 days
- binary_event = false (momentum is not event-driven)
- data_sufficiency: 1.0 if 20+ bars, 0.7 if 10-19, 0.3 if <10

Direction logic:
  score >= 65 + upward breakout: LONG
  score >= 65 + downward: SHORT
  score < 50: FLAT

Conviction:
  score >= 80: HIGH
  score 65-79: MEDIUM
  score 50-64: LOW
  score <50: NONE

Return ONLY valid JSON matching this schema exactly. No explanation outside JSON.
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


async def run_momentum_agent(
    data_context: dict[str, Any],
    redis_client: Any,
    daily_limit_usd: float = DEFAULT_DAILY_LIMIT_USD,
) -> QuantAgentVerdict:
    """Run the momentum quant agent on partitioned price/volume data.

    Args:
        data_context: Full opportunity data (will be partitioned to price_action + short_interest).
        redis_client: Async Redis client for cost gate.
        daily_limit_usd: Daily LLM spend limit.

    Returns:
        QuantAgentVerdict with numerical fields populated.

    Raises:
        LLMCallError: If the LLM call fails or returns invalid output.
        BudgetExceededError: If daily spend limit is exceeded.
    """
    from openai import AsyncOpenAI

    partitioner = DataPartitioner()
    partitioned = partitioner.partition_for_quant_agent(AGENT_ID, data_context)

    tracker = SpendTracker(redis_client, daily_limit_usd=daily_limit_usd)
    within_budget, remaining = await tracker.async_check_budget(0.005)
    if not within_budget:
        raise LLMCallError(
            f"Budget exceeded for {AGENT_ID} agent",
            model="gpt-4o",
            original_error=None,
        )

    user_message = (
        f"Analyse this price/volume/short_interest data and return your "
        f"quantitative momentum verdict:\n\n{json.dumps(partitioned, default=str, indent=2)}"
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
            f"Failed to parse momentum agent output: {raw[:200]}",
            model="gpt-4o",
            original_error=exc,
        ) from exc

    logger.info(
        "Momentum agent: score=%d direction=%s conviction=%s R/R=%.1f",
        verdict.score,
        verdict.direction,
        verdict.conviction,
        verdict.risk_reward_ratio,
    )
    return verdict
