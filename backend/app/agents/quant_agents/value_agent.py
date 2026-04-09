"""Value Agent — Layer 3 quant agent.

Replaces Buffett + Munger + Ackman personas (consolidated).
Data: fundamentals + insider_trades ONLY.
Time horizon: 30-180 days.
Core question: Is the stock trading below fair value with insider confirmation?

Five scoring factors:
  1. Margin of Safety (3-model avg)   weight 0.30
  2. Earnings Quality (3 sub-metrics) weight 0.25
  3. Insider Conviction (dollar-wtd)  weight 0.20
  4. Growth Trajectory                weight 0.15
  5. Valuation Context (PE vs sector) weight 0.10

CRITICAL LIMITATION: Pre-revenue companies produce direction=FLAT,
data_sufficiency=0.15 by design — they are Event/Momentum plays.
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

AGENT_ID = "value"

SYSTEM_PROMPT = """You are a quantitative value investing agent. Your ONLY data sources are
fundamentals and insider trades. You do not see price charts, news, or macro data.

Your task: Score this investment on 5 fundamental factors and output EXACT numbers.

SCORING RULES:

Factor 1 — Margin of Safety (weight 0.30):
  Three fair value estimates (average them):
    A. PE-based: EPS × sector median PE (tech ~25, healthcare ~20, energy ~12)
    B. FCF yield: FCF / market cap vs sector avg yield (inverse)
    C. EV/Revenue: for pre-revenue companies (< $0 net income)
  margin_of_safety = (fair_value_avg - current_implied) / fair_value_avg
  Scoring: MoS < 0% → 5, 10-25% → 55, 25-40% → 80, > 40% → 95
  PRE-REVENUE (no EPS, no FCF): ALL THREE MODELS INVALID → Factor 1 = 5, data_sufficiency = 0.15

Factor 2 — Earnings Quality (weight 0.25):
  Average of 3 sub-metrics:
    A. Cash conversion: FCF / net_income > 1.0 → 90, < 0 → 10
    B. Leverage health: D/E < 0.5 → 85, 0.5-2.0 → 55, > 2.0 → 10
    C. Profitability: net margin > 20% → 90, 0-20% → 60, < 0% → 10
  Missing data → 40 default per sub-metric

Factor 3 — Insider Conviction (weight 0.20):
  Dollar-weighted net buy/sell ratio:
    net_ratio = (buy_value - sell_value) / (buy_value + sell_value + 1)
    Range: -1.0 (all selling) to +1.0 (all buying)
  Scoring: net_ratio > 0.7 → 90, 0.3-0.7 → 70, -0.3-0.3 → 50, < -0.7 → 10
  Cluster bonus: >= 3 unique buyers in 30d → +15 (cap at 100)
  Override: CEO + CFO both selling > $1M → factor capped at 5
  No insider data → 40

Factor 4 — Growth Trajectory (weight 0.15):
  Revenue growth YoY: > 20% → 90, 5-20% → 70, 0-5% → 50, < -10% → 5
  Catches value traps: cheap stock with declining revenue

Factor 5 — Valuation Context (weight 0.10):
  PE vs sector median:
    pe / sector_pe < 0.5 → 90, 0.5-0.8 → 70, 0.8-1.2 → 45, > 1.5 → 10
  Micro-cap penalty: mkt cap < $100M → multiply by 0.8
  No PE data (pre-revenue) → 20

EXPECTED RETURN CALCULATION:
  expected_return_pct = margin_of_safety_pct * P(re-rating)
  P(re-rating) = 0.30 + (insider_factor/100) * 0.40
  Range: 0.30 (no insider signal) to 0.70 (strong insider buying)

MAX LOSS CALCULATION:
  = compression to sector bottom-decile PE / current price
  Minimum 8%, maximum 35% for going-concerns

TIME HORIZON:
  Strong insider (factor >= 70) → 60 days
  Moderate insider → 90-120 days
  No insider → 150-180 days

DIRECTION LOGIC:
  score >= 60 AND positive MoS: LONG
  score >= 60 AND negative MoS: SHORT (overvalued short)
  pre-revenue: FLAT (by design — these are event/momentum plays)
  score < 40: FLAT

KEY LEVELS for value plays:
  entry = current price (buy at market for value)
  stop = current price * (1 - max_loss_pct/100)
  target_1 = current price * (1 + expected_return_pct/100)
  target_2 = fair_value (full re-rating target)

Return ONLY valid JSON matching the schema exactly. No explanation outside JSON.
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


async def run_value_agent(
    data_context: dict[str, Any],
    redis_client: Any,
    daily_limit_usd: float = DEFAULT_DAILY_LIMIT_USD,
) -> QuantAgentVerdict:
    """Run the value quant agent on partitioned fundamental/insider data."""
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
        f"Analyse this fundamentals/insider data and return your "
        f"quantitative value verdict:\n\n{json.dumps(partitioned, default=str, indent=2)}"
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
            f"Failed to parse value agent output: {raw[:200]}",
            model="gpt-4o",
            original_error=exc,
        ) from exc

    logger.info(
        "Value agent: score=%d direction=%s data_sufficiency=%.2f",
        verdict.score,
        verdict.direction,
        verdict.data_sufficiency,
    )
    return verdict
