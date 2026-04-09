"""Trade evaluation engine — Layer 7 Learning & Feedback.

Evaluates closed trades against their original CIO v2 decisions and
produces TradeEvaluation records for agent calibration.

Called after each trade is closed by the paper trading engine.
Minimum 10 trades per agent before weights adjust (prevents overfitting).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.agents.schemas import CIODecisionV2, ClosedTrade, TradeEvaluation

logger = logging.getLogger(__name__)

_EVAL_HISTORY_KEY = "learning:trade_evaluations"
_MIN_TRADES_FOR_CALIBRATION = 10


def evaluate_closed_trade(
    trade: ClosedTrade,
    decision: CIODecisionV2,
) -> TradeEvaluation:
    """Evaluate a closed trade against its original CIO decision.

    Args:
        trade: The closed paper trade.
        decision: The original CIO v2 decision that opened the trade.

    Returns:
        TradeEvaluation with per-agent correctness assessment.
    """
    actual_return = trade.pnl_pct / 100.0
    direction_correct = trade.pnl_pct > 0

    # MC expected value as fraction: (expected_value - entry_price) / entry_price
    if decision.entry_price and decision.entry_price > 0:
        predicted_return = (decision.monte_carlo.expected_value - decision.entry_price) / decision.entry_price
    else:
        predicted_return = decision.monte_carlo.expected_value / 100.0  # fallback

    magnitude_error = abs(predicted_return - actual_return)

    # Per-agent assessment
    agent_scores: dict[str, int] = {}
    agent_directions: dict[str, str] = {}
    which_agents_correct: dict[str, bool] = {}

    # The agent_weights_used tells us which agents were active
    for agent_id in decision.agent_weights_used:
        agent_scores[agent_id] = 0  # scores not stored at decision level (future improvement)
        # All agents pointing to decision.decision direction
        agent_directions[agent_id] = (
            "LONG" if decision.decision == "BUY"
            else "SHORT" if decision.decision == "SELL"
            else "FLAT"
        )
        which_agents_correct[agent_id] = direction_correct

    evaluation = TradeEvaluation(
        trade_id=trade.trade_id,
        opportunity_id=trade.opportunity_id,
        ticker=trade.ticker,
        direction_correct=direction_correct,
        predicted_return_pct=round(predicted_return * 100, 4),
        actual_return_pct=round(actual_return * 100, 4),
        magnitude_error=round(magnitude_error, 6),
        agent_scores=agent_scores,
        agent_directions=agent_directions,
        which_agents_correct=which_agents_correct,
    )

    logger.info(
        "Trade evaluation: %s direction_correct=%s predicted=%.1f%% actual=%.1f%% "
        "magnitude_error=%.3f",
        trade.ticker,
        direction_correct,
        predicted_return * 100,
        actual_return * 100,
        magnitude_error,
    )
    return evaluation


def persist_evaluation(evaluation: TradeEvaluation, redis_client: Any) -> None:
    """Persist a trade evaluation to Redis for Learning System."""
    redis_client.rpush(_EVAL_HISTORY_KEY, evaluation.model_dump_json())


def get_recent_evaluations(
    redis_client: Any,
    limit: int = 50,
) -> list[TradeEvaluation]:
    """Load recent trade evaluations from Redis."""
    raw_list = redis_client.lrange(_EVAL_HISTORY_KEY, -limit, -1)
    evaluations: list[TradeEvaluation] = []
    for raw in raw_list:
        data_str = raw.decode() if isinstance(raw, bytes) else raw
        try:
            evaluations.append(TradeEvaluation(**json.loads(data_str)))
        except Exception:
            logger.exception("Failed to deserialize trade evaluation")
    return evaluations


def compute_rolling_accuracy(
    evaluations: list[TradeEvaluation],
    agent_id: str,
    window: int = 20,
) -> Optional[float]:
    """Compute rolling hit rate for a specific agent over last N trades.

    Returns None if fewer than MIN_TRADES_FOR_CALIBRATION trades are available.
    """
    relevant = [
        e for e in evaluations[-window:]
        if agent_id in e.which_agents_correct
    ]
    if len(relevant) < _MIN_TRADES_FOR_CALIBRATION:
        return None
    correct = sum(1 for e in relevant if e.which_agents_correct.get(agent_id, False))
    return correct / len(relevant)
