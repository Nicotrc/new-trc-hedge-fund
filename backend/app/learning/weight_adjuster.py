"""Dynamic agent weight adjuster — Layer 7 Learning.

Bayesian update of Meta-Agent weights based on rolling per-agent hit rates.
Runs after every closed trade, but only adjusts weights when each agent
has at least MIN_TRADES_FOR_CALIBRATION evaluations.

Weight update logic:
  - Hit rate > 60%: upweight agent (multiplier 1.1-1.3x)
  - Hit rate 40-60%: neutral (multiplier 1.0x)
  - Hit rate < 40%: downweight agent (multiplier 0.7-0.9x)

All weights re-normalised to sum to 1.0 after adjustment.
Floor: 0.05 per agent (no agent fully silenced).
"""

from __future__ import annotations

import logging
from typing import Any

from app.learning.feedback import (
    compute_rolling_accuracy,
    get_recent_evaluations,
)

logger = logging.getLogger(__name__)

_AGENT_IDS = ["momentum", "value", "event", "macro", "risk"]
_ACCURACY_KEY_PREFIX = "agent_accuracy:"
_MIN_TRADES = 10

# Default equal weights
_DEFAULT_WEIGHTS = {a: 0.20 for a in _AGENT_IDS}


def update_agent_weights(redis_client: Any) -> dict[str, float]:
    """Load evaluations, compute accuracy, and persist updated weights.

    Args:
        redis_client: Sync Redis client.

    Returns:
        Updated weight dict {agent_id: weight}.
    """
    evaluations = get_recent_evaluations(redis_client, limit=50)

    if len(evaluations) < _MIN_TRADES:
        logger.debug(
            "Weight adjuster: only %d evaluations — need %d before adjusting",
            len(evaluations),
            _MIN_TRADES,
        )
        return _DEFAULT_WEIGHTS.copy()

    weights: dict[str, float] = {}
    for agent_id in _AGENT_IDS:
        accuracy = compute_rolling_accuracy(evaluations, agent_id, window=20)
        if accuracy is None:
            weights[agent_id] = 0.20  # default
            continue

        # Bayesian-style multiplier
        if accuracy > 0.65:
            multiplier = 1.0 + (accuracy - 0.65) * 2.0  # up to 1.3x at 80%
        elif accuracy < 0.40:
            multiplier = max(0.5, accuracy / 0.40)  # down to 0.5x at 20%
        else:
            multiplier = 1.0

        weights[agent_id] = max(0.05, 0.20 * multiplier)

        # Persist individual accuracy to Redis for Meta-Agent to read
        redis_client.set(f"{_ACCURACY_KEY_PREFIX}{agent_id}", str(round(accuracy, 4)))
        logger.info(
            "Agent %s: accuracy=%.2f multiplier=%.2f new_weight=%.4f",
            agent_id, accuracy, multiplier, weights[agent_id],
        )

    # Re-normalise
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}
    logger.info("Updated agent weights: %s", weights)
    return weights


def get_current_accuracies(redis_client: Any) -> dict[str, float]:
    """Read current per-agent accuracy scores from Redis."""
    result: dict[str, float] = {}
    for agent_id in _AGENT_IDS:
        raw = redis_client.get(f"{_ACCURACY_KEY_PREFIX}{agent_id}")
        if raw:
            try:
                result[agent_id] = float(raw)
            except ValueError:
                result[agent_id] = 0.5
        else:
            result[agent_id] = 0.5
    return result
