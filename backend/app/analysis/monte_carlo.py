"""Monte Carlo simulation engine — Layer 5 component.

Runs 10,000 price-path simulations to produce probability distributions
for CIO decision-making.

Two simulation modes:
  1. Standard (Brownian motion): smooth drift + volatility
  2. Binary event (jump-diffusion): smooth drift + discrete jump at catalyst_date

The binary event mode is triggered when any agent sets binary_event=True.
It models the bimodal outcome distribution (two peaks, not one bell curve)
that characterizes biotech PDUFA and Phase 3 readout events.

Usage:
    result = run_monte_carlo(
        current_price=50.0,
        daily_drift=0.001,          # annualized return / 252
        daily_vol=0.025,            # realized vol / sqrt(252)
        days=21,
        binary_event=True,
        catalyst_day=12,
        p_success=0.66,
        upside_move=0.55,           # +55% on success
        downside_move=-0.35,        # -35% on failure
        n_simulations=10000,
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.agents.schemas import MonteCarloResult

logger = logging.getLogger(__name__)

_DEFAULT_N = 10_000
_SEED = 42  # reproducible simulations


def run_monte_carlo(
    current_price: float,
    daily_drift: float,
    daily_vol: float,
    days: int,
    binary_event: bool = False,
    catalyst_day: Optional[int] = None,
    p_success: float = 0.5,
    upside_move: float = 0.30,
    downside_move: float = -0.25,
    n_simulations: int = _DEFAULT_N,
    seed: Optional[int] = _SEED,
) -> MonteCarloResult:
    """Run Monte Carlo simulation and return probability distribution.

    Args:
        current_price: Current asset price.
        daily_drift: Expected daily return (drift) = annual_return / 252.
        daily_vol: Daily volatility = realized_vol_20d / sqrt(252).
        days: Number of trading days to simulate.
        binary_event: If True, inject a discrete jump at catalyst_day.
        catalyst_day: Day index (1-indexed) when event occurs. Required if binary_event.
        p_success: Probability of positive outcome for binary event.
        upside_move: Fractional return on success (e.g. 0.55 = +55%).
        downside_move: Fractional return on failure (e.g. -0.35 = -35%).
        n_simulations: Number of Monte Carlo paths.
        seed: Random seed for reproducibility (None for non-deterministic).

    Returns:
        MonteCarloResult with full probability distribution.
    """
    if days < 1:
        days = 1
    if daily_vol <= 0:
        daily_vol = 0.02  # floor at 2% daily vol

    rng = np.random.default_rng(seed)

    # Generate random daily returns: shape (n_simulations, days)
    daily_returns = rng.normal(
        loc=daily_drift,
        scale=daily_vol,
        size=(n_simulations, days),
    )

    # Inject binary event jump at catalyst_day
    if binary_event and catalyst_day is not None:
        day_idx = min(catalyst_day - 1, days - 1)  # 0-indexed
        outcomes = rng.random(n_simulations)
        jumps = np.where(outcomes < p_success, upside_move, downside_move)
        daily_returns[:, day_idx] += jumps

    # Compound returns to build price paths
    # cumulative_returns[i, d] = product of (1 + r) for day 0..d
    cum_returns = np.cumprod(1.0 + daily_returns, axis=1)
    # Final price = current_price * cumulative_return at last day
    final_prices = current_price * cum_returns[:, -1]
    final_returns = (final_prices - current_price) / current_price  # fractional

    # Track maximum drawdown across paths
    all_prices = current_price * cum_returns  # shape (n_sim, days)
    running_max = np.maximum.accumulate(all_prices, axis=1)
    drawdowns = (all_prices - running_max) / running_max  # negative
    max_drawdowns = drawdowns.min(axis=1)  # worst drawdown per path

    # --- Compute statistics ---
    mean_final = float(final_prices.mean())
    median_final = float(np.median(final_prices))

    mean_return = float(final_returns.mean())

    # Profit/loss split
    is_profit = final_returns > 0.0
    is_loss_10 = final_returns < -0.10
    is_gain_20 = final_returns > 0.20
    is_gain_50 = final_returns > 0.50

    prob_profit = float(is_profit.mean())
    prob_loss_10 = float(is_loss_10.mean())
    prob_gain_20 = float(is_gain_20.mean())
    prob_gain_50 = float(is_gain_50.mean())

    profitable_returns = final_returns[is_profit]
    loss_returns = final_returns[~is_profit]
    avg_profit = float(profitable_returns.mean()) if len(profitable_returns) > 0 else 0.0
    avg_loss = float(loss_returns.mean()) if len(loss_returns) > 0 else 0.0

    # VaR (Value at Risk) — loss at given confidence level
    var_95 = float(np.percentile(final_returns, 5))    # 5th percentile of returns
    var_99 = float(np.percentile(final_returns, 1))    # 1st percentile of returns

    # Percentiles
    percentiles = {
        "p5": float(np.percentile(final_returns, 5)),
        "p25": float(np.percentile(final_returns, 25)),
        "p50": float(np.percentile(final_returns, 50)),
        "p75": float(np.percentile(final_returns, 75)),
        "p95": float(np.percentile(final_returns, 95)),
    }

    max_drawdown_median = float(np.median(max_drawdowns))

    result = MonteCarloResult(
        expected_value=mean_final,
        median_value=median_final,
        prob_profit=prob_profit,
        prob_loss_10=prob_loss_10,
        prob_gain_20=prob_gain_20,
        prob_gain_50=prob_gain_50,
        var_95=var_95,
        var_99=var_99,
        max_drawdown_median=max_drawdown_median,
        avg_profit_pct=avg_profit * 100,
        avg_loss_pct=avg_loss * 100,
        percentiles=percentiles,
        n_simulations=n_simulations,
        time_horizon_days=days,
    )

    logger.info(
        "MC simulation: days=%d binary=%s P(profit)=%.2f E[return]=%.2f%% "
        "VaR95=%.2f%% P(loss>10%)=%.2f%%",
        days,
        binary_event,
        prob_profit,
        mean_return * 100,
        var_95 * 100,
        prob_loss_10,
    )
    return result


def mc_params_from_verdicts(
    verdicts: list,  # list[QuantAgentVerdict]
    agent_weights: dict[str, float],
    current_price: float,
    realized_vol_20d: float,
) -> dict:
    """Derive Monte Carlo parameters from weighted agent verdicts.

    Args:
        verdicts: List of QuantAgentVerdict objects.
        agent_weights: {agent_id: weight} from MetaAgentReport.
        current_price: Current asset price.
        realized_vol_20d: 20-day realized volatility (annualized, e.g. 0.45 = 45%).

    Returns:
        Dict of kwargs ready to pass to run_monte_carlo().
    """
    if not verdicts:
        return {
            "current_price": current_price,
            "daily_drift": 0.0,
            "daily_vol": realized_vol_20d / (252 ** 0.5),
            "days": 21,
        }

    # Build weight map
    weight_map = {v.agent_id: agent_weights.get(v.agent_id, 0.20) for v in verdicts}
    total_weight = sum(weight_map.values())

    # Weighted expected return (convert to daily drift)
    weighted_annual = sum(
        v.expected_return_pct * weight_map[v.agent_id]
        for v in verdicts
    ) / (total_weight or 1.0)
    # expected_return_pct is over time_horizon_days — convert to annualized
    avg_horizon = max(
        sum(v.time_horizon_days * weight_map[v.agent_id] for v in verdicts) / (total_weight or 1.0),
        1.0,
    )
    daily_drift = (weighted_annual / 100) / avg_horizon

    daily_vol = realized_vol_20d / (252 ** 0.5)
    days = int(max(avg_horizon, 5))

    # Binary event detection
    binary_verdicts = [v for v in verdicts if v.binary_event and v.catalyst_date]
    binary_event = len(binary_verdicts) > 0
    catalyst_day = None
    p_success = 0.5
    upside_move = 0.30
    downside_move = -0.25

    if binary_event and binary_verdicts:
        ev = binary_verdicts[0]
        # Estimate catalyst_day as days until catalyst_date
        if ev.catalyst_date:
            from datetime import date
            try:
                cat_date = date.fromisoformat(ev.catalyst_date)
                today = date.today()
                catalyst_day = max(1, (cat_date - today).days)
                catalyst_day = min(catalyst_day, days)
            except ValueError:
                catalyst_day = min(7, days)
        else:
            catalyst_day = min(7, days)

        # Derive upside/downside from event agent expected return
        p_success = ev.confidence  # event agent uses confidence as P(success) proxy
        # R/R implies: EV = p * upside + (1-p) * downside = expected_return
        # upside from key_levels or default
        if ev.key_levels.target_1 and current_price > 0:
            upside_move = (ev.key_levels.target_1 - current_price) / current_price
        else:
            upside_move = 0.40
        # downside from max_loss
        downside_move = -abs(ev.max_loss_pct / 100)

    return {
        "current_price": current_price,
        "daily_drift": daily_drift,
        "daily_vol": daily_vol,
        "days": days,
        "binary_event": binary_event,
        "catalyst_day": catalyst_day,
        "p_success": p_success,
        "upside_move": upside_move,
        "downside_move": downside_move,
    }
