"""Performance attribution — Layer 7 Learning.

Computes portfolio-level performance metrics: Sharpe ratio, Sortino ratio,
win rate by strategy, maximum drawdown, and per-agent attribution.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from app.agents.schemas import ClosedTrade

logger = logging.getLogger(__name__)

_RISK_FREE_RATE_DAILY = 0.05 / 252  # 5% annual risk-free rate


def compute_performance_stats(trades: list[ClosedTrade]) -> dict[str, Any]:
    """Compute portfolio performance statistics from closed trades.

    Args:
        trades: All closed paper trades.

    Returns:
        Dict with: total_return, sharpe, sortino, max_drawdown, win_rate,
        avg_hold_days, avg_win_pct, avg_loss_pct, profit_factor, n_trades.
    """
    if not trades:
        return {
            "n_trades": 0,
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "avg_hold_days": 0.0,
        }

    returns = [t.pnl_pct / 100.0 for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    total_return = sum(returns)
    win_rate = len(wins) / len(returns)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    # Profit factor: total wins / total losses
    total_win = sum(wins)
    total_loss = abs(sum(losses))
    profit_factor = total_win / total_loss if total_loss > 0 else float("inf")

    # Sharpe ratio (using trade returns as daily approximation)
    if len(returns) > 1:
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std_ret = math.sqrt(variance)
        sharpe = (mean_ret - _RISK_FREE_RATE_DAILY) / std_ret if std_ret > 0 else 0.0
        # Annualise (sqrt(252) factor)
        sharpe_annualised = sharpe * math.sqrt(252)
    else:
        sharpe_annualised = 0.0

    # Sortino ratio (only uses downside deviation)
    downside_returns = [r for r in returns if r < 0]
    if downside_returns and len(returns) > 1:
        downside_var = sum(r ** 2 for r in downside_returns) / len(returns)
        downside_std = math.sqrt(downside_var)
        mean_ret = sum(returns) / len(returns)
        sortino = (mean_ret - _RISK_FREE_RATE_DAILY) / downside_std if downside_std > 0 else 0.0
        sortino_annualised = sortino * math.sqrt(252)
    else:
        sortino_annualised = 0.0

    avg_hold_days = sum(t.hold_days for t in trades) / len(trades)

    return {
        "n_trades": len(trades),
        "total_return_pct": round(total_return * 100, 2),
        "win_rate": round(win_rate, 4),
        "avg_win_pct": round(avg_win * 100, 2),
        "avg_loss_pct": round(avg_loss * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe_annualised, 3),
        "sortino": round(sortino_annualised, 3),
        "avg_hold_days": round(avg_hold_days, 1),
    }


def compute_exit_reason_breakdown(trades: list[ClosedTrade]) -> dict[str, int]:
    """Count trades by exit reason."""
    breakdown: dict[str, int] = {}
    for t in trades:
        breakdown[t.exit_reason] = breakdown.get(t.exit_reason, 0) + 1
    return breakdown
