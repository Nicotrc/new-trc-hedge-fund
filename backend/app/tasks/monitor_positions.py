"""Celery task: monitor open paper trading positions every 5 minutes.

Checks all open positions against current market prices.
Triggers stop-loss and take-profit closes automatically.
Publishes POSITION_CLOSED events for the SSE feed.
"""

from __future__ import annotations

import logging
import os

import redis

from app.events.publisher import publish_event
from app.execution.position_manager import PositionManager
from app.learning.feedback import evaluate_closed_trade, persist_evaluation
from app.tasks.celery_app import app

logger = logging.getLogger(__name__)

_REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")


@app.task(name="app.tasks.monitor_positions.run", bind=True, max_retries=2)
def run(self) -> dict:
    """Fetch current prices and check all open positions for SL/TP triggers."""
    r = redis.from_url(_REDIS_URL)
    try:
        pm = PositionManager(r)
        positions = pm.get_positions()

        if not positions:
            return {"checked": 0, "closed": 0}

        tickers = list(positions.keys())
        current_prices = _fetch_prices(tickers)

        if not current_prices:
            return {"checked": len(tickers), "closed": 0, "reason": "no_prices"}

        closed_count = 0
        for ticker, position in list(positions.items()):
            price = current_prices.get(ticker)
            if price is None:
                continue

            exit_reason = None
            if position.current_stop and price <= position.current_stop:
                exit_reason = "STOP_LOSS"
            elif position.take_profit and price >= position.take_profit:
                exit_reason = "TAKE_PROFIT"

            # Update trailing stop
            if position.trailing_stop_pct and price > position.entry_price:
                new_stop = price * (1.0 - position.trailing_stop_pct)
                if position.current_stop is None or new_stop > position.current_stop:
                    position.current_stop = round(new_stop, 2)
                    pm.save_position(position)

            if exit_reason:
                _close_position(ticker, position, price, exit_reason, pm, r)
                closed_count += 1

        logger.info(
            "Position monitor: checked=%d closed=%d", len(tickers), closed_count
        )
        return {"checked": len(tickers), "closed": closed_count}

    finally:
        r.close()


def _fetch_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch current prices via YFinance."""
    prices: dict[str, float] = {}
    try:
        import yfinance as yf
        data = yf.download(tickers, period="1d", progress=False, auto_adjust=True)
        if data.empty:
            return prices
        closes = data["Close"] if "Close" in data else data
        if len(tickers) == 1:
            prices[tickers[0]] = float(closes.iloc[-1])
        else:
            for ticker in tickers:
                if ticker in closes.columns:
                    val = closes[ticker].dropna()
                    if not val.empty:
                        prices[ticker] = float(val.iloc[-1])
    except Exception:
        logger.exception("Failed to fetch prices for position monitoring")
    return prices


def _close_position(ticker, position, exit_price: float, exit_reason: str, pm, r) -> None:
    """Close a position and publish POSITION_CLOSED event."""
    from datetime import datetime, timezone
    import uuid
    from app.agents.schemas import ClosedTrade

    now = datetime.now(timezone.utc)
    entry_dt = datetime.fromisoformat(position.entry_date)
    hold_days = max(0, (now - entry_dt).days)

    pnl_dollar = (exit_price - position.entry_price) * position.shares
    pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100

    trade = ClosedTrade(
        trade_id=str(uuid.uuid4()),
        opportunity_id=position.opportunity_id,
        ticker=ticker,
        entry_price=position.entry_price,
        exit_price=exit_price,
        shares=position.shares,
        pnl_pct=round(pnl_pct, 4),
        pnl_dollar=round(pnl_dollar, 2),
        hold_days=hold_days,
        exit_reason=exit_reason,  # type: ignore[arg-type]
        entry_date=position.entry_date,
        exit_date=now.isoformat(),
        decision_id=position.decision_id,
    )

    pm.remove_position(ticker)
    pm.adjust_capital(exit_price * position.shares)
    pm.append_closed_trade(trade)

    publish_event(r, "POSITION_CLOSED", {
        "ticker": ticker,
        "exit_reason": exit_reason,
        "pnl_pct": round(pnl_pct, 2),
        "pnl_dollar": round(pnl_dollar, 2),
        "hold_days": hold_days,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
    })

    # Persist trade evaluation for Learning System
    try:
        decision_raw = r.get(f"decision:{position.opportunity_id}")
        if decision_raw:
            from app.agents.schemas import CIODecisionV2
            decision = CIODecisionV2(**__import__("json").loads(decision_raw))
            evaluation = evaluate_closed_trade(trade, decision)
            persist_evaluation(evaluation, r)
    except Exception:
        logger.debug("Could not persist trade evaluation for %s (decision not in Redis)", ticker)

    logger.info(
        "Position closed: %s exit=%s price=%.2f P&L=$%.0f (%.1f%%)",
        ticker, exit_reason, exit_price, pnl_dollar, pnl_pct,
    )
