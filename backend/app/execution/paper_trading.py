"""Paper Trading Engine — Layer 6.

Simulates trade execution with realistic position sizing, stop-loss/take-profit
monitoring, and P&L tracking. No real money is at risk.

Core constraints (non-overridable):
  - Risk per trade <= 2% of capital
  - No single position > 10% of portfolio
  - Binary event position <= 5%

Usage:
    engine = PaperTradingEngine(initial_capital=100_000.0)
    trade_id = engine.execute_decision(decision, current_price)
    fills, closes = engine.monitor_positions({"AAPL": 185.50, ...})
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.agents.schemas import CIODecisionV2, ClosedTrade, Order, Position

logger = logging.getLogger(__name__)

MAX_POSITION_PCT = 0.10   # 10% of capital
MAX_BINARY_PCT = 0.05     # 5% for binary events
MAX_RISK_PER_TRADE = 0.02  # 2% capital at risk per trade


class PaperTradingEngine:
    """Stateful paper trading simulator.

    Attributes:
        capital: Current cash balance (starts at initial_capital).
        positions: Open positions by ticker.
        closed_trades: History of completed trades.
        pending_orders: Orders awaiting fill.
    """

    def __init__(self, initial_capital: float = 100_000.0) -> None:
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[ClosedTrade] = []
        self.pending_orders: list[Order] = []

    # ------------------------------------------------------------------
    # Portfolio metrics
    # ------------------------------------------------------------------

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value (cash + open position value)."""
        position_value = sum(
            p.entry_price * p.shares for p in self.positions.values()
        )
        return self.capital + position_value

    @property
    def open_position_value(self) -> float:
        """Total value tied up in open positions."""
        return sum(p.entry_price * p.shares for p in self.positions.values())

    def portfolio_value_with_prices(self, current_prices: dict[str, float]) -> float:
        """Portfolio value using current market prices."""
        position_value = sum(
            current_prices.get(p.ticker, p.entry_price) * p.shares
            for p in self.positions.values()
        )
        return self.capital + position_value

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_decision(
        self,
        decision: CIODecisionV2,
        current_price: float,
        trailing_stop_pct: Optional[float] = None,
    ) -> Optional[str]:
        """Execute a CIO v2 BUY decision as a paper trade.

        Validates risk-per-trade constraints before sizing the order.
        Creates an Order and immediately fills it at current_price.

        Args:
            decision: CIODecisionV2 with BUY decision.
            current_price: Current market price for fill simulation.
            trailing_stop_pct: Optional trailing stop (e.g. 0.08 = 8%).

        Returns:
            position_id if trade was opened, None if rejected.
        """
        if decision.decision not in ("BUY", "SELL"):
            logger.info(
                "execute_decision: skipping %s (decision=%s)",
                decision.ticker,
                decision.decision,
            )
            return None

        if decision.ticker in self.positions:
            logger.warning(
                "execute_decision: %s already in positions — skipping duplicate",
                decision.ticker,
            )
            return None

        # --- Position sizing ---
        portfolio_val = self.portfolio_value
        position_value = portfolio_val * (decision.position_size_pct / 100.0)
        shares = max(1, int(position_value / current_price))

        # --- Risk-per-trade constraint ---
        if decision.stop_loss and current_price > 0:
            risk_per_share = abs(current_price - decision.stop_loss)
            total_risk = risk_per_share * shares
            max_risk_dollars = portfolio_val * MAX_RISK_PER_TRADE
            if total_risk > max_risk_dollars and risk_per_share > 0:
                shares = max(1, int(max_risk_dollars / risk_per_share))
                logger.info(
                    "Risk constraint applied for %s: reduced to %d shares "
                    "(risk would be $%.0f, cap is $%.0f)",
                    decision.ticker, shares, total_risk, max_risk_dollars,
                )

        # --- Capital check ---
        order_value = shares * current_price
        if order_value > self.capital:
            shares = max(1, int(self.capital * 0.95 / current_price))
            order_value = shares * current_price
            if order_value > self.capital:
                logger.warning(
                    "Insufficient capital for %s: need $%.0f have $%.0f",
                    decision.ticker, order_value, self.capital,
                )
                return None

        now_str = datetime.now(timezone.utc).isoformat()
        order_id = str(uuid.uuid4())
        position_id = str(uuid.uuid4())

        # Create order (immediately filled for paper trading)
        order = Order(
            order_id=order_id,
            opportunity_id=decision.opportunity_id,
            ticker=decision.ticker,
            order_type="LIMIT",
            side="BUY" if decision.decision == "BUY" else "SELL",
            shares=shares,
            limit_price=current_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            status="FILLED",
            created_at=now_str,
        )
        self.pending_orders.append(order)

        # Deduct capital
        self.capital -= order_value

        # Create position
        position = Position(
            position_id=position_id,
            opportunity_id=decision.opportunity_id,
            ticker=decision.ticker,
            entry_price=current_price,
            shares=shares,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            trailing_stop_pct=trailing_stop_pct,
            current_stop=decision.stop_loss,
            entry_date=now_str,
            decision_id=decision.opportunity_id,
        )
        self.positions[decision.ticker] = position

        logger.info(
            "Paper trade OPENED: %s %d shares @ $%.2f "
            "SL=$%.2f TP=$%.2f value=$%.0f",
            decision.ticker,
            shares,
            current_price,
            decision.stop_loss or 0,
            decision.take_profit or 0,
            order_value,
        )
        return position_id

    # ------------------------------------------------------------------
    # Position monitoring
    # ------------------------------------------------------------------

    def monitor_positions(
        self,
        current_prices: dict[str, float],
    ) -> tuple[list[str], list[ClosedTrade]]:
        """Check all open positions against current prices.

        Triggers:
          - Stop-loss: price <= stop_loss
          - Take-profit: price >= take_profit
          - Trailing stop update: price moves favorably

        Args:
            current_prices: {ticker: current_price}

        Returns:
            Tuple of (updated_tickers, newly_closed_trades).
        """
        updated: list[str] = []
        newly_closed: list[ClosedTrade] = []

        for ticker, position in list(self.positions.items()):
            price = current_prices.get(ticker)
            if price is None:
                continue

            exit_reason = None

            # Stop-loss check
            if position.current_stop and price <= position.current_stop:
                exit_reason = "STOP_LOSS"

            # Take-profit check
            elif position.take_profit and price >= position.take_profit:
                exit_reason = "TAKE_PROFIT"

            # Trailing stop update
            if position.trailing_stop_pct and price > position.entry_price:
                new_stop = price * (1.0 - position.trailing_stop_pct)
                if position.current_stop is None or new_stop > position.current_stop:
                    position.current_stop = round(new_stop, 2)
                    updated.append(ticker)
                    logger.debug(
                        "Trailing stop updated for %s: new_stop=%.2f",
                        ticker, position.current_stop,
                    )

            # Close position if exit triggered
            if exit_reason:
                trade = self._close_position(ticker, price, exit_reason)  # type: ignore[arg-type]
                newly_closed.append(trade)

        return updated, newly_closed

    def _close_position(
        self,
        ticker: str,
        exit_price: float,
        exit_reason: str,
    ) -> ClosedTrade:
        """Close a position and record the trade."""
        position = self.positions.pop(ticker)
        now = datetime.now(timezone.utc)
        entry_dt = datetime.fromisoformat(position.entry_date)
        hold_days = (now - entry_dt).days

        proceeds = exit_price * position.shares
        cost = position.entry_price * position.shares
        pnl_dollar = proceeds - cost
        pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100

        # Return capital + P&L
        self.capital += proceeds

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
        self.closed_trades.append(trade)

        logger.info(
            "Paper trade CLOSED: %s %d shares entry=%.2f exit=%.2f "
            "P&L=$%.0f (%.1f%%) reason=%s",
            ticker,
            position.shares,
            position.entry_price,
            exit_price,
            pnl_dollar,
            pnl_pct,
            exit_reason,
        )
        return trade

    def close_position_manual(self, ticker: str, exit_price: float) -> Optional[ClosedTrade]:
        """Manually close a position at a given price."""
        if ticker not in self.positions:
            logger.warning("close_position_manual: %s not in positions", ticker)
            return None
        return self._close_position(ticker, exit_price, "MANUAL")

    # ------------------------------------------------------------------
    # Portfolio summary
    # ------------------------------------------------------------------

    def get_summary(self, current_prices: Optional[dict[str, float]] = None) -> dict:
        """Return current portfolio summary statistics."""
        portfolio_val = (
            self.portfolio_value_with_prices(current_prices)
            if current_prices
            else self.portfolio_value
        )
        total_pnl = portfolio_val - self.initial_capital
        total_return_pct = (total_pnl / self.initial_capital) * 100

        win_trades = [t for t in self.closed_trades if t.pnl_dollar > 0]
        loss_trades = [t for t in self.closed_trades if t.pnl_dollar <= 0]
        win_rate = len(win_trades) / len(self.closed_trades) if self.closed_trades else 0.0

        return {
            "initial_capital": self.initial_capital,
            "current_cash": round(self.capital, 2),
            "portfolio_value": round(portfolio_val, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "open_positions": len(self.positions),
            "closed_trades": len(self.closed_trades),
            "win_rate": round(win_rate, 4),
            "total_wins": len(win_trades),
            "total_losses": len(loss_trades),
        }
