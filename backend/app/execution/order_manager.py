"""Order Manager — order lifecycle management for paper trading.

Tracks pending, filled, and cancelled orders. Provides order history
for audit trails and performance attribution.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

from app.agents.schemas import CIODecisionV2, Order

logger = logging.getLogger(__name__)

_ORDERS_KEY = "paper_trading:orders"
_ORDER_HISTORY_KEY = "paper_trading:order_history"


class OrderManager:
    """Manages paper trading order lifecycle."""

    def __init__(self, redis_client: Any) -> None:
        self._r = redis_client

    def create_order(
        self,
        decision: CIODecisionV2,
        shares: int,
        fill_price: float,
    ) -> Order:
        """Create and immediately fill a paper trading order."""
        order = Order(
            order_id=str(uuid.uuid4()),
            opportunity_id=decision.opportunity_id,
            ticker=decision.ticker,
            order_type="LIMIT",
            side="BUY" if decision.decision == "BUY" else "SELL",
            shares=shares,
            limit_price=fill_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            status="FILLED",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._save_order(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        raw = self._r.hget(_ORDERS_KEY, order_id)
        if not raw:
            return False
        data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        data["status"] = "CANCELLED"
        self._r.hset(_ORDERS_KEY, order_id, json.dumps(data))
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        raw = self._r.hget(_ORDERS_KEY, order_id)
        if not raw:
            return None
        data_str = raw.decode() if isinstance(raw, bytes) else raw
        return Order(**json.loads(data_str))

    def get_all_orders(self) -> list[Order]:
        raw_map = self._r.hgetall(_ORDERS_KEY)
        orders: list[Order] = []
        for _, data in raw_map.items():
            data_str = data.decode() if isinstance(data, bytes) else data
            try:
                orders.append(Order(**json.loads(data_str)))
            except Exception:
                pass
        return orders

    def get_orders_for_ticker(self, ticker: str) -> list[Order]:
        return [o for o in self.get_all_orders() if o.ticker == ticker]

    def _save_order(self, order: Order) -> None:
        self._r.hset(_ORDERS_KEY, order.order_id, order.model_dump_json())
        self._r.rpush(_ORDER_HISTORY_KEY, order.model_dump_json())
