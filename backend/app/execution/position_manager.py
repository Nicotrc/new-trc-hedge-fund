"""Position Manager — persists and retrieves paper trading state from Redis/DB.

Bridges the stateless Celery task environment with the stateful
PaperTradingEngine by serializing/deserializing engine state to Redis.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.agents.schemas import ClosedTrade, Position

logger = logging.getLogger(__name__)

_ENGINE_STATE_KEY = "paper_trading:engine_state"
_POSITIONS_KEY = "paper_trading:positions"
_CLOSED_TRADES_KEY = "paper_trading:closed_trades"
_CAPITAL_KEY = "paper_trading:capital"


class PositionManager:
    """Persistent position tracking backed by Redis.

    Provides atomic reads/writes of position state so that multiple
    Celery workers can safely share the paper trading portfolio.
    """

    def __init__(self, redis_client: Any, initial_capital: float = 100_000.0) -> None:
        self._r = redis_client
        self._initial_capital = initial_capital

    # ------------------------------------------------------------------
    # Capital
    # ------------------------------------------------------------------

    def get_capital(self) -> float:
        raw = self._r.get(_CAPITAL_KEY)
        if raw is None:
            self._r.set(_CAPITAL_KEY, str(self._initial_capital))
            return self._initial_capital
        return float(raw)

    def set_capital(self, value: float) -> None:
        self._r.set(_CAPITAL_KEY, str(round(value, 2)))

    def adjust_capital(self, delta: float) -> float:
        """Atomically adjust capital by delta (can be negative for purchases)."""
        current = self.get_capital()
        new_val = current + delta
        self.set_capital(new_val)
        return new_val

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, Position]:
        raw = self._r.hgetall(_POSITIONS_KEY)
        positions: dict[str, Position] = {}
        for ticker, data in raw.items():
            ticker_str = ticker.decode() if isinstance(ticker, bytes) else ticker
            data_str = data.decode() if isinstance(data, bytes) else data
            try:
                positions[ticker_str] = Position(**json.loads(data_str))
            except Exception:
                logger.exception("Failed to deserialize position for %s", ticker_str)
        return positions

    def save_position(self, position: Position) -> None:
        self._r.hset(_POSITIONS_KEY, position.ticker, position.model_dump_json())

    def remove_position(self, ticker: str) -> None:
        self._r.hdel(_POSITIONS_KEY, ticker)

    def has_position(self, ticker: str) -> bool:
        return bool(self._r.hexists(_POSITIONS_KEY, ticker))

    # ------------------------------------------------------------------
    # Closed trades
    # ------------------------------------------------------------------

    def append_closed_trade(self, trade: ClosedTrade) -> None:
        self._r.rpush(_CLOSED_TRADES_KEY, trade.model_dump_json())

    def get_closed_trades(self, limit: int = 100) -> list[ClosedTrade]:
        raw_list = self._r.lrange(_CLOSED_TRADES_KEY, -limit, -1)
        trades: list[ClosedTrade] = []
        for raw in raw_list:
            data_str = raw.decode() if isinstance(raw, bytes) else raw
            try:
                trades.append(ClosedTrade(**json.loads(data_str)))
            except Exception:
                logger.exception("Failed to deserialize closed trade")
        return trades

    def get_closed_trade_count(self) -> int:
        return int(self._r.llen(_CLOSED_TRADES_KEY))

    # ------------------------------------------------------------------
    # Engine state snapshot (full serialization)
    # ------------------------------------------------------------------

    def save_engine_state(self, engine_state: dict[str, Any]) -> None:
        self._r.set(_ENGINE_STATE_KEY, json.dumps(engine_state, default=str))

    def load_engine_state(self) -> Optional[dict[str, Any]]:
        raw = self._r.get(_ENGINE_STATE_KEY)
        if raw is None:
            return None
        return json.loads(raw)
