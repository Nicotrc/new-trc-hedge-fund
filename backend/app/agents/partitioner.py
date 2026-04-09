"""Data partitioner: enforces information asymmetry across quant agents.

V2: Extended to 8 data categories to support the redesigned quant agent suite
(Momentum, Value, Event, Macro, Risk).

Access matrix:
  Data Type          Momentum  Value  Event  Macro  Risk
  ----------------------------------------------------------------
  price_action          Y        -      -      Y      Y
  fundamentals          -        Y      -      -      Y
  insider_trades        -        Y      -      -      Y
  news                  -        -      Y      Y      Y
  macro_indicators      -        -      -      Y      Y
  options_flow          -        -      Y      -      Y
  event_calendar        -        -      Y      -      Y
  short_interest        Y        -      -      -      Y
  ----------------------------------------------------------------

Risk Agent has FULL ACCESS (deliberate asymmetry break).
No two agents (except Risk) see the same data combination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.schemas.financial import FinancialSnapshot

# ---------------------------------------------------------------------------
# V1 data type taxonomy (kept for backward compat with persona agents)
# ---------------------------------------------------------------------------

DATA_TYPE_FUNDAMENTALS = "fundamentals"
DATA_TYPE_PRICE_ACTION = "price_action"
DATA_TYPE_NEWS = "news"
DATA_TYPE_INSIDER_TRADES = "insider_trades"

# ---------------------------------------------------------------------------
# V2 extended data type taxonomy (8 categories)
# ---------------------------------------------------------------------------

DATA_TYPE_MACRO_INDICATORS = "macro_indicators"
DATA_TYPE_OPTIONS_FLOW = "options_flow"
DATA_TYPE_EVENT_CALENDAR = "event_calendar"
DATA_TYPE_SHORT_INTEREST = "short_interest"

# ---------------------------------------------------------------------------
# V1 Persona access control matrix (legacy — kept for committee.py compat)
# ---------------------------------------------------------------------------

PERSONA_DATA_ACCESS: dict[str, frozenset[str]] = {
    "buffett": frozenset({DATA_TYPE_FUNDAMENTALS}),
    "munger": frozenset({DATA_TYPE_FUNDAMENTALS, DATA_TYPE_NEWS}),
    "ackman": frozenset({DATA_TYPE_FUNDAMENTALS, DATA_TYPE_INSIDER_TRADES}),
    "cohen": frozenset({DATA_TYPE_PRICE_ACTION}),
    "dalio": frozenset({DATA_TYPE_PRICE_ACTION, DATA_TYPE_NEWS}),
}

# ---------------------------------------------------------------------------
# V2 Quant agent access control matrix
# ---------------------------------------------------------------------------

QUANT_AGENT_DATA_ACCESS: dict[str, frozenset[str]] = {
    "momentum": frozenset({DATA_TYPE_PRICE_ACTION, DATA_TYPE_SHORT_INTEREST}),
    "value": frozenset({DATA_TYPE_FUNDAMENTALS, DATA_TYPE_INSIDER_TRADES}),
    "event": frozenset({DATA_TYPE_NEWS, DATA_TYPE_OPTIONS_FLOW, DATA_TYPE_EVENT_CALENDAR}),
    "macro": frozenset({DATA_TYPE_MACRO_INDICATORS, DATA_TYPE_PRICE_ACTION, DATA_TYPE_NEWS}),
    "risk": frozenset({  # Full access — deliberate asymmetry break
        DATA_TYPE_PRICE_ACTION,
        DATA_TYPE_FUNDAMENTALS,
        DATA_TYPE_INSIDER_TRADES,
        DATA_TYPE_NEWS,
        DATA_TYPE_MACRO_INDICATORS,
        DATA_TYPE_OPTIONS_FLOW,
        DATA_TYPE_EVENT_CALENDAR,
        DATA_TYPE_SHORT_INTEREST,
    }),
}

# All recognised data type keys (V2)
_ALL_DATA_TYPES_V2: frozenset[str] = frozenset({
    DATA_TYPE_PRICE_ACTION,
    DATA_TYPE_FUNDAMENTALS,
    DATA_TYPE_INSIDER_TRADES,
    DATA_TYPE_NEWS,
    DATA_TYPE_MACRO_INDICATORS,
    DATA_TYPE_OPTIONS_FLOW,
    DATA_TYPE_EVENT_CALENDAR,
    DATA_TYPE_SHORT_INTEREST,
})


class DataPartitioner:
    """Filters a financial data context to only the fields a given agent may see.

    Supports both V1 persona names and V2 quant agent names.

    Usage (V2)::

        partitioner = DataPartitioner()
        context = partitioner.partition_for_quant_agent("momentum", full_data)
        # context contains only price_action + short_interest

    Usage (V1 legacy)::

        context = partitioner.partition_raw("buffett", data)
    """

    # All recognised top-level data keys (V1 — for backward compat)
    _ALL_DATA_TYPES: frozenset[str] = frozenset({
        DATA_TYPE_FUNDAMENTALS,
        DATA_TYPE_PRICE_ACTION,
        DATA_TYPE_NEWS,
        DATA_TYPE_INSIDER_TRADES,
    })

    def __init__(self) -> None:
        self._access = PERSONA_DATA_ACCESS  # V1
        self._quant_access = QUANT_AGENT_DATA_ACCESS  # V2

    # ------------------------------------------------------------------
    # V1 API (legacy — kept for backward compat)
    # ------------------------------------------------------------------

    def get_allowed_types(self, persona_name: str) -> frozenset[str]:
        """Return the set of data types a V1 persona may access."""
        name = persona_name.lower().strip()
        if name not in self._access:
            raise ValueError(
                f"Unknown persona '{name}'. "
                f"Valid options: {sorted(self._access.keys())}"
            )
        return self._access[name]

    def partition_for_persona(
        self,
        persona_name: str,
        snapshots: list[Any],
    ) -> dict[str, Any]:
        """Build a V1 persona-specific data context from a list of snapshots."""
        allowed = self.get_allowed_types(persona_name)
        restricted = self._ALL_DATA_TYPES - allowed

        partitioned: list[dict[str, Any]] = []
        for snap in snapshots:
            if hasattr(snap, "model_dump"):
                snap_dict: dict[str, Any] = snap.model_dump()
            elif hasattr(snap, "dict"):
                snap_dict = snap.dict()
            else:
                snap_dict = dict(snap)

            filtered = {k: v for k, v in snap_dict.items() if k not in restricted}
            partitioned.append(filtered)

        return {
            "persona": persona_name,
            "allowed_data_types": sorted(allowed),
            "snapshots": partitioned,
        }

    def partition_raw(
        self,
        persona_name: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Partition a raw dict for a V1 persona."""
        allowed = self.get_allowed_types(persona_name)
        restricted = self._ALL_DATA_TYPES - allowed
        return {k: v for k, v in data.items() if k not in restricted}

    # ------------------------------------------------------------------
    # V2 API (quant agents)
    # ------------------------------------------------------------------

    def get_allowed_types_v2(self, agent_id: str) -> frozenset[str]:
        """Return the set of data types a V2 quant agent may access."""
        name = agent_id.lower().strip()
        if name not in self._quant_access:
            raise ValueError(
                f"Unknown quant agent '{name}'. "
                f"Valid options: {sorted(self._quant_access.keys())}"
            )
        return self._quant_access[name]

    def partition_for_quant_agent(
        self,
        agent_id: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Partition a full data dict for a V2 quant agent.

        Args:
            agent_id: One of {"momentum", "value", "event", "macro", "risk"}.
            data: Dict whose top-level keys are V2 data-type labels.

        Returns:
            Filtered copy of *data* with only permitted keys, plus metadata.
        """
        allowed = self.get_allowed_types_v2(agent_id)
        restricted = _ALL_DATA_TYPES_V2 - allowed
        filtered = {k: v for k, v in data.items() if k not in restricted}
        return {
            "agent_id": agent_id,
            "allowed_data_types": sorted(allowed),
            **filtered,
        }
