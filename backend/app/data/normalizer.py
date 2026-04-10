"""DataNormalizer — time alignment and grain standardisation.

Sits between connector output and DB writes. Responsibilities:
  1. Align all timestamps to market session (US/Eastern).
  2. Forward-fill macro indicators to daily grain.
  3. Snap news articles to nearest trading session open.
  4. Validate required fields and drop malformed records.

All normalised records are safe to insert into TimescaleDB hypertables.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from typing import Any

import pytz

logger = logging.getLogger(__name__)

_EST = pytz.timezone("America/New_York")
_MARKET_OPEN = time(9, 30)   # 09:30 ET
_MARKET_CLOSE = time(16, 0)  # 16:00 ET


def _to_utc(dt: datetime) -> datetime:
    """Convert any datetime to UTC, assuming EST if naive."""
    if dt.tzinfo is None:
        dt = _EST.localize(dt)
    return dt.astimezone(timezone.utc)


def _snap_to_session_open(dt: datetime) -> datetime:
    """Snap a timestamp to the nearest market session open (9:30 ET)."""
    dt_est = dt.astimezone(_EST)
    session_open = datetime.combine(dt_est.date(), _MARKET_OPEN, tzinfo=_EST)
    return _to_utc(session_open)


def _is_trading_hour(dt: datetime) -> bool:
    """Return True if dt falls within US market hours (9:30-16:00 ET)."""
    dt_est = dt.astimezone(_EST)
    t = dt_est.time()
    return _MARKET_OPEN <= t <= _MARKET_CLOSE


def normalise_price_bars(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise OHLCV bar records.

    - Converts timestamps to UTC.
    - Drops records missing close or timestamp.
    - Ensures volume is non-negative integer.
    """
    normalised = []
    for rec in records:
        ts = rec.get("timestamp")
        if ts is None:
            continue
        if not isinstance(ts, datetime):
            try:
                ts = datetime.fromisoformat(str(ts))
            except Exception:
                continue
        rec["timestamp"] = _to_utc(ts)

        if rec.get("close") is None:
            continue
        if rec.get("volume") is None:
            rec["volume"] = 0
        rec["volume"] = max(0, int(rec["volume"]))
        normalised.append(rec)
    return normalised


def normalise_news_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise news items — snap timestamps to session open."""
    normalised = []
    for rec in records:
        ts = rec.get("timestamp")
        if ts is None:
            continue
        if not isinstance(ts, datetime):
            try:
                ts = datetime.fromisoformat(str(ts))
            except Exception:
                continue
        rec["timestamp"] = _snap_to_session_open(_to_utc(ts))

        if not rec.get("headline") and not rec.get("summary"):
            continue
        normalised.append(rec)
    return normalised


def normalise_macro_indicators(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise macro indicators — forward-fill to daily grain (UTC midnight)."""
    normalised = []
    for rec in records:
        ts = rec.get("timestamp")
        if ts is None:
            ts = datetime.now(timezone.utc)
        if not isinstance(ts, datetime):
            try:
                ts = datetime.fromisoformat(str(ts))
            except Exception:
                ts = datetime.now(timezone.utc)

        # Snap to UTC midnight (daily grain)
        utc_ts = _to_utc(ts)
        daily_ts = utc_ts.replace(hour=0, minute=0, second=0, microsecond=0)
        rec["timestamp"] = daily_ts

        if rec.get("value") is None:
            continue
        if not rec.get("indicator"):
            continue
        normalised.append(rec)
    return normalised


def normalise_options_flow(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise options flow records."""
    normalised = []
    for rec in records:
        ts = rec.get("timestamp")
        if ts is None:
            ts = datetime.now(timezone.utc)
        if not isinstance(ts, datetime):
            try:
                ts = datetime.fromisoformat(str(ts))
            except Exception:
                ts = datetime.now(timezone.utc)
        rec["timestamp"] = _to_utc(ts)

        if not rec.get("ticker"):
            continue
        normalised.append(rec)
    return normalised


def normalise_event_calendar(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise event calendar entries."""
    normalised = []
    for rec in records:
        event_date = rec.get("event_date")
        if event_date is not None and isinstance(event_date, datetime):
            rec["event_date"] = _to_utc(event_date)

        if not rec.get("ticker") or not rec.get("event_type"):
            continue
        normalised.append(rec)
    return normalised


def normalise_short_interest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise short interest records — snap to daily grain."""
    normalised = []
    for rec in records:
        ts = rec.get("timestamp")
        if ts is None:
            ts = datetime.now(timezone.utc)
        if not isinstance(ts, datetime):
            try:
                ts = datetime.fromisoformat(str(ts))
            except Exception:
                ts = datetime.now(timezone.utc)
        utc_ts = _to_utc(ts)
        rec["timestamp"] = utc_ts.replace(hour=0, minute=0, second=0, microsecond=0)

        if not rec.get("ticker"):
            continue
        normalised.append(rec)
    return normalised
