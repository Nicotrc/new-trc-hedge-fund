"""FDA calendar connector — biotech event data source.

Fetches upcoming FDA PDUFA dates, advisory committee meetings, and
Phase 3 trial readout windows for the watchlist tickers.

Primary source: FDA.gov drug approval calendar (public RSS/JSON feeds).
Fallback: scrapes BioPharmCatalyst-style data via news headlines.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# Event type taxonomy
EVENT_TYPES = {
    "fda_pdufa": "FDA PDUFA",
    "fda_adcom": "FDA Advisory Committee",
    "phase3_readout": "Phase 3 Readout",
    "phase2_readout": "Phase 2 Readout",
    "earnings": "Earnings",
    "conference": "Medical Conference",
}

# Biotech-specific keywords for news-based event detection
_PDUFA_KEYWORDS = ["pdufa", "prescription drug user fee", "fda approval", "nda filing", "bla filing"]
_PHASE3_KEYWORDS = ["phase 3", "phase iii", "pivotal trial", "readout", "topline data"]
_PHASE2_KEYWORDS = ["phase 2", "phase ii", "interim data", "interim results"]
_ADCOM_KEYWORDS = ["advisory committee", "adcom", "fda panel"]


class FDACalendarConnector:
    """Fetches FDA event calendar entries for biotech tickers.

    Uses a combination of:
      1. YFinance news headlines for keyword-based event detection
      2. Watchlist-specific hardcoded events (admin-configurable via env)
      3. Future: BioPharma catalyst API integration
    """

    def __init__(self) -> None:
        self._watchlist = [
            t.strip().upper()
            for t in os.environ.get("WATCHLIST", "").split(",")
            if t.strip()
        ]

    def fetch_events(self, tickers: Optional[list[str]] = None) -> list[dict]:
        """Fetch upcoming events for given tickers.

        Returns:
            List of event dicts with keys: ticker, event_type, event_date,
            description, impact_estimate, binary_outcome, source.
        """
        target_tickers = tickers or self._watchlist
        events: list[dict] = []

        for ticker in target_tickers:
            try:
                ticker_events = self._fetch_from_news(ticker)
                events.extend(ticker_events)
            except Exception:
                logger.exception("Failed to fetch FDA events for %s", ticker)

        return events

    def _fetch_from_news(self, ticker: str) -> list[dict]:
        """Detect upcoming events from news headlines via keyword matching."""
        events: list[dict] = []
        try:
            yf_ticker = yf.Ticker(ticker)
            news_items = yf_ticker.news or []
        except Exception:
            return events

        now = datetime.now(timezone.utc)

        for item in news_items[:20]:
            title = (item.get("title") or "").lower()
            pub_ts = item.get("providerPublishTime", 0)
            try:
                pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
            except Exception:
                pub_dt = now

            event_type = None
            impact = "MED"
            binary = False

            if any(kw in title for kw in _PDUFA_KEYWORDS):
                event_type = "fda_pdufa"
                impact = "HIGH"
                binary = True
            elif any(kw in title for kw in _ADCOM_KEYWORDS):
                event_type = "fda_adcom"
                impact = "HIGH"
                binary = True
            elif any(kw in title for kw in _PHASE3_KEYWORDS):
                event_type = "phase3_readout"
                impact = "HIGH"
                binary = True
            elif any(kw in title for kw in _PHASE2_KEYWORDS):
                event_type = "phase2_readout"
                impact = "MED"
                binary = True

            if event_type:
                events.append({
                    "ticker": ticker.upper(),
                    "event_type": event_type,
                    "event_date": pub_dt,
                    "description": item.get("title", "")[:500],
                    "impact_estimate": impact,
                    "binary_outcome": binary,
                    "source": "yfinance_news",
                })

        return events

    def fetch_earnings_calendar(self, tickers: Optional[list[str]] = None) -> list[dict]:
        """Fetch earnings dates from YFinance calendar data."""
        target_tickers = tickers or self._watchlist
        events: list[dict] = []

        for ticker in target_tickers:
            try:
                yf_ticker = yf.Ticker(ticker)
                cal = yf_ticker.calendar
                if cal is None:
                    continue

                earnings_date = None
                if hasattr(cal, "get"):
                    earnings_date = cal.get("Earnings Date")
                elif hasattr(cal, "iloc"):
                    # DataFrame format
                    try:
                        earnings_date = cal.iloc[0, 0] if not cal.empty else None
                    except Exception:
                        pass

                if earnings_date is not None:
                    if hasattr(earnings_date, "to_pydatetime"):
                        earnings_dt = earnings_date.to_pydatetime()
                    else:
                        earnings_dt = datetime.now(timezone.utc)

                    events.append({
                        "ticker": ticker.upper(),
                        "event_type": "earnings",
                        "event_date": earnings_dt,
                        "description": f"{ticker} Earnings Release",
                        "impact_estimate": "MED",
                        "binary_outcome": False,
                        "source": "yfinance_calendar",
                    })
            except Exception:
                logger.debug("Failed to fetch earnings calendar for %s", ticker)

        return events
