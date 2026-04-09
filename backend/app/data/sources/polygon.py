"""Polygon.io connector — intraday OHLCV and short interest data source.

Fetches 1m/5m bars, pre-market data, and short interest.
Requires POLYGON_API_KEY env var ($29/mo starter plan).
Falls back to YFinance 1d bars when key is not set.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)

_POLYGON_BASE = "https://api.polygon.io"


class PolygonConnector:
    """Fetches intraday price bars and short interest from Polygon.io.

    When POLYGON_API_KEY is not set, falls back to YFinance for
    daily bars (no intraday available in free tier).
    """

    def __init__(self) -> None:
        self._api_key = os.environ.get("POLYGON_API_KEY", "")
        self._use_polygon = bool(self._api_key)

    def fetch_intraday_bars(
        self,
        ticker: str,
        multiplier: int = 5,
        timespan: str = "minute",
        days_back: int = 2,
    ) -> list[dict]:
        """Fetch intraday OHLCV bars.

        Args:
            ticker: Asset ticker.
            multiplier: Bar multiplier (5 = 5-minute bars).
            timespan: "minute" | "hour".
            days_back: Number of trading days of history to fetch.

        Returns:
            List of OHLCV bar dicts with timestamp, open, high, low, close, volume.
        """
        if self._use_polygon:
            return self._polygon_aggs(ticker, multiplier, timespan, days_back)
        return self._yfinance_fallback(ticker, days_back)

    def fetch_short_interest(self, tickers: list[str]) -> list[dict]:
        """Fetch short interest data for given tickers.

        Returns basic float/short interest from YFinance info as fallback.
        """
        results: list[dict] = []
        now = datetime.now(timezone.utc)

        for ticker in tickers:
            try:
                import yfinance as yf
                info = yf.Ticker(ticker).info
                if not info:
                    continue

                shares_outstanding = info.get("sharesOutstanding", 0) or 0
                shares_short = info.get("sharesShort", 0) or 0
                days_to_cover = info.get("shortRatio", None)
                borrow_rate = None  # not available in YFinance

                si_pct = (
                    round((shares_short / shares_outstanding) * 100, 2)
                    if shares_outstanding > 0
                    else None
                )

                results.append({
                    "ticker": ticker.upper(),
                    "si_pct": Decimal(str(si_pct)) if si_pct is not None else None,
                    "days_to_cover": Decimal(str(days_to_cover)) if days_to_cover else None,
                    "borrow_rate": borrow_rate,
                    "shares_short": int(shares_short),
                    "timestamp": now,
                    "source": "yfinance_info",
                })
            except Exception:
                logger.exception("Failed to fetch short interest for %s", ticker)

        return results

    def _polygon_aggs(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        days_back: int,
    ) -> list[dict]:
        """Fetch aggregates from Polygon REST API."""
        try:
            to_dt = datetime.now(timezone.utc)
            from_dt = to_dt - timedelta(days=days_back + 3)  # +3 for weekends
            from_str = from_dt.strftime("%Y-%m-%d")
            to_str = to_dt.strftime("%Y-%m-%d")

            url = (
                f"{_POLYGON_BASE}/v2/aggs/ticker/{ticker.upper()}/range"
                f"/{multiplier}/{timespan}/{from_str}/{to_str}"
                f"?adjusted=true&sort=asc&limit=1000&apiKey={self._api_key}"
            )
            with urllib.request.urlopen(url, timeout=15) as resp:
                raw = json.loads(resp.read())

            bars = []
            for result in raw.get("results", []):
                bars.append({
                    "timestamp": datetime.fromtimestamp(result["t"] / 1000, tz=timezone.utc),
                    "ticker": ticker.upper(),
                    "open": Decimal(str(result.get("o", 0))),
                    "high": Decimal(str(result.get("h", 0))),
                    "low": Decimal(str(result.get("l", 0))),
                    "close": Decimal(str(result.get("c", 0))),
                    "volume": int(result.get("v", 0)),
                    "source": "polygon",
                })
            return bars
        except Exception:
            logger.exception("Polygon API error for %s — falling back to YFinance", ticker)
            return self._yfinance_fallback(ticker, days_back)

    def _yfinance_fallback(self, ticker: str, days_back: int) -> list[dict]:
        """Fetch daily bars from YFinance as intraday fallback."""
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period=f"{max(days_back, 5)}d")
            if hist.empty:
                return []
            bars = []
            for ts, row in hist.iterrows():
                dt = ts.to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                bars.append({
                    "timestamp": dt,
                    "ticker": ticker.upper(),
                    "open": Decimal(str(round(float(row["Open"]), 4))),
                    "high": Decimal(str(round(float(row["High"]), 4))),
                    "low": Decimal(str(round(float(row["Low"]), 4))),
                    "close": Decimal(str(round(float(row["Close"]), 4))),
                    "volume": int(row["Volume"]),
                    "source": "yfinance_daily",
                })
            return bars
        except Exception:
            logger.exception("YFinance fallback failed for %s", ticker)
            return []
