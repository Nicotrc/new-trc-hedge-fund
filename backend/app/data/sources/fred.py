"""FRED API connector — macro indicators data source.

Fetches: VIX (via YFinance ^VIX), DXY (DX-Y.NYB), US10Y (^TNX),
US2Y (^IRX), and HY credit spreads proxy.

FRED API is free (no key required for public series).
Fallback to YFinance for market-traded proxies.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)

# Macro indicator ticker map: indicator_name -> yfinance ticker
_YFINANCE_MACRO_TICKERS = {
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
    "US2Y": "^IRX",
    "SP500": "^GSPC",
    "HY_SPREAD_PROXY": "HYG",  # iShares HY Corporate Bond ETF as spread proxy
}

# FRED series IDs (requires FRED_API_KEY env var)
_FRED_SERIES = {
    "FEDFUNDS": "FEDFUNDS",
    "DGS10": "DGS10",   # 10-year Treasury
    "DGS2": "DGS2",     # 2-year Treasury
    "BAMLH0A0HYM2": "BAMLH0A0HYM2",  # HY credit spread
}


class FREDConnector:
    """Fetches macro indicator data from FRED API and YFinance fallbacks.

    Indicators produced:
      VIX, DXY, US10Y, US2Y, FEDFUNDS, credit_spread (HY OAS proxy)
    """

    def __init__(self) -> None:
        self._fred_api_key = os.environ.get("FRED_API_KEY", "")

    def fetch_macro_indicators(
        self,
        lookback_days: int = 30,
    ) -> list[dict]:
        """Fetch current macro indicator values.

        Returns:
            List of dicts with keys: indicator, value, change_1d, change_5d,
            timestamp, source.
        """
        results: list[dict] = []
        now = datetime.now(timezone.utc)

        # Try FRED API first, fall back to YFinance
        for indicator, yticker in _YFINANCE_MACRO_TICKERS.items():
            try:
                data = self._fetch_yfinance(yticker, lookback_days)
                if data:
                    results.append({
                        "indicator": indicator,
                        "value": data["current"],
                        "change_1d": data.get("change_1d"),
                        "change_5d": data.get("change_5d"),
                        "timestamp": now,
                        "source": "yfinance",
                    })
            except Exception:
                logger.exception("Failed to fetch macro indicator %s (%s)", indicator, yticker)

        # FRED API for FEDFUNDS if key available
        if self._fred_api_key:
            try:
                fedfunds = self._fetch_fred("FEDFUNDS")
                if fedfunds is not None:
                    results.append({
                        "indicator": "FEDFUNDS",
                        "value": Decimal(str(fedfunds)),
                        "change_1d": None,
                        "change_5d": None,
                        "timestamp": now,
                        "source": "fred",
                    })
            except Exception:
                logger.exception("Failed to fetch FEDFUNDS from FRED")

        return results

    def _fetch_yfinance(self, ticker: str, lookback_days: int) -> Optional[dict]:
        """Fetch historical data from YFinance and compute changes."""
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period=f"{lookback_days}d")
            if hist.empty:
                return None
            closes = hist["Close"].dropna()
            if len(closes) < 2:
                return None
            current = float(closes.iloc[-1])
            prev_1d = float(closes.iloc[-2]) if len(closes) >= 2 else current
            prev_5d = float(closes.iloc[-6]) if len(closes) >= 6 else prev_1d
            return {
                "current": Decimal(str(round(current, 4))),
                "change_1d": Decimal(str(round(current - prev_1d, 4))),
                "change_5d": Decimal(str(round(current - prev_5d, 4))),
            }
        except Exception:
            logger.exception("YFinance fetch failed for %s", ticker)
            return None

    def _fetch_fred(self, series_id: str) -> Optional[float]:
        """Fetch latest value from FRED API."""
        try:
            import urllib.request
            import json as _json
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}&api_key={self._fred_api_key}"
                f"&file_type=json&sort_order=desc&limit=1"
            )
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = _json.loads(resp.read())
            obs = data.get("observations", [])
            if obs and obs[0].get("value") not in (".", None):
                return float(obs[0]["value"])
        except Exception:
            logger.exception("FRED API fetch failed for %s", series_id)
        return None

    def compute_yield_curve_spread(self, indicators: list[dict]) -> Optional[Decimal]:
        """Compute 10Y-2Y yield spread from fetched indicators."""
        us10y = next((i["value"] for i in indicators if i["indicator"] == "US10Y"), None)
        us2y = next((i["value"] for i in indicators if i["indicator"] == "US2Y"), None)
        if us10y is not None and us2y is not None:
            try:
                return Decimal(str(float(us10y) - float(us2y)))
            except Exception:
                pass
        return None
