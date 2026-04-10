"""Unusual Whales connector — options flow data source.

Fetches unusual options activity, put/call ratios, and IV rank.
Primary: Unusual Whales API ($57/mo) — env var UNUSUAL_WHALES_API_KEY.
Fallback: YFinance options chain for basic P/C ratio and IV.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)

_UW_BASE_URL = "https://api.unusualwhales.com/api"


class UnusualWhalesConnector:
    """Fetches options flow data for watchlist tickers.

    When UNUSUAL_WHALES_API_KEY is not set, falls back to YFinance
    options chain for basic put/call volume and ATM implied volatility.
    """

    def __init__(self) -> None:
        self._api_key = os.environ.get("UNUSUAL_WHALES_API_KEY", "")
        self._use_uw = bool(self._api_key)

    def fetch_options_flow(self, tickers: list[str]) -> list[dict]:
        """Fetch options flow data for given tickers.

        Returns:
            List of dicts with keys: ticker, call_volume, put_volume,
            put_call_ratio, unusual_activity_score, iv_rank,
            largest_trade_json, timestamp, source.
        """
        results: list[dict] = []
        for ticker in tickers:
            try:
                if self._use_uw:
                    data = self._fetch_from_uw(ticker)
                else:
                    data = self._fetch_from_yfinance(ticker)
                if data:
                    results.append(data)
            except Exception:
                logger.exception("Failed to fetch options flow for %s", ticker)
        return results

    def _fetch_from_uw(self, ticker: str) -> Optional[dict]:
        """Fetch from Unusual Whales API."""
        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            url = f"{_UW_BASE_URL}/stock/{ticker}/flow-alerts"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read())

            data = raw.get("data", {})
            return {
                "ticker": ticker.upper(),
                "call_volume": int(data.get("call_volume", 0)),
                "put_volume": int(data.get("put_volume", 0)),
                "put_call_ratio": Decimal(str(data.get("put_call_ratio", 1.0))),
                "unusual_activity_score": Decimal(str(data.get("unusual_score", 0.0))),
                "iv_rank": Decimal(str(data.get("iv_rank", 50.0))),
                "largest_trade_json": json.dumps(data.get("largest_trade", {})),
                "timestamp": datetime.now(timezone.utc),
                "source": "unusual_whales",
            }
        except Exception:
            logger.exception("Unusual Whales API error for %s", ticker)
            return None

    def _fetch_from_yfinance(self, ticker: str) -> Optional[dict]:
        """Fallback: compute basic P/C ratio from YFinance options chain."""
        try:
            import yfinance as yf
            yf_ticker = yf.Ticker(ticker)
            expirations = yf_ticker.options
            if not expirations:
                return None

            # Use nearest expiration
            exp = expirations[0]
            chain = yf_ticker.option_chain(exp)
            calls = chain.calls
            puts = chain.puts

            call_vol = int(calls["volume"].fillna(0).sum())
            put_vol = int(puts["volume"].fillna(0).sum())
            pc_ratio = round(put_vol / call_vol, 4) if call_vol > 0 else 1.0

            # ATM IV estimate (nearest strike)
            info = yf_ticker.fast_info
            current_price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", 50.0)
            calls_sorted = calls.copy()
            calls_sorted["dist"] = abs(calls_sorted["strike"] - current_price)
            atm_row = calls_sorted.nsmallest(1, "dist")
            iv = float(atm_row["impliedVolatility"].iloc[0]) * 100 if not atm_row.empty else 50.0

            # Unusual activity: volume > 3x open interest
            unusual_score = 0.0
            if not calls.empty:
                calls_unusual = calls[calls["volume"] > calls["openInterest"] * 3]
                puts_unusual = puts[puts["volume"] > puts["openInterest"] * 3]
                total_unusual = len(calls_unusual) + len(puts_unusual)
                unusual_score = min(100.0, total_unusual * 10.0)

            return {
                "ticker": ticker.upper(),
                "call_volume": call_vol,
                "put_volume": put_vol,
                "put_call_ratio": Decimal(str(round(pc_ratio, 4))),
                "unusual_activity_score": Decimal(str(round(unusual_score, 2))),
                "iv_rank": Decimal(str(round(iv, 2))),
                "largest_trade_json": None,
                "timestamp": datetime.now(timezone.utc),
                "source": "yfinance_options",
            }
        except Exception:
            logger.exception("YFinance options fallback failed for %s", ticker)
            return None
