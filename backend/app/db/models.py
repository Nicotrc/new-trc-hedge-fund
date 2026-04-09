from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import BigInteger, Boolean, DateTime, Integer, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ===========================================================================
# V1 tables (existing — kept unchanged)
# ===========================================================================


class PriceOHLCV(Base):
    """OHLCV price data — TimescaleDB hypertable on timestamp."""

    __tablename__ = "price_ohlcv"
    __table_args__ = {
        "timescaledb_hypertable": {"time_column_name": "timestamp"},
    }

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    open: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    high: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    low: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    close: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    volume: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


class Fundamentals(Base):
    """Fundamental financial data — TimescaleDB hypertable on timestamp."""

    __tablename__ = "fundamentals"
    __table_args__ = {
        "timescaledb_hypertable": {"time_column_name": "timestamp"},
    }

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    pe_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    revenue: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    net_income: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    eps: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    debt_to_equity: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    free_cash_flow: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    market_cap: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


class InsiderTrade(Base):
    """Insider trading records — TimescaleDB hypertable on timestamp."""

    __tablename__ = "insider_trades"
    __table_args__ = {
        "timescaledb_hypertable": {"time_column_name": "timestamp"},
    }

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    insider_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    trade_type: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    shares: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    trade_value: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


class NewsItem(Base):
    """News items with sentiment — TimescaleDB hypertable on timestamp."""

    __tablename__ = "news_items"
    __table_args__ = {
        "timescaledb_hypertable": {"time_column_name": "timestamp"},
    }

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    headline: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    article_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


class DetectedSignal(Base):
    """Detected trading signals — TimescaleDB hypertable on detected_at."""

    __tablename__ = "detected_signals"
    __table_args__ = {
        "timescaledb_hypertable": {"time_column_name": "detected_at"},
    }

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    signal_type: Mapped[str] = mapped_column(String(50), primary_key=True)
    score: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    composite_score: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    passed_gate: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(
        String(50), nullable=False, default="scanner"
    )


class AgentVerdictRecord(Base):
    """Per-agent verdict — TimescaleDB hypertable on analysed_at.

    V2: supports both V1 persona names and V2 quant agent IDs via the
    `agent_version` column ("v1" | "v2").
    """

    __tablename__ = "agent_verdicts"
    __table_args__ = {
        "timescaledb_hypertable": {"time_column_name": "analysed_at"},
    }

    analysed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    opportunity_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    persona: Mapped[str] = mapped_column(String(20), primary_key=True)
    verdict: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)
    verdict_json: Mapped[str] = mapped_column(Text, nullable=False)


class CIODecisionRecord(Base):
    """CIO-level investment decision — TimescaleDB hypertable on decided_at.

    V2: decision_json may contain a CIODecisionV2 payload (with MC results,
    Kelly sizing, scenarios) when agent_version='v2'.
    """

    __tablename__ = "cio_decisions"
    __table_args__ = {
        "timescaledb_hypertable": {"time_column_name": "decided_at"},
    }

    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    opportunity_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    conviction_score: Mapped[int] = mapped_column(Integer, nullable=False)
    suggested_allocation_pct: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    final_verdict: Mapped[str] = mapped_column(String(10), nullable=False)
    decision_json: Mapped[str] = mapped_column(Text, nullable=False)


# ===========================================================================
# V2 tables (new — migration 0004)
# ===========================================================================


class MacroIndicator(Base):
    """Macro market indicators — TimescaleDB hypertable on timestamp.

    Indicators: VIX, DXY, US10Y, US2Y, FEDFUNDS, credit_spread.
    """

    __tablename__ = "macro_indicators"
    __table_args__ = {
        "timescaledb_hypertable": {"time_column_name": "timestamp"},
    }

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    indicator: Mapped[str] = mapped_column(String(30), primary_key=True)
    value: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    change_1d: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    change_5d: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


class OptionsFlow(Base):
    """Options flow data — TimescaleDB hypertable on timestamp."""

    __tablename__ = "options_flow"
    __table_args__ = {
        "timescaledb_hypertable": {"time_column_name": "timestamp"},
    }

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    call_volume: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    put_volume: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    put_call_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    unusual_activity_score: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    iv_rank: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    largest_trade_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


class EventCalendar(Base):
    """Upcoming event calendar — standard table (not hypertable)."""

    __tablename__ = "event_calendar"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    event_type: Mapped[str] = mapped_column(String(30), nullable=False)
    event_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    impact_estimate: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    binary_outcome: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


class ShortInterest(Base):
    """Short interest data — TimescaleDB hypertable on timestamp."""

    __tablename__ = "short_interest"
    __table_args__ = {
        "timescaledb_hypertable": {"time_column_name": "timestamp"},
    }

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    si_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    days_to_cover: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    borrow_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    shares_short: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
