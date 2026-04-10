"""V2 data tables: macro_indicators, options_flow, event_calendar, short_interest

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-09

New tables for Layer 1 (Data Layer) V2:
  - macro_indicators: VIX, DXY, US10Y, US2Y, FEDFUNDS, credit_spread
  - options_flow: IV rank, put/call ratio, unusual activity
  - event_calendar: FDA PDUFA, earnings, conference events
  - short_interest: SI%, days-to-cover, borrow rate
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # macro_indicators hypertable
    # ------------------------------------------------------------------
    op.create_table(
        "macro_indicators",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("indicator", sa.String(30), nullable=False),
        sa.Column("value", sa.Numeric(), nullable=True),
        sa.Column("change_1d", sa.Numeric(), nullable=True),
        sa.Column("change_5d", sa.Numeric(), nullable=True),
        sa.Column("source", sa.String(50), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "indicator"),
    )
    op.execute(
        "SELECT create_hypertable('macro_indicators', 'timestamp', "
        "chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE)"
    )
    op.create_index(
        "ix_macro_indicators_indicator_timestamp",
        "macro_indicators",
        ["indicator", "timestamp"],
    )

    # ------------------------------------------------------------------
    # options_flow hypertable
    # ------------------------------------------------------------------
    op.create_table(
        "options_flow",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("call_volume", sa.BigInteger(), nullable=True),
        sa.Column("put_volume", sa.BigInteger(), nullable=True),
        sa.Column("put_call_ratio", sa.Numeric(), nullable=True),
        sa.Column("unusual_activity_score", sa.Numeric(), nullable=True),
        sa.Column("iv_rank", sa.Numeric(), nullable=True),
        sa.Column("largest_trade_json", sa.Text(), nullable=True),
        sa.Column("source", sa.String(50), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "ticker"),
    )
    op.execute(
        "SELECT create_hypertable('options_flow', 'timestamp', "
        "chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE)"
    )
    op.create_index(
        "ix_options_flow_ticker_timestamp",
        "options_flow",
        ["ticker", "timestamp"],
    )

    # ------------------------------------------------------------------
    # event_calendar standard table (not hypertable — event dates vary)
    # ------------------------------------------------------------------
    op.create_table(
        "event_calendar",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("event_type", sa.String(30), nullable=False),
        sa.Column("event_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("impact_estimate", sa.String(10), nullable=True),
        sa.Column("binary_outcome", sa.Boolean(), nullable=False, default=False),
        sa.Column("source", sa.String(50), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_event_calendar_ticker_event_date",
        "event_calendar",
        ["ticker", "event_date"],
    )
    op.create_index(
        "ix_event_calendar_event_type",
        "event_calendar",
        ["event_type"],
    )

    # ------------------------------------------------------------------
    # short_interest hypertable
    # ------------------------------------------------------------------
    op.create_table(
        "short_interest",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("si_pct", sa.Numeric(), nullable=True),
        sa.Column("days_to_cover", sa.Numeric(), nullable=True),
        sa.Column("borrow_rate", sa.Numeric(), nullable=True),
        sa.Column("shares_short", sa.BigInteger(), nullable=True),
        sa.Column("source", sa.String(50), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "ticker"),
    )
    op.execute(
        "SELECT create_hypertable('short_interest', 'timestamp', "
        "chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE)"
    )
    op.create_index(
        "ix_short_interest_ticker_timestamp",
        "short_interest",
        ["ticker", "timestamp"],
    )


def downgrade() -> None:
    op.drop_table("short_interest")
    op.drop_table("event_calendar")
    op.drop_table("options_flow")
    op.drop_table("macro_indicators")
