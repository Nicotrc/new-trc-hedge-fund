"""Celery task: ingest options flow data every hour."""

from __future__ import annotations

import logging
import os

from app.data.normalizer import normalise_options_flow
from app.data.sources.unusual_whales import UnusualWhalesConnector
from app.db.engine import SyncSessionLocal
from app.db.models import OptionsFlow
from app.tasks.celery_app import app

logger = logging.getLogger(__name__)


@app.task(name="app.tasks.ingest_options.run", bind=True, max_retries=3)
def run(self) -> dict:
    """Fetch and persist options flow data for the watchlist."""
    watchlist = [
        t.strip().upper()
        for t in os.environ.get("WATCHLIST", "").split(",")
        if t.strip()
    ]
    if not watchlist:
        return {"persisted": 0, "reason": "empty_watchlist"}

    connector = UnusualWhalesConnector()
    try:
        raw_records = connector.fetch_options_flow(watchlist)
    except Exception as exc:
        logger.exception("Options flow fetch failed")
        raise self.retry(exc=exc, countdown=120)

    normalised = normalise_options_flow(raw_records)
    if not normalised:
        logger.warning("No options flow records to persist")
        return {"persisted": 0}

    persisted = 0
    with SyncSessionLocal() as session:
        for rec in normalised:
            try:
                row = OptionsFlow(
                    timestamp=rec["timestamp"],
                    ticker=rec["ticker"],
                    call_volume=rec.get("call_volume"),
                    put_volume=rec.get("put_volume"),
                    put_call_ratio=rec.get("put_call_ratio"),
                    unusual_activity_score=rec.get("unusual_activity_score"),
                    iv_rank=rec.get("iv_rank"),
                    largest_trade_json=rec.get("largest_trade_json"),
                    source=rec.get("source", "options"),
                )
                session.merge(row)
                persisted += 1
            except Exception:
                logger.exception("Failed to persist options row: %s", rec.get("ticker"))
        session.commit()

    logger.info("Options flow ingestion complete: %d records", persisted)
    return {"persisted": persisted}
