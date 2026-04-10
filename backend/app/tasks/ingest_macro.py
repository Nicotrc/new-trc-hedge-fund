"""Celery task: ingest macro indicators every 6 hours."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.data.normalizer import normalise_macro_indicators
from app.data.sources.fred import FREDConnector
from app.db.engine import SyncSessionLocal
from app.db.models import MacroIndicator
from app.tasks.celery_app import app

logger = logging.getLogger(__name__)


@app.task(name="app.tasks.ingest_macro.run", bind=True, max_retries=3)
def run(self) -> dict:
    """Fetch and persist macro indicator data (VIX, DXY, yields)."""
    connector = FREDConnector()
    try:
        raw_records = connector.fetch_macro_indicators(lookback_days=30)
    except Exception as exc:
        logger.exception("Macro ingestion fetch failed")
        raise self.retry(exc=exc, countdown=300)

    normalised = normalise_macro_indicators(raw_records)
    if not normalised:
        logger.warning("No macro indicator records to persist")
        return {"persisted": 0}

    persisted = 0
    with SyncSessionLocal() as session:
        for rec in normalised:
            try:
                row = MacroIndicator(
                    timestamp=rec["timestamp"],
                    indicator=rec["indicator"],
                    value=rec.get("value"),
                    change_1d=rec.get("change_1d"),
                    change_5d=rec.get("change_5d"),
                    source=rec.get("source", "fred"),
                )
                session.merge(row)
                persisted += 1
            except Exception:
                logger.exception("Failed to persist macro row: %s", rec.get("indicator"))
        session.commit()

    logger.info("Macro ingestion complete: %d records persisted", persisted)
    return {"persisted": persisted}
