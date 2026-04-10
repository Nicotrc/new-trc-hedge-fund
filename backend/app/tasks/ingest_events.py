"""Celery task: ingest event calendar daily at 07:00 EST."""

from __future__ import annotations

import logging

from app.data.normalizer import normalise_event_calendar
from app.data.sources.fda_calendar import FDACalendarConnector
from app.db.engine import SyncSessionLocal
from app.db.models import EventCalendar
from app.tasks.celery_app import app

logger = logging.getLogger(__name__)


@app.task(name="app.tasks.ingest_events.run", bind=True, max_retries=3)
def run(self) -> dict:
    """Fetch and persist FDA events and earnings calendar."""
    connector = FDACalendarConnector()
    try:
        fda_events = connector.fetch_events()
        earnings_events = connector.fetch_earnings_calendar()
        raw_records = fda_events + earnings_events
    except Exception as exc:
        logger.exception("Event calendar fetch failed")
        raise self.retry(exc=exc, countdown=300)

    normalised = normalise_event_calendar(raw_records)
    if not normalised:
        logger.info("No event calendar records to persist")
        return {"persisted": 0}

    persisted = 0
    with SyncSessionLocal() as session:
        for rec in normalised:
            try:
                row = EventCalendar(
                    ticker=rec["ticker"],
                    event_type=rec["event_type"],
                    event_date=rec.get("event_date"),
                    description=rec.get("description"),
                    impact_estimate=rec.get("impact_estimate"),
                    binary_outcome=rec.get("binary_outcome", False),
                    source=rec.get("source", "fda_calendar"),
                )
                session.add(row)
                persisted += 1
            except Exception:
                logger.exception("Failed to persist event: %s %s", rec.get("ticker"), rec.get("event_type"))
        session.commit()

    logger.info("Event calendar ingestion complete: %d records", persisted)
    return {"persisted": persisted}
