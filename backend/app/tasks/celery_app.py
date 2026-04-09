import os

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL_SECONDS", "900"))

app = Celery(
    "hedgefund",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "app.tasks.ingest_price",
        "app.tasks.ingest_fundamentals",
        "app.tasks.ingest_insider",
        "app.tasks.ingest_news",
        "app.tasks.ingest_macro",
        "app.tasks.ingest_options",
        "app.tasks.ingest_events",
        "app.tasks.scan_market",
        "app.tasks.analyse_opportunity",
        "app.tasks.monitor_positions",
    ],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    beat_schedule={
        # --- Data ingestion (L1) ---
        "ingest-price": {
            "task": "app.tasks.ingest_price.run",
            "schedule": 900,  # every 15 minutes (was 5m, relaxed to reduce load)
        },
        "ingest-fundamentals": {
            "task": "app.tasks.ingest_fundamentals.run",
            "schedule": crontab(hour=6, minute=0),
        },
        "ingest-insider": {
            "task": "app.tasks.ingest_insider.run",
            "schedule": crontab(hour=18, minute=0),  # post-market
        },
        "ingest-news": {
            "task": "app.tasks.ingest_news.run",
            "schedule": 1800,  # every 30 minutes
        },
        # --- New V2 data sources ---
        "ingest-macro": {
            "task": "app.tasks.ingest_macro.run",
            "schedule": 21600,  # every 6 hours
        },
        "ingest-options": {
            "task": "app.tasks.ingest_options.run",
            "schedule": 3600,  # every 1 hour
        },
        "ingest-events": {
            "task": "app.tasks.ingest_events.run",
            "schedule": crontab(hour=7, minute=0),  # daily pre-market
        },
        # --- Signal scanning (L2) ---
        "scan-market": {
            "task": "app.tasks.scan_market.run",
            "schedule": SCAN_INTERVAL,
        },
        # --- Position monitoring (L6) ---
        "monitor-positions": {
            "task": "app.tasks.monitor_positions.run",
            "schedule": 300,  # every 5 minutes during market hours
        },
    },
)


@worker_ready.connect
def start_consume_queue(sender, **kwargs):
    """Auto-start the BLPOP consumer when a Celery worker comes online."""
    app.send_task("app.tasks.analyse_opportunity.consume_queue")
