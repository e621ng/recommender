"""Prometheus metrics for the updater."""
from prometheus_client import Counter, Gauge, Histogram

events_processed_total = Counter(
    "updater_events_processed_total",
    "Total favorite events processed",
)

current_watermark = Gauge(
    "updater_watermark_event_id",
    "Current last_event_id watermark",
)

event_lag = Gauge(
    "updater_event_lag",
    "Lag between max event_id in DB and last processed",
)

changed_posts_total = Counter(
    "updater_changed_posts_total",
    "Total posts with tag changes processed",
)

index_rebuild_seconds = Histogram(
    "updater_index_rebuild_seconds",
    "Time to rebuild the ANN index",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

run_duration_seconds = Histogram(
    "updater_run_duration_seconds",
    "Total updater run duration",
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
)
