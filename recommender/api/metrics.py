"""Prometheus metrics for the API."""
from prometheus_client import Counter, Gauge, Histogram

request_latency = Histogram(
    "api_request_duration_seconds",
    "Request latency",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

requests_total = Counter(
    "api_requests_total",
    "Total requests",
    ["endpoint", "status"],
)

model_version_info = Gauge(
    "api_model_version_loaded",
    "Model version currently loaded (label)",
    ["version"],
)

ann_index_size = Gauge(
    "api_ann_index_size",
    "Number of posts in the ANN index",
)
