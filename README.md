# e621ng Recommender

A self-contained recommendation service for the [e621ng](https://github.com/e621ng/e621ng) imageboard.  
Given a post ID, it returns a ranked list of similar posts.


## Basics

Posts are represented as hybrid dense vectors combining two signals:

- **Collaborative**: incremental SGD embeddings learned from `(user, post)` favorite/unfavorite events
- **Tag**: weighted sum of learned tag embeddings, derived from each post's `tag_string`

These are blended and L2-normalized into a final vector, then indexed with [hnswlib](https://github.com/nmslib/hnswlib) for approximate nearest-neighbor retrieval.

The service ships as a **single Docker image** with two entrypoints:

- `recommender api` — FastAPI serving process
- `recommender update` — batch updater (run daily via cron or a K8s CronJob)

The updater consumes new favorite events incrementally using a watermark on `favorite_events.event_id`, so unfavorites are handled naturally and no full retraining is required.


## Prerequisites

- Docker and Docker Compose
- A running e621ng PostgreSQL database with migrations applied


## System requirements

Estimates are based on a full-scale e621ng dataset (~6.3M posts, ~1.25B favorite edges).

|  | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 8 cores |
| RAM | 16 GB | 24–32 GB |
| Disk (model volume) | 50 GB | 100 GB |

**RAM breakdown:**
- The API holds the HNSW index (~4 GB) and post vectors (~0.8 GB) in memory at all times. If `explain=true` queries are common, budget an additional ~2–5 GB for the per-post tag features.
- The updater peaks at ~10–13 GB while rebuilding the index during a daily run.
- At 16 GB, the API and updater cannot run comfortably at the same time. Stop the API before running the updater, then restart it once the new model is promoted.

**Disk:** each model version is ~7–10 GB; `keep_versions=3` (the default) retains three versions for rollback.


## Quick start

```sh
# Copy and edit configuration
cp .env.sample .env

# Build and start the API
docker compose up --build api

# In a separate terminal, run the one-time backfill to build the initial model
docker compose run --rm updater update --backfill

# The API is now available at http://localhost:8000
```

For production, point `RECOMMENDER_DB_DSN` at your real e621ng database and mount a persistent volume for `/models`.
See [`.env.sample`](.env.sample) for all available configuration options.


## API reference

### `GET /similar`

Returns posts similar to the given post.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `post_id` | int | required | Source post ID |
| `limit` | int | `6` | Number of results (max 20) |
| `explain` | bool | `false` | Include shared tags and favorite counts |
| `include_scores` | bool | `true` | Include similarity scores in results |

**Example:**

```
GET /similar?post_id=123&limit=10&explain=true
```

**Response:**

```json
{
  "post_id": 123,
  "model_version": "2026-03-26T00:00:00Z",
  "results": [
    {
      "post_id": 456,
      "score": 0.8123,
      "explanation": {
        "shared_tags": ["wolf", "solo", "male"],
        "fav_counts": { "query": 12034, "candidate": 9833 }
      }
    }
  ]
}
```

If `post_id` is not in the index, `results` is an empty list (HTTP 200).

### `GET /healthz`

Returns `{"status": "ok"}` when the process is alive.

### `GET /readyz`

Returns `{"status": "ready", "model_version": "..."}` when the model is loaded. Returns HTTP 503 if not ready.

### `GET /metrics`

Prometheus metrics endpoint.


## Operational lifecycle

### First run (backfill)

Scans the full `favorites` and `posts` tables, trains embeddings from scratch, builds the ANN index, and writes the first versioned model. Sets watermarks so subsequent incremental runs only process new events.

```sh
docker compose run --rm updater update --backfill
```

### Daily incremental updates

Run once per day (via cron, systemd timer, or K8s CronJob):

```sh
docker compose run --rm updater update
```

Each run fetches new events since the last watermark, refreshes tag features for updated posts, recomputes hybrid vectors, rebuilds the ANN index, and atomically promotes the new model to `/models/current`.

The API does not hot-reload. Restart it after a new model is promoted.

### Rollback

Versioned artifacts are retained under `$RECOMMENDER_MODEL_DIR/versions/`. To roll back, point the `current` symlink at an older version and restart the API:

```sh
ln -sfn /models/versions/<old-version> /models/current
```


## Development

Install [uv](https://docs.astral.sh/uv/), then:

```sh
# Install dependencies including test extras
uv sync --extra dev

# Run tests
uv run pytest tests/
```

Tests cover the core model logic (embeddings, tag weighting, vector blending,
binary serialization) and run in under a second with no database or model
artifacts required.


## Observability

Prometheus metrics are exposed at `/metrics`:

| Metric | Description |
|---|---|
| `api_request_duration_seconds` | Request latency histogram by endpoint |
| `api_requests_total` | Request counter by endpoint and status |
| `api_ann_index_size` | Number of posts in the loaded index |
| `updater_events_processed_total` | Cumulative favorite events processed |
| `updater_watermark_event_id` | Current event ID watermark |
| `updater_event_lag` | Events behind the DB head |
| `updater_index_rebuild_seconds` | ANN index rebuild time histogram |
| `updater_run_duration_seconds` | Total updater run time histogram |

Logs are structured JSON by default. Set `RECOMMENDER_LOG_JSON=false` for human-readable output during local development.
