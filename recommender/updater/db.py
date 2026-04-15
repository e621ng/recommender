"""PostgreSQL access for the updater. Returns plain dataclasses, no ORM."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Generator

import psycopg

from recommender.model.tags import TagMeta
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class FavoriteEvent:
    event_id: int
    user_id: int
    post_id: int
    action: int          # +1 or -1
    created_at: datetime


@dataclass
class PostRecord:
    id: int
    tag_string: str
    updated_at: datetime

    def __post_init__(self):
        # Normalize to naive so comparisons with our watermarks don't raise.
        # The DB may return aware datetimes (timestamptz) or naive ones depending
        # on the column type; we treat everything as naive UTC internally.
        if self.updated_at is not None and self.updated_at.tzinfo is not None:
            self.updated_at = self.updated_at.replace(tzinfo=None)


def connect(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
def connect_with_retry(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn)


def fetch_max_event_id(conn: psycopg.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(event_id), 0) FROM public.favorite_events").fetchone()
    return int(row[0])


def fetch_max_updated_at(conn: psycopg.Connection) -> datetime | None:
    row = conn.execute("SELECT MAX(updated_at) FROM public.posts").fetchone()
    return row[0]


def fetch_event_batches(
    conn: psycopg.Connection,
    after_event_id: int,
    batch_size: int = 50_000,
) -> Generator[list[FavoriteEvent], None, None]:
    """Yield batches of FavoriteEvent ordered by event_id."""
    last_id = after_event_id
    while True:
        rows = conn.execute(
            """
            SELECT event_id, user_id, post_id, action, created_at
            FROM public.favorite_events
            WHERE event_id > %s
            ORDER BY event_id
            LIMIT %s
            """,
            (last_id, batch_size),
        ).fetchall()
        if not rows:
            break
        events = [FavoriteEvent(*row) for row in rows]
        yield events
        last_id = events[-1].event_id
        if len(rows) < batch_size:
            break


def fetch_changed_posts_batches(
    conn: psycopg.Connection,
    after_updated_at: datetime | None,
    batch_size: int = 10_000,
) -> Generator[list[PostRecord], None, None]:
    """Yield batches of PostRecord with updated_at > after_updated_at."""
    if after_updated_at is None:
        after_updated_at = datetime(1970, 1, 1)

    last_ts = after_updated_at
    last_id = -1
    while True:
        rows = conn.execute(
            """
            SELECT id, tag_string, updated_at
            FROM public.posts
            WHERE (updated_at, id) > (%s, %s)
            ORDER BY updated_at, id
            LIMIT %s
            """,
            (last_ts, last_id, batch_size),
        ).fetchall()
        if not rows:
            break
        posts = [PostRecord(*row) for row in rows]
        yield posts
        last_ts = posts[-1].updated_at
        last_id = posts[-1].id
        if len(rows) < batch_size:
            break


def fetch_all_favorites(
    conn: psycopg.Connection,
    batch_size: int = 50_000,
) -> Generator[list[tuple[int, int]], None, None]:
    """Yield (user_id, post_id) pairs from the full favorites table (backfill)."""
    last_user_id = -1
    last_post_id = -1
    while True:
        rows = conn.execute(
            """
            SELECT user_id, post_id FROM public.favorites
            WHERE (user_id, post_id) > (%s, %s)
            ORDER BY user_id, post_id
            LIMIT %s
            """,
            (last_user_id, last_post_id, batch_size),
        ).fetchall()
        if not rows:
            break
        yield [(r[0], r[1]) for r in rows]
        last_user_id, last_post_id = rows[-1]
        if len(rows) < batch_size:
            break


def fetch_tag_metadata(conn: psycopg.Connection) -> dict[str, TagMeta]:
    """Return {tag_name: TagMeta} for all tags."""
    rows = conn.execute(
        "SELECT name, category, post_count FROM public.tags"
    ).fetchall()
    return {name: TagMeta(category=category, post_count=post_count) for name, category, post_count in rows}


def fetch_post_count(conn: psycopg.Connection) -> int:
    """Return total number of posts."""
    row = conn.execute("SELECT COUNT(*) FROM public.posts").fetchone()
    return int(row[0])


def fetch_all_posts(
    conn: psycopg.Connection,
    batch_size: int = 10_000,
) -> Generator[list[PostRecord], None, None]:
    """Yield all posts (backfill)."""
    last_id = -1
    while True:
        rows = conn.execute(
            "SELECT id, tag_string, updated_at FROM public.posts WHERE id > %s ORDER BY id LIMIT %s",
            (last_id, batch_size),
        ).fetchall()
        if not rows:
            break
        posts = [PostRecord(*row) for row in rows]
        yield posts
        last_id = posts[-1].id
        if len(rows) < batch_size:
            break
