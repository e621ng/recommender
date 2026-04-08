"""Updater state: watermarks and training-artifact paths."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import orjson


@dataclass
class UpdaterState:
    last_event_id: int = 0
    last_event_created_at: str = "1970-01-01T00:00:00"  # created_at of last processed event — partition pruning hint
    last_posts_updated_at: str = "1970-01-01T00:00:00"  # naive ISO-8601, matches DB timestamps
    model_version: str = ""

    def last_event_created_at_dt(self) -> datetime:
        return datetime.fromisoformat(self.last_event_created_at).replace(tzinfo=None)

    def last_posts_updated_at_dt(self) -> datetime:
        dt = datetime.fromisoformat(self.last_posts_updated_at)
        return dt.replace(tzinfo=None)

    def to_dict(self) -> dict:
        return {
            "last_event_id": self.last_event_id,
            "last_event_created_at": self.last_event_created_at,
            "last_posts_updated_at": self.last_posts_updated_at,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UpdaterState":
        return cls(
            last_event_id=int(data.get("last_event_id", 0)),
            last_event_created_at=data.get("last_event_created_at", "1970-01-01T00:00:00"),
            last_posts_updated_at=data.get("last_posts_updated_at", "1970-01-01T00:00:00"),
            model_version=data.get("model_version", ""),
        )


def load_state(path: Path) -> UpdaterState:
    if not path.exists():
        return UpdaterState()
    return UpdaterState.from_dict(orjson.loads(path.read_bytes()))


def save_state(state: UpdaterState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(state.to_dict(), option=orjson.OPT_INDENT_2))
