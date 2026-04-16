"""
Disk-backed store for the training-cache copy of post_top_tags.

On-disk format (training directory):
    post_top_tags_post_ids.npy   int64  (N,)    sorted post IDs
    post_top_tags.offsets.u64    uint64 (N+1,)  byte offsets into payload
    post_top_tags.payload.bin    bytes           packed <If records (8 bytes each)

The payload/offsets format is identical to the serving artifacts written by
store/top_tags.py.  The post-IDs file is new — the serving layer uses a
positional array index; the training cache needs lookup by post ID.

Writes accumulate in an in-memory delta dict and are merged into the binary
base on save().  The two-pointer merge streams the old payload directly to the
new file without buffering a second copy in RAM.  All three output files are
written to .tmp siblings first and then atomically renamed so load() always
sees a consistent generation.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from recommender.store.top_tags import decode_post

log = structlog.get_logger(__name__)

_POST_IDS_FILE = "post_top_tags_post_ids.npy"
_OFFSETS_FILE  = "post_top_tags.offsets.u64"
_PAYLOAD_FILE  = "post_top_tags.payload.bin"
_LEGACY_JSON   = "post_top_tags.pkl.json"

# 256 MB: auto-migrate only small (dev-scale) JSON; warn and start empty for larger
_MIGRATE_LIMIT = 256 * 1024 * 1024

# Mirrors top_tags._RECORD_FMT / _RECORD_SIZE — not imported to avoid coupling
# to a private symbol.
_RECORD_FMT  = "<If"   # little-endian uint32 tag_id + float32 weight
_RECORD_SIZE = 8


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _encode_tags(tags: list[tuple[int, float]]) -> bytes:
    return b"".join(struct.pack(_RECORD_FMT, tid, w) for tid, w in tags)


def _empty_ids() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


def _empty_offsets() -> np.ndarray:
    # Valid offsets array for 0 posts: a single sentinel zero.
    return np.zeros(1, dtype=np.uint64)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class PostTopTagsStore:
    """
    Dict-like interface to the binary training cache for post top-tags.

    All existing call sites in runner.py / backfill.py work unchanged:
        store[post_id] = tags          (__setitem__ → delta)
        store.get(post_id, [])         (delta first, then base binary)
        set(store.keys())              (union of base IDs and delta keys)
        store.is_dirty                 (True if any unsaved changes)
    """

    def __init__(
        self,
        post_ids: np.ndarray,
        offsets: np.ndarray,
        payload: bytes,
        delta: Optional[dict[int, list[tuple[int, float]]]] = None,
    ) -> None:
        self._post_ids: np.ndarray = post_ids          # int64 (N,) sorted
        self._offsets:  np.ndarray = offsets            # uint64 (N+1,)
        self._payload:  bytes = payload                 # raw bytes
        self._payload_mv: Optional[memoryview] = memoryview(payload) if payload else None
        self._payload_path: Optional[Path] = None       # set by save() for lazy reload
        self._delta: dict[int, list[tuple[int, float]]] = delta if delta is not None else {}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def _empty(cls) -> PostTopTagsStore:
        return cls(_empty_ids(), _empty_offsets(), b"")

    @classmethod
    def load(cls, tdir: Path) -> PostTopTagsStore:
        """
        Load from tdir.  Priority:

        1. All 3 binary files present → load and return.
        2. Some but not all binary files present → warn (partial/corrupt store),
           fall through to JSON or empty.
        3. Legacy JSON ≤ _MIGRATE_LIMIT → auto-migrate to binary, return.
        4. Legacy JSON too large, or no files → return empty store (backfill
           required to repopulate).
        """
        ids_path     = tdir / _POST_IDS_FILE
        offsets_path = tdir / _OFFSETS_FILE
        payload_path = tdir / _PAYLOAD_FILE
        json_path    = tdir / _LEGACY_JSON

        ids_exists     = ids_path.exists()
        offsets_exists = offsets_path.exists()
        payload_exists = payload_path.exists()

        if ids_exists and offsets_exists and payload_exists:
            try:
                post_ids = np.load(str(ids_path), allow_pickle=False).astype(np.int64, copy=False)
                offsets  = np.fromfile(str(offsets_path), dtype=np.uint64)
                payload  = payload_path.read_bytes()

                if len(offsets) != len(post_ids) + 1:
                    raise ValueError(
                        f"offsets length mismatch: expected {len(post_ids) + 1}, "
                        f"got {len(offsets)}"
                    )
                if int(offsets[-1]) != len(payload):
                    raise ValueError(
                        f"payload size mismatch: offsets[-1]={int(offsets[-1])}, "
                        f"len(payload)={len(payload)}"
                    )

                log.debug("post_top_tags_store.loaded_binary", n_posts=len(post_ids))
                return cls(post_ids, offsets, payload)
            except (OSError, ValueError, IndexError) as exc:
                log.warning(
                    "post_top_tags_store.corrupt_binary_store",
                    error=str(exc),
                    hint="binary store is corrupt — run --backfill to rebuild",
                )
                # Fall through: attempt legacy JSON migration or return empty.

        if ids_exists or offsets_exists or payload_exists:
            log.warning(
                "post_top_tags_store.partial_binary_store",
                ids_exists=ids_exists,
                offsets_exists=offsets_exists,
                payload_exists=payload_exists,
                hint="binary store is incomplete — run --backfill to rebuild",
            )
            # Fall through: attempt legacy JSON migration or return empty.

        if json_path.exists():
            size = json_path.stat().st_size
            if size > _MIGRATE_LIMIT:
                log.warning(
                    "post_top_tags_store.json_too_large_starting_empty",
                    size_mb=round(size / 1024 / 1024, 1),
                    hint="run --backfill to repopulate the binary store",
                )
                return cls._empty()

            log.warning(
                "post_top_tags_store.migrating_from_json",
                size_mb=round(size / 1024 / 1024, 1),
            )
            import orjson
            raw = orjson.loads(json_path.read_bytes())
            delta: dict[int, list[tuple[int, float]]] = {
                int(k): [tuple(x) for x in v] for k, v in raw.items()
            }
            store = cls(_empty_ids(), _empty_offsets(), b"", delta)
            store.save(tdir)
            return store

        log.debug("post_top_tags_store.no_data_found")
        return cls._empty()

    @classmethod
    def from_dict(cls, d: dict[int, list[tuple[int, float]]]) -> PostTopTagsStore:
        """Construct from an in-memory dict (backfill path).

        Base is empty; everything lives in delta.  A single save() call writes
        the full binary from scratch.
        """
        return cls(_empty_ids(), _empty_offsets(), b"", delta=dict(d))

    # ------------------------------------------------------------------
    # Dict-like interface
    # ------------------------------------------------------------------

    @property
    def is_dirty(self) -> bool:
        """True if there are unsaved changes in the delta."""
        return bool(self._delta)

    def __setitem__(self, post_id: int, tags: list[tuple[int, float]]) -> None:
        self._delta[post_id] = tags

    def _lookup_base(self, post_id: int) -> Optional[list[tuple[int, float]]]:
        """O(log N) lookup into the sorted base array."""
        if len(self._post_ids) == 0:
            return None

        # Lazy reload after save() — payload was invalidated to avoid double-buffering.
        if self._payload_mv is None and self._payload_path is not None:
            self._payload = self._payload_path.read_bytes()
            self._payload_mv = memoryview(self._payload) if self._payload else None

        if self._payload_mv is None:
            return None

        idx = int(np.searchsorted(self._post_ids, post_id))
        if idx < len(self._post_ids) and int(self._post_ids[idx]) == post_id:
            return decode_post(idx, self._offsets, self._payload_mv)
        return None

    def __getitem__(self, post_id: int) -> list[tuple[int, float]]:
        if post_id in self._delta:
            return self._delta[post_id]
        result = self._lookup_base(post_id)
        if result is None:
            raise KeyError(post_id)
        return result

    def get(
        self,
        post_id: int,
        default: Optional[list[tuple[int, float]]] = None,
    ) -> Optional[list[tuple[int, float]]]:
        if post_id in self._delta:
            return self._delta[post_id]
        result = self._lookup_base(post_id)
        return result if result is not None else default

    def keys(self) -> set[int]:
        """Union of base post IDs and delta keys.

        Matches the `set(post_top_tags.keys())` call site in runner.py.
        """
        base: set[int] = {int(pid) for pid in self._post_ids}
        return base | set(self._delta.keys())

    def get_many(
        self, sorted_post_ids: np.ndarray
    ) -> list[list[tuple[int, float]]]:
        """Return tags for every ID in sorted_post_ids via a two-pointer scan.

        O(N + Q) where N = len(base) and Q = len(sorted_post_ids), vs O(Q log N)
        for Q individual .get() calls.  sorted_post_ids must be sorted ascending.
        """
        n_query = len(sorted_post_ids)
        result: list[list[tuple[int, float]]] = [[] for _ in range(n_query)]
        if n_query == 0:
            return result

        # Lazy payload reload (mirrors _lookup_base).
        if len(self._post_ids) > 0 and self._payload_mv is None and self._payload_path is not None:
            self._payload = self._payload_path.read_bytes()
            self._payload_mv = memoryview(self._payload) if self._payload else None

        delta_items = sorted(self._delta.items())
        n_base  = len(self._post_ids)
        n_delta = len(delta_items)

        bi = 0  # base pointer
        di = 0  # delta pointer

        for qi in range(n_query):
            qid = int(sorted_post_ids[qi])

            # Advance each pointer past entries strictly less than the query ID.
            while bi < n_base and int(self._post_ids[bi]) < qid:
                bi += 1
            while di < n_delta and delta_items[di][0] < qid:
                di += 1

            base_hit  = bi < n_base  and int(self._post_ids[bi]) == qid
            delta_hit = di < n_delta and delta_items[di][0] == qid

            if delta_hit:
                result[qi] = delta_items[di][1]
            elif base_hit and self._payload_mv is not None:
                tags = decode_post(bi, self._offsets, self._payload_mv)
                if tags is not None:
                    result[qi] = tags

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, tdir: Path) -> None:
        """Merge delta into base and write new binary files.

        Uses a streaming two-pointer merge so neither the old nor new payload
        needs to be fully buffered in RAM.  All three files are written to .tmp
        siblings and then atomically renamed so load() always sees a consistent
        generation.  After writing, updates self in-place (with lazy payload
        reload) so subsequent reads on the same instance see the merged data —
        this is required because _build_tag_matrix runs after _refresh_posts
        calls save().
        """
        ids_path     = tdir / _POST_IDS_FILE
        offsets_path = tdir / _OFFSETS_FILE
        payload_path = tdir / _PAYLOAD_FILE

        # np.save appends .npy if absent, so the ids temp file must end with .npy.
        ids_tmp     = tdir / "post_top_tags_post_ids.tmp.npy"
        offsets_tmp = tdir / (_OFFSETS_FILE + ".tmp")
        payload_tmp = tdir / (_PAYLOAD_FILE + ".tmp")

        # Sorted inputs for the two-pointer merge
        delta_items: list[tuple[int, list[tuple[int, float]]]] = sorted(self._delta.items())

        n_base  = len(self._post_ids)
        n_delta = len(delta_items)

        new_post_ids:     list[int] = []
        new_offsets_list: list[int] = [0]
        running_offset = 0

        # If payload was invalidated by a prior save() but we still have base
        # IDs to carry forward, reload it now before overwriting the file.
        if n_base and not self._payload and self._payload_mv is None:
            reload_src = self._payload_path or payload_path
            if reload_src.exists():
                self._payload = reload_src.read_bytes()
                self._payload_mv = memoryview(self._payload)

        # memoryview of the old payload for zero-copy slicing into the new file.
        old_mv: Optional[memoryview] = (
            self._payload_mv if self._payload_mv is not None
            else (memoryview(self._payload) if self._payload else None)
        )

        # Refuse to proceed if base IDs exist but the payload is unreadable —
        # the merge would emit empty chunks for every base entry, silently
        # wiping all existing tags.
        if n_base > 0 and old_mv is None:
            raise RuntimeError(
                "PostTopTagsStore.save(): base IDs exist but payload is unreadable; "
                "refusing to save to avoid data loss. Reload the store from disk first."
            )

        with open(payload_tmp, "wb") as f:
            bi = 0  # base pointer
            di = 0  # delta pointer

            while bi < n_base or di < n_delta:
                base_pid   = int(self._post_ids[bi]) if bi < n_base  else None
                delta_pid  = delta_items[di][0]      if di < n_delta else None
                delta_tags = delta_items[di][1]  if di < n_delta else None

                if base_pid is not None and (delta_pid is None or base_pid < delta_pid):
                    # Emit from base — write memoryview slice directly (buffer protocol,
                    # no extra copy).
                    pid   = base_pid
                    start = int(self._offsets[bi])
                    end   = int(self._offsets[bi + 1])
                    chunk = old_mv[start:end] if old_mv is not None else b""
                    bi += 1
                elif delta_pid is not None and (base_pid is None or delta_pid < base_pid):
                    # New post ID — insert from delta.
                    pid   = delta_pid
                    chunk = _encode_tags(delta_tags)
                    di += 1
                else:
                    # Same post ID: delta wins (update).
                    pid   = delta_pid
                    chunk = _encode_tags(delta_tags)
                    bi += 1
                    di += 1

                f.write(chunk)
                new_post_ids.append(pid)
                running_offset += len(chunk)
                new_offsets_list.append(running_offset)

        new_ids_arr     = np.array(new_post_ids, dtype=np.int64)
        new_offsets_arr = np.array(new_offsets_list, dtype=np.uint64)

        np.save(str(ids_tmp), new_ids_arr)
        new_offsets_arr.tofile(str(offsets_tmp))

        # Atomic publish: replace() is atomic per-file (POSIX rename) so the
        # previous generation remains readable until each file is replaced.  A
        # crash between renames leaves one or two old-generation files in place;
        # the partial-store warning in load() catches any incomplete set.
        payload_tmp.replace(payload_path)
        offsets_tmp.replace(offsets_path)
        ids_tmp.replace(ids_path)

        # Update self in-place.  Invalidate payload so _lookup_base reloads
        # lazily on the next read (avoids re-reading a potentially large file
        # immediately if the caller never reads from base after save).
        self._post_ids     = new_ids_arr
        self._offsets      = new_offsets_arr
        self._payload      = b""
        self._payload_mv   = None
        self._payload_path = payload_path
        self._delta.clear()
