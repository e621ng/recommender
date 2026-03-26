"""
Encode/decode the packed top-tags binary format.

Layout:
  offsets file: N+1 uint64 values — offsets[i] is byte offset in payload for post i,
                offsets[N] is total payload size.
  payload file: packed records of (tag_id: u32, weight: f32) = 8 bytes each.

Post order matches post_ids.npy.
"""
import struct
from pathlib import Path

import numpy as np

_RECORD_FMT = "<If"   # little-endian: uint32 tag_id, float32 weight
_RECORD_SIZE = 8


def encode(
    post_top_tags: list[list[tuple[int, float]]],
    offsets_path: Path,
    payload_path: Path,
) -> None:
    """Write offsets + payload files from a list-of-lists."""
    n = len(post_top_tags)
    offsets = np.zeros(n + 1, dtype=np.uint64)
    chunks: list[bytes] = []

    for i, tags in enumerate(post_top_tags):
        chunk = b"".join(struct.pack(_RECORD_FMT, tid, w) for tid, w in tags)
        chunks.append(chunk)
        offsets[i + 1] = offsets[i] + len(chunk)

    offsets.tofile(str(offsets_path))
    with open(payload_path, "wb") as f:
        for chunk in chunks:
            f.write(chunk)


def decode_post(
    post_index: int,
    offsets: np.ndarray,
    payload_mm: memoryview,
) -> list[tuple[int, float]]:
    """Decode top tags for one post given a memoryview of the payload."""
    start = int(offsets[post_index])
    end = int(offsets[post_index + 1])
    n_records = (end - start) // _RECORD_SIZE
    result: list[tuple[int, float]] = []
    for r in range(n_records):
        off = start + r * _RECORD_SIZE
        tag_id, weight = struct.unpack_from(_RECORD_FMT, payload_mm, off)
        result.append((tag_id, weight))
    return result
