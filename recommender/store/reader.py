"""Loads a complete versioned model bundle for serving."""
from pathlib import Path

import numpy as np
import orjson
import zstandard as zstd

import hnswlib

from recommender.store import layout, top_tags as tt


class ModelBundle:
    """All data needed to answer similarity queries."""

    def __init__(
        self,
        version: str,
        post_ids: np.ndarray,           # int64 (N,)
        post_vectors: np.ndarray,       # float16 (N, D)
        ann: hnswlib.Index,
        tag_vocab: dict[str, str],      # tag_id_str -> tag_string
        top_tags_offsets: np.ndarray,   # uint64 (N+1,)
        top_tags_payload: memoryview,
        post_index: dict[int, int],     # post_id -> array index
        fav_count: np.ndarray,          # uint32 (N,)
    ):
        self.version = version
        self.post_ids = post_ids
        self.post_vectors = post_vectors
        self.ann = ann
        self.tag_vocab = tag_vocab
        self.top_tags_offsets = top_tags_offsets
        self.top_tags_payload = top_tags_payload
        self.post_index = post_index
        self.fav_count = fav_count

    def get_top_tags(self, array_index: int) -> list[tuple[int, float]]:
        return tt.decode_post(array_index, self.top_tags_offsets, self.top_tags_payload)

    def tag_name(self, tag_id: int) -> str:
        return self.tag_vocab.get(str(tag_id), f"tag_{tag_id}")


class ArtifactReader:
    def __init__(self, model_dir: str):
        self._model_dir = model_dir

    def load_current(self) -> ModelBundle:
        current = layout.current_link(self._model_dir)
        vdir = Path(current).resolve()
        return self._load(vdir)

    def _load(self, vdir: Path) -> ModelBundle:
        manifest = orjson.loads(layout.manifest(vdir).read_bytes())
        version = manifest["version"]
        dim = manifest["embedding_dim"]

        post_ids = np.load(str(layout.post_ids(vdir)))
        post_vectors = np.load(str(layout.post_vectors(vdir)))
        fav_count = np.load(str(layout.fav_count(vdir)))

        ann = hnswlib.Index(space="cosine", dim=dim)
        ann.load_index(str(layout.ann_index(vdir)))

        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(layout.tag_vocab(vdir).read_bytes())
        tag_vocab: dict[str, str] = orjson.loads(raw)

        top_tags_offsets = np.fromfile(str(layout.top_tags_offsets(vdir)), dtype=np.uint64)
        payload_bytes = layout.top_tags_payload(vdir).read_bytes()
        top_tags_payload = memoryview(payload_bytes)

        post_index = {int(pid): i for i, pid in enumerate(post_ids)}

        return ModelBundle(
            version=version,
            post_ids=post_ids,
            post_vectors=post_vectors,
            ann=ann,
            tag_vocab=tag_vocab,
            top_tags_offsets=top_tags_offsets,
            top_tags_payload=top_tags_payload,
            post_index=post_index,
            fav_count=fav_count,
        )
