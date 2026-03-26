"""Atomic versioned artifact writer."""
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import orjson
import zstandard as zstd

from recommender.store import layout, top_tags as tt


class ArtifactWriter:
    def __init__(self, model_dir: str):
        self._model_dir = model_dir

    def write_version(
        self,
        *,
        version: str,
        state_data: dict,
        post_id_array: np.ndarray,            # shape (N,), int64
        post_vector_array: np.ndarray,         # shape (N, D), float16
        ann_index_obj,                         # hnswlib index
        tag_vocab_data: dict,                  # {tag_id_str: tag_string}
        post_top_tags_list: list[list[tuple[int, float]]],
        post_fav_count_array: np.ndarray,      # shape (N,), uint32
        keep_versions: int = 3,
    ) -> Path:
        versions_root = Path(self._model_dir) / "versions"
        versions_root.mkdir(parents=True, exist_ok=True)

        # Write to a temp dir first, then rename atomically
        tmp_dir = Path(tempfile.mkdtemp(dir=versions_root, prefix="_tmp_"))
        try:
            self._write_artifacts(
                tmp_dir, version, state_data,
                post_id_array, post_vector_array, ann_index_obj,
                tag_vocab_data, post_top_tags_list, post_fav_count_array,
            )
            final_dir = layout.version_dir(self._model_dir, version)
            os.rename(tmp_dir, final_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        self._update_current_link(version)
        self._prune_old_versions(versions_root, keep_versions)
        return final_dir

    def _write_artifacts(
        self, vdir: Path, version: str, state_data: dict,
        post_ids: np.ndarray, post_vectors: np.ndarray, ann_index_obj,
        tag_vocab_data: dict,
        post_top_tags_list: list[list[tuple[int, float]]],
        post_fav_count: np.ndarray,
    ) -> None:
        vdir.mkdir(parents=True, exist_ok=True)

        # manifest
        manifest = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_posts": int(len(post_ids)),
            "embedding_dim": int(post_vectors.shape[1]) if post_vectors.ndim == 2 else 0,
        }
        layout.manifest(vdir).write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

        # state
        layout.state(vdir).write_bytes(orjson.dumps(state_data, option=orjson.OPT_INDENT_2))

        # arrays
        np.save(str(layout.post_ids(vdir)), post_ids.astype(np.int64))
        np.save(str(layout.post_vectors(vdir)), post_vectors.astype(np.float16))
        np.save(str(layout.fav_count(vdir)), post_fav_count.astype(np.uint32))

        # ANN index
        ann_index_obj.save_index(str(layout.ann_index(vdir)))

        # tag vocab (zstd-compressed JSON)
        cctx = zstd.ZstdCompressor(level=3)
        raw = orjson.dumps(tag_vocab_data)
        layout.tag_vocab(vdir).write_bytes(cctx.compress(raw))

        # top tags binary
        tt.encode(
            post_top_tags_list,
            layout.top_tags_offsets(vdir),
            layout.top_tags_payload(vdir),
        )

    def _update_current_link(self, version: str) -> None:
        link = layout.current_link(self._model_dir)
        target = str(layout.version_dir(self._model_dir, version))
        tmp_link = str(link) + ".tmp"
        if os.path.lexists(tmp_link):
            os.remove(tmp_link)
        os.symlink(target, tmp_link)
        os.rename(tmp_link, link)

    def _prune_old_versions(self, versions_root: Path, keep: int) -> None:
        dirs = sorted(
            [d for d in versions_root.iterdir() if d.is_dir() and not d.name.startswith("_tmp_")],
            key=lambda d: d.name,
        )
        for old in dirs[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
