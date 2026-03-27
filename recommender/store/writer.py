"""Atomic versioned artifact writer."""
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
        self._pending_tmp_dir: Path | None = None
        self._pending_version: str | None = None
        self._pending_n: int | None = None
        self._pending_embedding_dim: int | None = None

    def __enter__(self) -> "ArtifactWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._pending_tmp_dir is not None:
            shutil.rmtree(self._pending_tmp_dir, ignore_errors=True)
            self._pending_tmp_dir = None
            self._pending_version = None
            self._pending_n = None
            self._pending_embedding_dim = None
        return False

    def begin_version(
        self,
        *,
        version: str,
        modes: list[str],
        embedding_dim: int,
        state_data: dict,
        post_id_array: np.ndarray,          # shape (N,), int64
        tag_vocab_data: dict,               # {tag_id_str: tag_string}
        post_top_tags_list: list[list[tuple[int, float]]],
        post_fav_count_array: np.ndarray,   # shape (N,), uint32
    ) -> None:
        if self._pending_tmp_dir is not None:
            raise RuntimeError("begin_version() called while a version is already in progress")
        if not modes:
            raise ValueError("modes list is empty; at least one mode must be provided")

        versions_root = Path(self._model_dir) / "versions"
        versions_root.mkdir(parents=True, exist_ok=True)

        tmp_dir = Path(tempfile.mkdtemp(dir=versions_root, prefix="_tmp_"))
        try:
            n = len(post_id_array)

            # manifest
            manifest = {
                "version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "n_posts": int(n),
                "embedding_dim": embedding_dim,
                "modes": modes,
            }
            layout.manifest(tmp_dir).write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

            # state
            layout.state(tmp_dir).write_bytes(orjson.dumps(state_data, option=orjson.OPT_INDENT_2))

            # shared arrays
            np.save(str(layout.post_ids(tmp_dir)), post_id_array.astype(np.int64))
            np.save(str(layout.fav_count(tmp_dir)), post_fav_count_array.astype(np.uint32))

            # tag vocab (zstd-compressed JSON)
            cctx = zstd.ZstdCompressor(level=3)
            layout.tag_vocab(tmp_dir).write_bytes(cctx.compress(orjson.dumps(tag_vocab_data)))

            # top tags binary
            tt.encode(
                post_top_tags_list,
                layout.top_tags_offsets(tmp_dir),
                layout.top_tags_payload(tmp_dir),
            )

            self._pending_tmp_dir = tmp_dir
            self._pending_version = version
            self._pending_n = n
            self._pending_embedding_dim = embedding_dim
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def write_mode(self, mode_name: str, hybrid: np.ndarray, ann) -> None:
        if self._pending_tmp_dir is None:
            raise RuntimeError("begin_version() must be called before write_mode()")
        if hybrid.ndim != 2:
            raise ValueError(f"mode {mode_name!r}: hybrid must be 2-D, got shape {hybrid.shape}")
        if hybrid.shape[0] != self._pending_n:
            raise ValueError(
                f"mode {mode_name!r}: hybrid length {hybrid.shape[0]} != post count {self._pending_n}"
            )
        if hybrid.shape[1] != self._pending_embedding_dim:
            raise ValueError(
                f"mode {mode_name!r}: hybrid dim {hybrid.shape[1]} != manifest embedding_dim {self._pending_embedding_dim}"
            )
        np.save(str(layout.post_vectors(self._pending_tmp_dir, mode_name)), hybrid.astype(np.float16))
        ann.save_index(str(layout.ann_index(self._pending_tmp_dir, mode_name)))

    def finalize_version(self, keep_versions: int = 3) -> Path:
        if self._pending_tmp_dir is None:
            raise RuntimeError("begin_version() must be called before finalize_version()")

        tmp_dir = self._pending_tmp_dir
        version = self._pending_version

        final_dir = layout.version_dir(self._model_dir, version)
        try:
            os.rename(tmp_dir, final_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._pending_tmp_dir = None
            self._pending_version = None
            self._pending_n = None
            self._pending_embedding_dim = None
            raise

        self._pending_tmp_dir = None
        self._pending_version = None
        self._pending_n = None
        self._pending_embedding_dim = None

        versions_root = Path(self._model_dir) / "versions"
        self._update_current_link(version)
        self._prune_old_versions(versions_root, keep_versions)
        return final_dir

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
