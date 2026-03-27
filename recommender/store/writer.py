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
        preset_artifacts: dict,               # mode -> (vectors: float16 (N,D), ann_index)
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
                post_id_array, preset_artifacts,
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
        post_ids: np.ndarray, preset_artifacts: dict,
        tag_vocab_data: dict,
        post_top_tags_list: list[list[tuple[int, float]]],
        post_fav_count: np.ndarray,
    ) -> None:
        vdir.mkdir(parents=True, exist_ok=True)

        if not preset_artifacts:
            raise ValueError("preset_artifacts is empty; at least one mode must be provided")

        n = len(post_ids)
        dims = set()
        for mode_name, (vectors, _) in preset_artifacts.items():
            if vectors.ndim != 2:
                raise ValueError(f"mode {mode_name!r}: vectors must be 2-D, got shape {vectors.shape}")
            if vectors.shape[0] != n:
                raise ValueError(
                    f"mode {mode_name!r}: vectors length {vectors.shape[0]} != post_ids length {n}"
                )
            dims.add(vectors.shape[1])
        if len(dims) > 1:
            raise ValueError(f"modes have inconsistent embedding dims: {dims}")

        first_vectors = next(iter(preset_artifacts.values()))[0]

        # manifest
        manifest = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_posts": int(len(post_ids)),
            "embedding_dim": int(first_vectors.shape[1]) if first_vectors.ndim == 2 else 0,
            "modes": list(preset_artifacts.keys()),
        }
        layout.manifest(vdir).write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

        # state
        layout.state(vdir).write_bytes(orjson.dumps(state_data, option=orjson.OPT_INDENT_2))

        # shared arrays
        np.save(str(layout.post_ids(vdir)), post_ids.astype(np.int64))
        np.save(str(layout.fav_count(vdir)), post_fav_count.astype(np.uint32))

        # per-mode vectors and ANN indexes
        for mode_name, (vectors, ann_index_obj) in preset_artifacts.items():
            np.save(str(layout.post_vectors(vdir, mode_name)), vectors.astype(np.float16))
            ann_index_obj.save_index(str(layout.ann_index(vdir, mode_name)))

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
