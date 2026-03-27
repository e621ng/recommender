"""Single source of truth for artifact filenames within a version directory."""
import re
from pathlib import Path

_SAFE_MODE = re.compile(r'^[a-zA-Z0-9_-]+$')


def _checked_mode(mode: str) -> str:
    if not _SAFE_MODE.match(mode):
        raise ValueError(f"invalid mode name {mode!r}: only letters, digits, hyphens and underscores are allowed")
    return mode


def version_dir(model_dir: str, version: str) -> Path:
    return Path(model_dir) / "versions" / version


def current_link(model_dir: str) -> Path:
    return Path(model_dir) / "current"


def training_dir(model_dir: str) -> Path:
    """Updater-only: stores mutable training-side arrays between runs."""
    return Path(model_dir) / "training"


# --- Serving artifacts (inside a version dir) ---

def manifest(vdir: Path) -> Path:
    return vdir / "manifest.json"


def state(vdir: Path) -> Path:
    return vdir / "state.json"


def post_ids(vdir: Path) -> Path:
    return vdir / "post_ids.npy"


def post_vectors(vdir: Path, mode: str = "favorites") -> Path:
    return vdir / f"post_vectors.{_checked_mode(mode)}.f16.npy"


def ann_index(vdir: Path, mode: str = "favorites") -> Path:
    return vdir / f"ann.{_checked_mode(mode)}.index"


def tag_vocab(vdir: Path) -> Path:
    return vdir / "tag_vocab.json.zst"


def top_tags_offsets(vdir: Path) -> Path:
    return vdir / "post_top_tags.offsets.u64"


def top_tags_payload(vdir: Path) -> Path:
    return vdir / "post_top_tags.payload.bin"


def fav_count(vdir: Path) -> Path:
    return vdir / "post_fav_count.u32.npy"


# --- Training-only artifacts (in training_dir) ---

def user_embeddings(tdir: Path) -> Path:
    return tdir / "user_embeddings.f32.npy"


def post_embeddings_cf(tdir: Path) -> Path:
    return tdir / "post_embeddings_cf.f32.npy"


def tag_embeddings(tdir: Path) -> Path:
    return tdir / "tag_embeddings.f32.npy"


def updater_state(tdir: Path) -> Path:
    return tdir / "updater_state.json"
