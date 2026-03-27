"""Daily incremental update orchestrator."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import structlog

from recommender.config import Settings
from recommender.model.ann import build_index
from recommender.model.embeddings import EmbeddingTable, apply_event_batch
from recommender.model.hybrid import compute_hybrid_vectors
from recommender.model.tags import TagMeta, TagVocab, compute_post_top_tags, compute_tag_vector
from recommender.store.layout import training_dir, updater_state as state_path
from recommender.store.layout import (
    user_embeddings as ue_path,
    post_embeddings_cf as pe_path,
    tag_embeddings as te_path,
)
from recommender.store.writer import ArtifactWriter
from recommender.updater import db as dbmod
from recommender.updater import metrics
from recommender.updater.state import UpdaterState, load_state, save_state

log = structlog.get_logger(__name__)


def run_update(cfg: Settings) -> None:
    t0 = time.time()
    tdir = Path(training_dir(cfg.model_dir))
    tdir.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path(tdir))
    log.info("updater.start", last_event_id=state.last_event_id,
             last_posts_updated_at=state.last_posts_updated_at)

    # --- Load training artifacts ---
    user_table, post_table, tag_emb, vocab, post_top_tags, fav_count = _load_training_state(
        tdir, cfg
    )

    conn = dbmod.connect_with_retry(cfg.db_dsn)

    try:
        # --- 1. Consume favorite events ---
        n_events = _consume_events(conn, state, user_table, post_table, fav_count, cfg)
        log.info("updater.events_done", n_events=n_events, watermark=state.last_event_id)

        # --- 2. Fetch tag metadata for weight computation ---
        tag_metadata = dbmod.fetch_tag_metadata(conn)
        n_posts_total = dbmod.fetch_post_count(conn)
        log.info("updater.tag_metadata_fetched", n_tags=len(tag_metadata), n_posts=n_posts_total)

        # --- 3. Refresh changed posts/tags ---
        n_posts, tag_emb = _refresh_posts(conn, state, vocab, post_top_tags, tag_emb, tag_metadata, n_posts_total, cfg)
        log.info("updater.posts_done", n_posts=n_posts)

        # --- 4. Compute hybrid vectors ---
        # Build post universe from the union of CF-trained posts and tag-known posts.
        # Posts present in post_top_tags but absent from post_table have no collaborative
        # signal yet; they receive a zero CF vector so they are still reachable via tags.
        all_post_ids = sorted(set(post_table.ids()) | set(post_top_tags.keys()))
        if not all_post_ids:
            log.warning("updater.no_posts")
            return

        post_id_arr = np.array(all_post_ids, dtype=np.int64)
        cf_ids, cf_vecs = post_table.to_arrays()
        cf_lookup: dict[int, int] = {int(pid): i for i, pid in enumerate(cf_ids.tolist())}
        _zero_cf = np.zeros(cfg.embedding_dim, dtype=np.float32)
        cf_matrix = np.stack([
            cf_vecs[cf_lookup[pid]] if pid in cf_lookup else _zero_cf
            for pid in all_post_ids
        ])
        tag_matrix = _build_tag_matrix(post_id_arr, post_top_tags, tag_emb, cfg.embedding_dim)

        # --- 5. Build aligned arrays and begin writing versioned artifacts ---
        fav_arr = np.array([fav_count.get(int(pid), 0) for pid in post_id_arr], dtype=np.uint32)
        top_tags_list = [post_top_tags.get(int(pid), []) for pid in post_id_arr]

        version = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with ArtifactWriter(cfg.model_dir) as writer:
            writer.begin_version(
                version=version,
                modes=list(cfg.weight_presets),
                embedding_dim=cfg.embedding_dim,
                state_data=state.to_dict(),
                post_id_array=post_id_arr,
                tag_vocab_data=vocab.to_dict(),
                post_top_tags_list=top_tags_list,
                post_fav_count_array=fav_arr,
            )

            # --- 6. Build and write one ANN index per mode, freeing each before the next ---
            log.info("updater.building_indexes", n_posts=len(post_id_arr), modes=list(cfg.weight_presets))
            t_idx = time.time()
            for mode_name, (w_cf, w_tag) in cfg.weight_presets.items():
                hybrid = compute_hybrid_vectors(cf_matrix, tag_matrix, w_cf, w_tag)
                ann = build_index(
                    hybrid,
                    post_id_arr,
                    m=cfg.hnsw_m,
                    ef_construction=cfg.hnsw_ef_construction,
                    ef_search=cfg.hnsw_ef_search,
                )
                writer.write_mode(mode_name, hybrid, ann)
                del hybrid, ann
            metrics.index_rebuild_seconds.observe(time.time() - t_idx)

            writer.finalize_version(keep_versions=cfg.keep_versions)
        log.info("updater.version_promoted", version=version)

        # --- 8. Save training artifacts ---
        _save_training_state(tdir, user_table, post_table, tag_emb)
        state.model_version = version
        save_state(state, state_path(tdir))

    finally:
        conn.close()

    metrics.run_duration_seconds.observe(time.time() - t0)
    log.info("updater.done", elapsed_s=round(time.time() - t0, 1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_training_state(
    tdir: Path, cfg: Settings
) -> tuple[EmbeddingTable, EmbeddingTable, np.ndarray, TagVocab, dict, dict]:
    import orjson
    import zstandard as zstd
    from recommender.store.layout import tag_vocab, current_link, version_dir
    from pathlib import Path as _Path

    dim = cfg.embedding_dim

    # User embeddings
    up = ue_path(tdir)
    if up.exists():
        data = np.load(str(up), allow_pickle=False)
        # saved as [ids_row, matrix...] or separate — we use paired files
        pass  # handled below with separate id files

    user_table = _load_embedding_table(tdir / "user_embeddings.f32.npy",
                                       tdir / "user_ids.npy", dim)
    post_table = _load_embedding_table(tdir / "post_embeddings_cf.f32.npy",
                                       tdir / "post_ids_cf.npy", dim)

    # Tag embeddings
    te = te_path(tdir)
    if te.exists():
        tag_emb = np.load(str(te), allow_pickle=False).astype(np.float32)
    else:
        tag_emb = np.zeros((0, dim), dtype=np.float32)

    # Tag vocab — try to load from current serving version
    vocab_path = tdir / "tag_vocab_training.json"
    if vocab_path.exists():
        vocab = TagVocab.from_dict(orjson.loads(vocab_path.read_bytes()))
    else:
        vocab = TagVocab()

    # Post top tags
    ptt_path = tdir / "post_top_tags.pkl.json"  # stored as orjson
    post_top_tags: dict[int, list[tuple[int, float]]] = {}
    if ptt_path.exists():
        raw = orjson.loads(ptt_path.read_bytes())
        post_top_tags = {int(k): [tuple(x) for x in v] for k, v in raw.items()}

    # Fav count
    fc_path = tdir / "fav_count.json"
    fav_count: dict[int, int] = {}
    if fc_path.exists():
        fav_count = {int(k): v for k, v in orjson.loads(fc_path.read_bytes()).items()}

    return user_table, post_table, tag_emb, vocab, post_top_tags, fav_count


def _load_embedding_table(matrix_path: Path, ids_path: Path, dim: int) -> EmbeddingTable:
    if matrix_path.exists() and ids_path.exists():
        matrix = np.load(str(matrix_path), allow_pickle=False).astype(np.float32)
        ids = np.load(str(ids_path), allow_pickle=False).astype(np.int64)
        return EmbeddingTable.from_arrays(ids, matrix)
    return EmbeddingTable(dim=dim)


def _save_training_state(
    tdir: Path,
    user_table: EmbeddingTable,
    post_table: EmbeddingTable,
    tag_emb: np.ndarray,
) -> None:
    import orjson

    u_ids, u_mat = user_table.to_arrays()
    p_ids, p_mat = post_table.to_arrays()

    np.save(str(tdir / "user_embeddings.f32.npy"), u_mat)
    np.save(str(tdir / "user_ids.npy"), u_ids)
    np.save(str(tdir / "post_embeddings_cf.f32.npy"), p_mat)
    np.save(str(tdir / "post_ids_cf.npy"), p_ids)
    np.save(str(tdir / "tag_embeddings.f32.npy"), tag_emb)


def _consume_events(
    conn, state: UpdaterState,
    user_table: EmbeddingTable, post_table: EmbeddingTable,
    fav_count: dict[int, int], cfg: Settings,
) -> int:
    import orjson
    from recommender.store.layout import training_dir
    tdir = Path(training_dir(cfg.model_dir))

    n_total = 0
    max_db_event_id = dbmod.fetch_max_event_id(conn)
    metrics.event_lag.set(max(0, max_db_event_id - state.last_event_id))

    for batch in dbmod.fetch_event_batches(conn, state.last_event_id, cfg.events_batch_size):
        events = [(e.user_id, e.post_id, e.action) for e in batch]
        apply_event_batch(user_table, post_table, fav_count, events, cfg.sgd_lr, cfg.sgd_reg)
        state.last_event_id = batch[-1].event_id
        n_total += len(batch)
        metrics.events_processed_total.inc(len(batch))
        metrics.current_watermark.set(state.last_event_id)

    # Persist fav_count
    fc_path = tdir / "fav_count.json"
    fc_path.write_bytes(orjson.dumps({str(k): v for k, v in fav_count.items()}))

    return n_total


def _refresh_posts(
    conn, state: UpdaterState,
    vocab: TagVocab,
    post_top_tags: dict[int, list[tuple[int, float]]],
    tag_emb: np.ndarray,
    tag_metadata: dict[str, TagMeta],
    n_posts_total: int,
    cfg: Settings,
) -> tuple[int, np.ndarray]:
    import orjson

    tdir = Path(training_dir(cfg.model_dir))
    after_dt = state.last_posts_updated_at_dt()
    n_total = 0
    max_seen = after_dt
    cat_multipliers = cfg.category_multipliers

    for batch in dbmod.fetch_changed_posts_batches(conn, after_dt, cfg.posts_batch_size):
        for post in batch:
            top_tags = compute_post_top_tags(
                post.tag_string, vocab,
                n_top=cfg.n_top_tags,
                n_posts=n_posts_total,
                tag_metadata=tag_metadata,
                category_multipliers=cat_multipliers,
            )
            post_top_tags[post.id] = top_tags
            if post.updated_at > max_seen:
                max_seen = post.updated_at

        n_total += len(batch)
        metrics.changed_posts_total.inc(len(batch))

    # Grow tag embeddings if vocab expanded
    new_vocab_size = len(vocab)
    if new_vocab_size > len(tag_emb):
        extra = new_vocab_size - len(tag_emb)
        rng = np.random.default_rng()
        new_rows = rng.standard_normal((extra, cfg.embedding_dim)).astype(np.float32) * 0.01
        tag_emb = np.concatenate([tag_emb, new_rows], axis=0) if len(tag_emb) > 0 else new_rows
        np.save(str(te_path(tdir)), tag_emb)

    # Persist vocab and top tags
    (tdir / "tag_vocab_training.json").write_bytes(orjson.dumps(vocab.to_dict()))
    (tdir / "post_top_tags.pkl.json").write_bytes(
        orjson.dumps({str(k): v for k, v in post_top_tags.items()})
    )

    if max_seen > after_dt:
        state.last_posts_updated_at = max_seen.isoformat()

    return n_total, tag_emb


def _build_tag_matrix(
    post_id_arr: np.ndarray,
    post_top_tags: dict[int, list[tuple[int, float]]],
    tag_emb: np.ndarray,
    dim: int,
) -> np.ndarray:
    n = len(post_id_arr)
    matrix = np.zeros((n, dim), dtype=np.float32)
    for i, pid in enumerate(post_id_arr.tolist()):
        top_tags = post_top_tags.get(int(pid), [])
        matrix[i] = compute_tag_vector(top_tags, tag_emb, dim)
    return matrix
