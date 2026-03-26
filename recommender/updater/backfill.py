"""One-time bootstrap: full-table scan of favorites + posts."""
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
from recommender.model.tags import TagVocab, compute_post_top_tags, compute_tag_vector
from recommender.store.layout import training_dir, updater_state as state_path
from recommender.store.writer import ArtifactWriter
from recommender.updater import db as dbmod
from recommender.updater.state import UpdaterState, save_state

log = structlog.get_logger(__name__)


def run_backfill(cfg: Settings) -> None:
    t0 = time.time()
    tdir = Path(training_dir(cfg.model_dir))
    tdir.mkdir(parents=True, exist_ok=True)

    dim = cfg.embedding_dim
    user_table = EmbeddingTable(dim=dim)
    post_table = EmbeddingTable(dim=dim)
    fav_count: dict[int, int] = {}
    vocab = TagVocab()
    post_top_tags: dict[int, list[tuple[int, float]]] = {}

    conn = dbmod.connect_with_retry(cfg.db_dsn)
    try:
        # --- Scan all posts first (build vocab) ---
        log.info("backfill.posts_start")
        n_posts = 0
        max_updated_at = datetime(1970, 1, 1)
        for batch in dbmod.fetch_all_posts(conn, cfg.posts_batch_size):
            for post in batch:
                top_tags = compute_post_top_tags(post.tag_string, vocab, cfg.n_top_tags)
                post_top_tags[post.id] = top_tags
                if post.updated_at and post.updated_at > max_updated_at:
                    max_updated_at = post.updated_at
            n_posts += len(batch)
            if n_posts % 100_000 == 0:
                log.info("backfill.posts_progress", n=n_posts)

        log.info("backfill.posts_done", n=n_posts, vocab_size=len(vocab))

        # Initialize tag embeddings
        vocab_size = len(vocab)
        rng = np.random.default_rng(42)
        tag_emb = rng.standard_normal((vocab_size, dim)).astype(np.float32) * 0.01

        # --- Scan all favorites ---
        log.info("backfill.favorites_start")
        n_events = 0
        for batch in dbmod.fetch_all_favorites(conn, cfg.events_batch_size):
            events = [(uid, pid, 1) for uid, pid in batch]
            apply_event_batch(user_table, post_table, fav_count, events, cfg.sgd_lr, cfg.sgd_reg)
            n_events += len(batch)
            if n_events % 1_000_000 == 0:
                log.info("backfill.favorites_progress", n=n_events)

        log.info("backfill.favorites_done", n=n_events)

        # --- Compute hybrid vectors ---
        post_id_arr, cf_matrix = post_table.to_arrays()
        n = len(post_id_arr)
        tag_matrix = np.zeros((n, dim), dtype=np.float32)
        for i, pid in enumerate(post_id_arr.tolist()):
            top_tags = post_top_tags.get(int(pid), [])
            tag_matrix[i] = compute_tag_vector(top_tags, tag_emb, dim)

        hybrid = compute_hybrid_vectors(cf_matrix, tag_matrix, cfg.w_cf, cfg.w_tag)

        # --- Build ANN ---
        log.info("backfill.building_index", n=n)
        ann = build_index(
            hybrid.astype(np.float32), post_id_arr,
            m=cfg.hnsw_m, ef_construction=cfg.hnsw_ef_construction, ef_search=cfg.hnsw_ef_search,
        )

        fav_arr = np.array([fav_count.get(int(p), 0) for p in post_id_arr], dtype=np.uint32)
        top_tags_list = [post_top_tags.get(int(p), []) for p in post_id_arr]

        # --- Write version ---
        max_event_id = dbmod.fetch_max_event_id(conn)
        version = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        state = UpdaterState(
            last_event_id=max_event_id,
            last_posts_updated_at=max_updated_at.isoformat(),
            model_version=version,
        )

        writer = ArtifactWriter(cfg.model_dir)
        writer.write_version(
            version=version,
            state_data=state.to_dict(),
            post_id_array=post_id_arr,
            post_vector_array=hybrid,
            ann_index_obj=ann,
            tag_vocab_data=vocab.to_dict(),
            post_top_tags_list=top_tags_list,
            post_fav_count_array=fav_arr,
            keep_versions=cfg.keep_versions,
        )
        log.info("backfill.version_promoted", version=version)

        # --- Save training artifacts ---
        import orjson
        u_ids, u_mat = user_table.to_arrays()
        p_ids, p_mat = post_table.to_arrays()
        np.save(str(tdir / "user_embeddings.f32.npy"), u_mat)
        np.save(str(tdir / "user_ids.npy"), u_ids)
        np.save(str(tdir / "post_embeddings_cf.f32.npy"), p_mat)
        np.save(str(tdir / "post_ids_cf.npy"), p_ids)
        np.save(str(tdir / "tag_embeddings.f32.npy"), tag_emb)
        (tdir / "tag_vocab_training.json").write_bytes(orjson.dumps(vocab.to_dict()))
        (tdir / "post_top_tags.pkl.json").write_bytes(
            orjson.dumps({str(k): v for k, v in post_top_tags.items()})
        )
        (tdir / "fav_count.json").write_bytes(
            orjson.dumps({str(k): v for k, v in fav_count.items()})
        )

        save_state(state, state_path(tdir))

    finally:
        conn.close()

    log.info("backfill.done", elapsed_s=round(time.time() - t0, 1))
