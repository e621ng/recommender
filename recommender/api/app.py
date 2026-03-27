"""FastAPI application factory."""
from contextlib import asynccontextmanager

import structlog

from fastapi import FastAPI

from recommender import __version__
from recommender.api import metrics as m
from recommender.api.engine import SimilarityEngine
from recommender.api.routes import router
from recommender.config import Settings
from recommender.store.reader import ArtifactReader

log = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    cfg = Settings()
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = None
        log.info("api.loading_model", model_dir=cfg.model_dir)
        try:
            reader = ArtifactReader(cfg.model_dir)
            bundle = reader.load_current()
            app.state.engine = SimilarityEngine(bundle, m_shared_tags=cfg.m_shared_tags)
            m.model_version_info.labels(version=bundle.version).set(1)
            m.ann_index_size.set(next(iter(bundle.indexes.values())).get_current_count())
            log.info("api.model_loaded", version=bundle.version, n_posts=len(bundle.post_ids))
        except FileNotFoundError:
            log.warning("api.no_model", model_dir=cfg.model_dir,
                        msg="no model found; /readyz will return 503 until a model is built")

        yield

        app.state.engine = None
        log.info("api.shutdown")

    app = FastAPI(
        title="e621ng Recommender",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
