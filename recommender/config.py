from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECOMMENDER_", env_file=".env", extra="ignore")

    # Database
    db_dsn: str = Field(default="postgresql://postgres:postgres@localhost:5432/e621")

    # Model store
    model_dir: str = Field(default="/models")

    # Serving
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)

    # Hybrid blend weights
    w_cf: float = Field(default=1.0)
    w_tag: float = Field(default=0.3)

    # Embedding dimensions
    embedding_dim: int = Field(default=64)

    # Per-post top-tag limit
    n_top_tags: int = Field(default=100)

    # Fallback total post count used in IDF when DB is unavailable
    n_posts_fallback: int = Field(default=6_280_000)

    # Category weight multipliers (category values per e621ng schema)
    tag_weight_general: float = Field(default=1.0)    # category 0
    tag_weight_artist: float = Field(default=3.0)     # category 1
    tag_weight_copyright: float = Field(default=1.5)  # category 3
    tag_weight_character: float = Field(default=2.5)  # category 4
    tag_weight_species: float = Field(default=2.0)    # category 5
    tag_weight_invalid: float = Field(default=0.0)    # category 6 (excluded)
    tag_weight_meta: float = Field(default=0.5)       # category 7
    tag_weight_lore: float = Field(default=1.2)       # category 8

    @property
    def category_multipliers(self) -> dict[int, float]:
        return {
            0: self.tag_weight_general,
            1: self.tag_weight_artist,
            3: self.tag_weight_copyright,
            4: self.tag_weight_character,
            5: self.tag_weight_species,
            6: self.tag_weight_invalid,
            7: self.tag_weight_meta,
            8: self.tag_weight_lore,
        }

    # Explainability
    m_shared_tags: int = Field(default=6)

    # SGD hyper-params
    sgd_lr: float = Field(default=0.01)
    sgd_reg: float = Field(default=0.001)

    # Updater batch sizes
    events_batch_size: int = Field(default=50_000)
    posts_batch_size: int = Field(default=10_000)

    # ANN index build params
    hnsw_m: int = Field(default=32)
    hnsw_ef_construction: int = Field(default=200)
    hnsw_ef_search: int = Field(default=100)

    # Logging
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True)

    # How many old model versions to keep
    keep_versions: int = Field(default=3)
