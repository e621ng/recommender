"""CLI entrypoint: `recommender api` and `recommender update`."""
import typer

app = typer.Typer(name="recommender", add_completion=False)


@app.command("api")
def cmd_api(
    host: str = typer.Option(None, envvar="RECOMMENDER_API_HOST"),
    port: int = typer.Option(None, envvar="RECOMMENDER_API_PORT"),
    workers: int = typer.Option(None, envvar="RECOMMENDER_API_WORKERS"),
):
    """Start the FastAPI serving process."""
    import uvicorn
    from recommender.config import Settings
    from recommender.logging import configure_logging

    cfg = Settings()
    configure_logging(level=cfg.log_level, json=cfg.log_json)

    uvicorn.run(
        "recommender.api.app:create_app",
        factory=True,
        host=host or cfg.api_host,
        port=port or cfg.api_port,
        workers=workers or cfg.api_workers,
        log_config=None,  # use our structlog config
    )


@app.command("update")
def cmd_update(
    backfill: bool = typer.Option(False, "--backfill", help="Run one-time full backfill"),
):
    """Run the incremental updater (or full backfill with --backfill)."""
    from recommender.config import Settings
    from recommender.logging import configure_logging

    cfg = Settings()
    configure_logging(level=cfg.log_level, json=cfg.log_json)

    if backfill:
        from recommender.updater.backfill import run_backfill
        run_backfill(cfg)
    else:
        from recommender.updater.runner import run_update
        run_update(cfg)


if __name__ == "__main__":
    app()
