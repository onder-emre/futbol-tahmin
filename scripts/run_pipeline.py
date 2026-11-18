"""Veri boru hattını çalıştırır.

Örnekler:
    python -m scripts.run_pipeline                       # tüm ligler, son 10 sezon
    python -m scripts.run_pipeline --leagues 203 39      # sadece Süper Lig + Premier Lig
    python -m scripts.run_pipeline --seasons 2023 2024 --max-requests 80
    python -m scripts.run_pipeline --skip-stats          # sadece maç listeleri
    python -m scripts.run_pipeline --init-db             # şemayı Python'dan kur (DBeaver yerine)
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from api.client import ApiFootballClient
from config.settings import PROJECT_ROOT, Settings
from db.connection import DatabaseManager
from db.repository import FootballRepository
from pipeline.ingestion import IngestionPipeline


def _configure_logging(verbose: bool) -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="API-Football → PostgreSQL artımlı veri çekme")
    parser.add_argument("--leagues", type=int, nargs="+", help="Lig ID'leri (varsayılan: .env / tüm ligler)")
    parser.add_argument("--seasons", type=int, nargs="+", help="Sezonlar, ör. 2023 2024")
    parser.add_argument("--max-requests", type=int, help="Bu çalıştırmada en fazla kaç API isteği yapılsın")
    parser.add_argument("--skip-fixtures", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--init-db", action="store_true", help="schema.sql'i uygula ve çık")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    log = logging.getLogger("run_pipeline")
    settings = Settings.from_env()

    db = DatabaseManager(settings.db)
    log.info("Veritabanı: %s", db.ping().split(",")[0])
    if args.init_db:
        db.apply_schema()
        log.info("Şema uygulandı.")
        return 0

    pipeline_settings = settings.pipeline
    if args.leagues:
        pipeline_settings = replace(pipeline_settings, league_ids=tuple(args.leagues))
    if args.seasons:
        pipeline_settings = replace(pipeline_settings, start_season=min(args.seasons), end_season=max(args.seasons))

    client = ApiFootballClient(settings.api)
    client.request_budget = args.max_requests
    pipeline = IngestionPipeline(client, FootballRepository(db.engine), pipeline_settings)

    report = pipeline.run(fetch_fixtures=not args.skip_fixtures, fetch_stats=not args.skip_stats)
    log.info("ÖZET → %s", report)
    if client.daily_remaining is not None:
        log.info("Günlük kalan API isteği: %s", client.daily_remaining)
    db.dispose()
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
