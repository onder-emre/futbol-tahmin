"""football-data.co.uk aynasından PostgreSQL'e veri yükler.

    python -m scripts.load_fdcsv                       # 6 lig, 2019 → güncel sezon
    python -m scripts.load_fdcsv --start 2015          # daha eski geçmiş
    python -m scripts.load_fdcsv --leagues T1 E0       # sadece bazı ligler
    python -m scripts.load_fdcsv --keep                # eski veriyi silme (upsert)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DatabaseSettings, current_season  # noqa: E402
from db.connection import DatabaseManager  # noqa: E402
from db.repository import FootballRepository  # noqa: E402
from pipeline.fdcsv import LEAGUE_MAP, FootballDataCsvSource, write_frames  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="football-data.co.uk aynası → PostgreSQL")
    p.add_argument("--leagues", nargs="+", default=list(LEAGUE_MAP), help="Lig kodları (T1 E0 SP1 I1 D1 F1)")
    p.add_argument("--start", type=int, default=2019, help="En eski sezon başlangıç yılı (vars: 2019)")
    p.add_argument("--end", type=int, default=None, help="En yeni sezon başlangıç yılı (vars: güncel sezon)")
    p.add_argument("--base-url", default=None, help="Alternatif ayna kök URL'i")
    p.add_argument("--keep", action="store_true", help="Eski veriyi silme (varsayılan: temiz yeniden yükleme)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    log = logging.getLogger("load_fdcsv")

    end = args.end or current_season()
    years = list(range(args.start, end + 1))
    log.info("Sezonlar: %s | Ligler: %s", years, args.leagues)

    src = FootballDataCsvSource(args.base_url) if args.base_url else FootballDataCsvSource()
    frames = src.build(args.leagues, years)
    if frames.maclar.empty:
        log.error("Hiç maç çekilemedi — ayna erişilebilir mi?")
        return 1

    db = DatabaseManager(DatabaseSettings.from_env())
    repo = FootballRepository(db.engine)

    if not args.keep:
        log.info("Temiz yeniden yükleme (eski veri silinecek)")
    n = write_frames(repo, db.engine, frames, truncate=not args.keep)
    log.info("ÖZET → lig: %d | takım: %d | maç: %d | istatistik satırı: %d",
             n["lig"], n["takim"], n["mac"], n["istatistik"])
    db.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
