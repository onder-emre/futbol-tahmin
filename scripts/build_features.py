"""Öznitelikleri hesaplar ve PostgreSQL'de `mac_ozellikleri` tablosuna yazar (DBeaver'dan incelenebilir).

    python -m scripts.build_features                 # tüm ligler
    python -m scripts.build_features --leagues 203   # sadece Süper Lig
    python -m scripts.build_features --window 10     # son 10 maçlık form
"""
from __future__ import annotations

import argparse
import logging
import sys

from config.settings import DatabaseSettings
from db.connection import DatabaseManager
from models.dataset import FEATURE_TABLE, build_and_store_features
from models.features import FeatureConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Form / H2H öznitelikleri üret")
    parser.add_argument("--leagues", type=int, nargs="+")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--h2h-window", type=int, default=5)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("build_features")

    db = DatabaseManager(DatabaseSettings.from_env())
    fs = build_and_store_features(db.engine, args.leagues, FeatureConfig(window=args.window, h2h_window=args.h2h_window))
    if fs.frame.empty:
        log.warning("Veritabanında maç yok. Önce: python -m scripts.run_pipeline")
        return 1
    log.info("%d maç (%d eğitim, %d yaklaşan), %d öznitelik → tablo: %s",
             len(fs.frame), len(fs.train), len(fs.upcoming), len(fs.feature_columns), FEATURE_TABLE)
    db.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
