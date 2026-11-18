"""Yaklaşan maçlar için olasılık tahmini üretir, ekrana yazar ve `tahminler` tablosuna kaydeder.

    python -m scripts.predict                 # önümüzdeki 7 gün
    python -m scripts.predict --days 3 --leagues 203
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config.settings import DatabaseSettings
from db.connection import DatabaseManager
from db.repository import FootballRepository
from models.features import FeatureSet
from models.predictor import MatchPredictor, ModelStore, format_prediction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Yaklaşan maçları tahmin et")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--leagues", type=int, nargs="+")
    parser.add_argument("--model", default="latest", help="Model versiyonu (varsayılan: latest)")
    parser.add_argument("--no-save", action="store_true", help="Veritabanına yazma")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("predict")

    db = DatabaseManager(DatabaseSettings.from_env())
    repo = FootballRepository(db.engine)
    fs = FeatureSet.from_frame(repo.load_feature_table())

    now = pd.Timestamp.now(tz="UTC")
    upcoming = fs.upcoming[(fs.upcoming["tarih"] >= now) & (fs.upcoming["tarih"] <= now + pd.Timedelta(days=args.days))]
    if args.leagues:
        upcoming = upcoming[upcoming["lig_id"].isin(args.leagues)]
    if upcoming.empty:
        log.info("Önümüzdeki %d günde tahmin edilecek maç yok (önce run_pipeline + build_features).", args.days)
        return 0

    artifact = ModelStore().load(args.model)
    preds = MatchPredictor(artifact).predict(upcoming)
    labels = repo.get_match_labels(preds["mac_id"].tolist())
    report = preds.merge(labels, on="mac_id").sort_values("tarih")

    print(f"\nModel: {artifact.version}\n")
    for lig, grp in report.groupby("lig", sort=False):
        print(f"── {lig} " + "─" * 40)
        for _, r in grp.iterrows():
            when = pd.Timestamp(r["tarih"]).tz_convert("Europe/Istanbul").strftime("%d.%m %H:%M")
            text = format_prediction(r["ev_sahibi"], r["deplasman"], r["olasilik_1"], r["olasilik_x"], r["olasilik_2"])
            print(f"{when}  {text}   → tahmin: {r['tahmin']}")
        print()

    if not args.no_save:
        n = repo.save_predictions(preds)
        log.info("%d tahmin `tahminler` tablosuna kaydedildi.", n)
    db.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
