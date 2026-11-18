"""Modeli `mac_ozellikleri` tablosundan eğitir, zamana dayalı test eder ve kaydeder.

    python -m scripts.train_model                       # CV ile XGBoost / Random Forest arasından seç
    python -m scripts.train_model --algorithm xgboost
    python -m scripts.train_model --test-fraction 0.2
"""
from __future__ import annotations

import argparse
import logging
import sys

from config.settings import DatabaseSettings
from db.connection import DatabaseManager
from db.repository import FootballRepository
from models.features import FeatureSet
from models.predictor import ModelArtifact, ModelStore
from models.trainer import ALGORITHMS, ModelTrainer, TrainConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maç sonucu modelini eğit")
    parser.add_argument("--algorithm", choices=("auto", *ALGORITHMS), default="auto")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--min-history", type=int, default=3)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("train_model")

    db = DatabaseManager(DatabaseSettings.from_env())
    fs = FeatureSet.from_frame(FootballRepository(db.engine).load_feature_table())
    db.dispose()

    config = TrainConfig(algorithm=args.algorithm, test_fraction=args.test_fraction, min_history=args.min_history)
    result = ModelTrainer(config).train(fs)
    print("\n" + result.report() + "\n")

    path = ModelStore().save(ModelArtifact.from_training(result))
    log.info("Model kaydedildi: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
