"""FAZ 4/5 — Günlük iş: veri → günün fikstürü → öznitelik → (gerekirse) eğitim → tahmin → Telegram.

Her adım bağımsız loglanır. Veri çekme adımı başarısız olsa bile (ör. API kotası),
veritabanındaki mevcut veriyle tahmin ve bildirim yine yapılır.
Aynı gün iki kez çalışırsa aynı maçlar için ikinci mesaj ATILMAZ (tahminler.bildirim_gonderildi).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from api.client import ApiFootballClient, QuotaExhaustedError
from bot.formatter import MessageFormatter
from bot.notifier import TelegramNotifier
from config.settings import DailyJobSettings, Settings
from db.repository import FootballRepository
from models.dataset import build_and_store_features
from models.features import FeatureSet
from models.predictor import MatchPredictor, ModelArtifact, ModelStore
from models.trainer import ModelTrainer, TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class DailyReport:
    steps: dict[str, str] = field(default_factory=dict)
    todays_matches: int = 0
    predictions: int = 0
    confident: int = 0
    messages_sent: int = 0

    @property
    def ok(self) -> bool:
        return not any(v.startswith("HATA") for v in self.steps.values())

    def __str__(self) -> str:
        lines = [f"  {k:<12}: {v}" for k, v in self.steps.items()]
        lines.append(f"  Bugünkü maç: {self.todays_matches} | Tahmin: {self.predictions} | "
                     f"Güvenilir: {self.confident} | Gönderilen mesaj: {self.messages_sent}")
        return "\n".join(lines)


class DailyJob:
    def __init__(
        self,
        settings: Settings,
        daily: DailyJobSettings,
        repository: FootballRepository,
        client: ApiFootballClient | None,
        notifier: TelegramNotifier | None,
        model_store: ModelStore | None = None,
        dry_run: bool = False,
    ) -> None:
        self.settings = settings
        self.daily = daily
        self.repo = repository
        self.client = client
        self.notifier = notifier
        self.store = model_store or ModelStore()
        self.dry_run = dry_run
        self.tz = ZoneInfo(daily.timezone)

    # ------------------------------------------------------------------ run
    def run(self, skip_ingest: bool = False, force_retrain: bool = False, today: pd.Timestamp | None = None) -> DailyReport:
        report = DailyReport()
        today = (today or pd.Timestamp.now(tz=self.tz)).normalize()
        if self.client is not None:
            self.client.request_budget = self.daily.ingest_max_requests

        self._step(report, "veri", lambda: "atlandı" if skip_ingest else self._ingest())
        self._step(report, "fikstür", lambda: "atlandı" if skip_ingest else self._fetch_todays_fixtures(today))
        fs = self._step(report, "öznitelik", self._features)
        if fs is None:
            return report
        artifact = self._step(report, "model", lambda: self._ensure_model(fs, force_retrain))
        if artifact is None:
            return report
        preds = self._step(report, "tahmin", lambda: self._predict_today(fs, artifact, today, report))
        if preds is not None:
            self._step(report, "telegram", lambda: self._notify(preds, artifact, today, report))
        return report

    def _step(self, report: DailyReport, name: str, fn):
        start = time.monotonic()
        try:
            result = fn()
            summary = result if isinstance(result, str) else "tamam"
            report.steps[name] = f"{summary} ({time.monotonic() - start:.1f}s)"
            logger.info("[%s] %s", name, report.steps[name])
            return result
        except QuotaExhaustedError as exc:
            report.steps[name] = f"kota sınırı: {exc}"
            logger.warning("[%s] %s", name, report.steps[name])
            return None
        except Exception as exc:
            report.steps[name] = f"HATA: {type(exc).__name__}: {exc}"
            logger.exception("[%s] başarısız", name)
            return None

    # ---------------------------------------------------------------- steps
    def _ingest(self) -> str:
        """Güncel sezonu ücretsiz football-data.co.uk aynasından yeniler (anahtar/kota yok)."""
        from config.settings import current_season
        from pipeline.fdcsv import LIG_ID_TO_DIV, FootballDataCsvSource, write_frames

        divs = [LIG_ID_TO_DIV[lid] for lid in self.settings.pipeline.league_ids if lid in LIG_ID_TO_DIV]
        season = current_season()
        frames = FootballDataCsvSource().build(divs, [season])
        if frames.maclar.empty:
            return "aynadan güncel maç gelmedi (sezon henüz başlamamış olabilir)"
        n = write_frames(self.repo, self.repo.engine, frames, truncate=False)
        return f"ayna: {n['mac']} maç, {n['istatistik']} istatistik ({season}/{season + 1} sezonu)"

    def _fetch_todays_fixtures(self, today: pd.Timestamp) -> str:
        """Gelecek maçları TheSportsDB'den çeker (ücretsiz), DB takımlarına eşler, durum='NS' yazar."""
        from sqlalchemy import text

        from pipeline.fixtures_live import LiveFixtureSource

        league_ids = list(self.settings.pipeline.league_ids)
        df = LiveFixtureSource().upcoming_matches(self.repo, league_ids, days_ahead=7)
        self.repo.table("maclar")  # yansıtmayı transaction öncesi ısıt
        with self.repo.engine.begin() as conn:
            # Bayat 'NS' satırlarını temizle (her çalışmada taze fikstür)
            conn.execute(text("DELETE FROM maclar WHERE durum='NS'"))
            if not df.empty:
                self.repo.upsert_dataframe(conn, df, "maclar", ["mac_id"])
        if df.empty:
            return "önümüzdeki 7 günde takip edilen liglerde maç bulunamadı"
        return f"{len(df)} gelecek maç (TheSportsDB)"

    def _features(self) -> FeatureSet:
        fs = build_and_store_features(self.repo.engine, list(self.settings.pipeline.league_ids))
        if fs.frame.empty:
            raise RuntimeError("Veritabanında maç yok — önce veri çekilmeli.")
        return fs

    def _ensure_model(self, fs: FeatureSet, force: bool) -> ModelArtifact:
        latest = self.store.directory / "latest.joblib"
        age_days = (time.time() - latest.stat().st_mtime) / 86_400 if latest.exists() else None
        if force or age_days is None or age_days > self.daily.retrain_days:
            logger.info("Model yeniden eğitiliyor (yaş: %s gün)", None if age_days is None else round(age_days, 1))
            result = ModelTrainer(TrainConfig()).train(fs)
            logger.info("\n%s", result.report())
            artifact = ModelArtifact.from_training(result)
            self.store.save(artifact)
            return artifact
        return self.store.load()

    def _predict_today(self, fs: FeatureSet, artifact: ModelArtifact, today: pd.Timestamp,
                       report: DailyReport) -> pd.DataFrame:
        up = fs.upcoming
        local_day = up["tarih"].dt.tz_convert(self.tz).dt.normalize()
        todays = up[(local_day == today) & (up["tarih"] > datetime.now(timezone.utc))]
        report.todays_matches = len(todays)
        if todays.empty:
            return pd.DataFrame()
        preds = MatchPredictor(artifact).predict(todays)
        self.repo.save_predictions(preds)
        report.predictions = len(preds)
        return preds.merge(self.repo.get_match_labels(preds["mac_id"].tolist()), on="mac_id")

    def _notify(self, preds: pd.DataFrame, artifact: ModelArtifact, today: pd.Timestamp, report: DailyReport) -> str:
        formatter = MessageFormatter(threshold=self.daily.confidence_threshold, timezone=self.daily.timezone)
        if preds.empty:
            return "bugün maç yok, mesaj atılmadı"
        pending_ids = self.repo.get_unnotified_ids(preds["mac_id"].tolist())
        pending = preds[preds["mac_id"].isin(pending_ids)]
        if pending.empty:
            return "bugünün tahminleri zaten gönderilmiş"

        report.confident = len(formatter.select_confident(pending))
        messages = formatter.build(pending, today, artifact.version)
        if self.dry_run or self.notifier is None:
            print("\n\n".join(messages))
            return f"dry-run: {len(messages)} mesaj ekrana yazıldı"
        self.notifier.send(messages)
        self.repo.mark_notified(pending["mac_id"].tolist())
        report.messages_sent = len(messages)
        return f"{len(messages)} mesaj gönderildi ({report.confident} güvenilir tahmin)"
