"""Artımlı (incremental) veri çekme boru hattı.

Kota stratejisi:
  1. Tamamlanmış (veri_cekim_durumu.tamamlandi) lig-sezonlar hiç istenmez → 0 istek.
  2. Diğer lig-sezonlar için maç listesi tek istekte çekilir; sadece yeni/değişen maçlar yazılır.
  3. İstatistikler sadece "bitmiş ama istatistiği yok" maçlar için, 20'li gruplar halinde
     /fixtures?ids= ile çekilir → 20 maç = 1 istek.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from api.client import ApiFootballClient, QuotaExhaustedError
from api.transformers import FixtureFrames, FixtureTransformer
from config.settings import PipelineSettings, current_season
from db.repository import FootballRepository

logger = logging.getLogger(__name__)

_COMPARE_COLS = ["durum", "ev_gol", "dep_gol", "tarih"]


@dataclass
class IngestionReport:
    seasons_skipped: int = 0
    seasons_fetched: int = 0
    matches_written: int = 0
    stats_written: int = 0
    api_requests: int = 0
    stopped_by_quota: bool = False
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Atlanan sezon: {self.seasons_skipped} | Çekilen sezon: {self.seasons_fetched} | "
            f"Yazılan maç: {self.matches_written} | Yazılan istatistik satırı: {self.stats_written} | "
            f"API isteği: {self.api_requests} | Kota nedeniyle durdu: {self.stopped_by_quota} | "
            f"Hata: {len(self.errors)}"
        )


def select_changed_matches(api_matches: pd.DataFrame, db_snapshot: pd.DataFrame) -> pd.DataFrame:
    """API'den gelen maçlardan DB'de olmayan veya durumu/skoru/tarihi değişenleri döndürür."""
    if db_snapshot.empty:
        return api_matches
    merged = api_matches.merge(db_snapshot, on="mac_id", how="left", suffixes=("", "_db"), indicator=True)
    is_new = merged["_merge"] == "left_only"
    changed = pd.Series(False, index=merged.index)
    for col in _COMPARE_COLS:
        a, b = merged[col], merged[f"{col}_db"]
        changed |= ~((a == b).fillna(False) | (a.isna() & b.isna()))
    return api_matches.loc[(is_new | changed).to_numpy()].reset_index(drop=True)


class IngestionPipeline:
    def __init__(
        self,
        client: ApiFootballClient,
        repository: FootballRepository,
        settings: PipelineSettings,
        transformer: FixtureTransformer | None = None,
    ) -> None:
        self._client = client
        self._repo = repository
        self._settings = settings
        self._transformer = transformer or FixtureTransformer()

    def run(self, fetch_fixtures: bool = True, fetch_stats: bool = True) -> IngestionReport:
        report = IngestionReport()
        try:
            if fetch_fixtures:
                self._sync_fixtures(report)
            if fetch_stats:
                self._sync_statistics(report)
        except QuotaExhaustedError as exc:
            report.stopped_by_quota = True
            logger.warning("Kota sınırı: %s — kalan iş bir sonraki çalıştırmada devam edecek.", exc)
        report.api_requests = self._client.requests_made
        return report

    # -------------------------------------------------------------- fixtures
    def _sync_fixtures(self, report: IngestionReport) -> None:
        season_now = current_season()
        for league_id in self._settings.league_ids:
            for season in self._settings.seasons:
                if self._repo.is_season_completed(league_id, season):
                    report.seasons_skipped += 1
                    continue
                try:
                    self._sync_league_season(league_id, season, season_now, report)
                except QuotaExhaustedError:
                    raise
                except Exception as exc:  # tek sezonun hatası tüm pipeline'ı durdurmasın
                    logger.exception("Lig %s sezon %s başarısız", league_id, season)
                    report.errors.append(f"{league_id}/{season}: {exc}")

    def _sync_league_season(self, league_id: int, season: int, season_now: int, report: IngestionReport) -> None:
        fixtures = self._client.get_fixtures(league_id, season)
        report.seasons_fetched += 1
        frames = self._transformer.transform(fixtures)
        if frames.is_empty:
            logger.info("Lig %s sezon %s: API boş döndü (plan bu sezonu kapsamıyor olabilir).", league_id, season)
            return

        changed = select_changed_matches(frames.matches, self._repo.get_match_snapshot(league_id, season))
        if not changed.empty:
            team_ids = set(changed["ev_sahibi_id"]) | set(changed["deplasman_id"])
            counts = self._repo.save_fixture_frames(FixtureFrames(
                leagues=frames.leagues,
                teams=frames.teams[frames.teams["takim_id"].isin(team_ids)],
                matches=changed,
                statistics=frames.statistics,
            ))
            report.matches_written += counts["maclar"]

        completed = season < season_now and self._repo.all_final(frames.matches["durum"])
        self._repo.save_sync_state(league_id, season, len(frames.matches), completed)
        logger.info(
            "Lig %s sezon %s: %d maç, %d yeni/değişen%s",
            league_id, season, len(frames.matches), len(changed), " (sezon tamamlandı)" if completed else "",
        )

    # ------------------------------------------------------------ statistics
    def _sync_statistics(self, report: IngestionReport) -> None:
        missing = self._repo.get_fixture_ids_missing_stats(self._settings.league_ids, self._settings.seasons)
        logger.info("İstatistiği eksik bitmiş maç: %d (≈%d istek)",
                    len(missing), -(-len(missing) // ApiFootballClient.MAX_IDS_PER_REQUEST))
        step = ApiFootballClient.MAX_IDS_PER_REQUEST
        for start in range(0, len(missing), step):
            batch = missing[start:start + step]
            fixtures = self._client.get_fixtures_by_ids(batch)
            frames = self._transformer.transform(fixtures)
            if frames.is_empty:
                continue
            counts = self._repo.save_fixture_frames(frames, stats_requested_ids=batch)
            report.stats_written += counts["istatistikler"]
            logger.info("İstatistik: %d/%d maç işlendi", min(start + step, len(missing)), len(missing))
