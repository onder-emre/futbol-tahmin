"""API-Football JSON → düz (flat) pandas DataFrame dönüşümleri."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FINISHED_STATUSES = frozenset({"FT", "AET", "PEN"})
# Bir daha değişmeyecek durumlar (sezon tamamlandı kontrolü için)
FINAL_STATUSES = FINISHED_STATUSES | {"CANC", "ABD", "AWD", "WO"}

# API istatistik tipi → istatistikler tablosu kolonu
STAT_COLUMN_MAP: dict[str, str] = {
    "expected_goals": "xg",
    "Ball Possession": "topla_oynama",
    "Total Shots": "toplam_sut",
    "Shots on Goal": "isabetli_sut",
    "Shots off Goal": "isabetsiz_sut",
    "Blocked Shots": "engellenen_sut",
    "Shots insidebox": "ceza_sahasi_ici_sut",
    "Shots outsidebox": "ceza_sahasi_disi_sut",
    "Corner Kicks": "korner",
    "Fouls": "faul",
    "Offsides": "ofsayt",
    "Yellow Cards": "sari_kart",
    "Red Cards": "kirmizi_kart",
    "Goalkeeper Saves": "kaleci_kurtarisi",
    "Total passes": "toplam_pas",
    "Passes accurate": "isabetli_pas",
    "Passes %": "pas_isabet_yuzdesi",
}
DECIMAL_STAT_COLUMNS = ("xg", "topla_oynama", "pas_isabet_yuzdesi")
STAT_COLUMNS = ["mac_id", "takim_id", *STAT_COLUMN_MAP.values()]

# json_normalize(sep="_") çıktısı → maclar tablosu kolonu
_MATCH_COLUMN_MAP: dict[str, str] = {
    "fixture_id": "mac_id",
    "league_id": "lig_id",
    "league_season": "sezon",
    "league_round": "hafta",
    "fixture_date": "tarih",
    "fixture_referee": "hakem",
    "fixture_venue_name": "stadyum",
    "fixture_status_short": "durum",
    "teams_home_id": "ev_sahibi_id",
    "teams_away_id": "deplasman_id",
    "goals_home": "ev_gol",
    "goals_away": "dep_gol",
    "score_halftime_home": "ilk_yari_ev_gol",
    "score_halftime_away": "ilk_yari_dep_gol",
}
_INT_MATCH_COLUMNS = ("ev_gol", "dep_gol", "ilk_yari_ev_gol", "ilk_yari_dep_gol")


@dataclass(frozen=True)
class FixtureFrames:
    leagues: pd.DataFrame
    teams: pd.DataFrame
    matches: pd.DataFrame
    statistics: pd.DataFrame

    @property
    def is_empty(self) -> bool:
        return self.matches.empty


def parse_stat_value(value: Any) -> float:
    """'55%' → 55.0, '1.23' → 1.23, 7 → 7.0, None/'-' → NaN."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().rstrip("%"))
    except ValueError:
        return np.nan


class FixtureTransformer:
    """/fixtures yanıtını (istatistikli veya istatistiksiz) tablolara ayırır."""

    def transform(self, fixtures: list[dict[str, Any]]) -> FixtureFrames:
        if not fixtures:
            empty = pd.DataFrame()
            return FixtureFrames(empty, empty, empty, pd.DataFrame(columns=STAT_COLUMNS))

        flat = pd.json_normalize(fixtures, sep="_")
        return FixtureFrames(
            leagues=self._leagues(flat),
            teams=self._teams(flat),
            matches=self._matches(flat),
            statistics=self._statistics(fixtures),
        )

    @staticmethod
    def _leagues(flat: pd.DataFrame) -> pd.DataFrame:
        return (
            flat[["league_id", "league_name", "league_country", "league_logo"]]
            .rename(columns={"league_id": "lig_id", "league_name": "ad",
                             "league_country": "ulke", "league_logo": "logo_url"})
            .drop_duplicates("lig_id")
            .reset_index(drop=True)
        )

    @staticmethod
    def _teams(flat: pd.DataFrame) -> pd.DataFrame:
        frames = [
            flat[[f"teams_{side}_id", f"teams_{side}_name", f"teams_{side}_logo", "league_country"]]
            .set_axis(["takim_id", "ad", "logo_url", "ulke"], axis=1)
            for side in ("home", "away")
        ]
        return pd.concat(frames).drop_duplicates("takim_id").reset_index(drop=True)

    @staticmethod
    def _matches(flat: pd.DataFrame) -> pd.DataFrame:
        df = flat.reindex(columns=list(_MATCH_COLUMN_MAP)).rename(columns=_MATCH_COLUMN_MAP)
        df["tarih"] = pd.to_datetime(df["tarih"], utc=True)
        for col in _INT_MATCH_COLUMNS:
            df[col] = df[col].astype("Int64")
        return df.drop_duplicates("mac_id").reset_index(drop=True)

    @staticmethod
    def _statistics(fixtures: list[dict[str, Any]]) -> pd.DataFrame:
        long_frames = []
        for fx in fixtures:
            team_blocks = fx.get("statistics") or []
            if not team_blocks:
                continue
            long_df = pd.json_normalize(team_blocks, record_path="statistics", meta=[["team", "id"]])
            if long_df.empty:
                continue
            long_df["mac_id"] = fx["fixture"]["id"]
            long_frames.append(long_df)

        if not long_frames:
            return pd.DataFrame(columns=STAT_COLUMNS)

        long_df = pd.concat(long_frames, ignore_index=True).rename(columns={"team.id": "takim_id"})
        long_df["kolon"] = long_df["type"].map(STAT_COLUMN_MAP)
        long_df = long_df.dropna(subset=["kolon"])
        long_df["deger"] = long_df["value"].map(parse_stat_value)

        wide = (
            long_df.groupby(["mac_id", "takim_id", "kolon"])["deger"].first()
            .unstack("kolon")
            .reset_index()
            .reindex(columns=STAT_COLUMNS)
        )
        wide.columns.name = None
        int_cols = [c for c in STAT_COLUMNS[2:] if c not in DECIMAL_STAT_COLUMNS]
        wide[int_cols] = wide[int_cols].round().astype("Int64")
        wide["mac_id"] = wide["mac_id"].astype("int64")
        wide["takim_id"] = wide["takim_id"].astype("int64")
        return wide
