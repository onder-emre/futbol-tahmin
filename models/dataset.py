"""Ham maç + istatistik verisini PostgreSQL'den modelleme için yükler."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, bindparam, text

from api.transformers import FINISHED_STATUSES
from models.features import FeatureBuilder, FeatureConfig, FeatureSet

FEATURE_TABLE = "mac_ozellikleri"

UPCOMING_STATUSES = ("NS", "TBD")

_MATCH_SQL = text("""
    SELECT
        m.mac_id, m.lig_id, m.sezon, m.tarih, m.durum,
        m.ev_sahibi_id, m.deplasman_id, m.ev_gol, m.dep_gol, m.sonuc,
        sh.xg            AS ev_xg,            sa.xg            AS dep_xg,
        sh.topla_oynama  AS ev_topla_oynama,  sa.topla_oynama  AS dep_topla_oynama,
        sh.toplam_sut    AS ev_sut,           sa.toplam_sut    AS dep_sut,
        sh.isabetli_sut  AS ev_isabetli_sut,  sa.isabetli_sut  AS dep_isabetli_sut,
        sh.korner        AS ev_korner,        sa.korner        AS dep_korner
    FROM maclar m
    LEFT JOIN istatistikler sh ON sh.mac_id = m.mac_id AND sh.takim_id = m.ev_sahibi_id
    LEFT JOIN istatistikler sa ON sa.mac_id = m.mac_id AND sa.takim_id = m.deplasman_id
    WHERE m.durum IN :durumlar
      AND (:lig_ids_bos OR m.lig_id IN :lig_ids)
    ORDER BY m.tarih, m.mac_id
""").bindparams(bindparam("durumlar", expanding=True), bindparam("lig_ids", expanding=True))


class MatchDataLoader:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def load(self, league_ids: list[int] | None = None) -> pd.DataFrame:
        """Oynanmış (FT/AET/PEN) ve oynanacak (NS/TBD) maçları tek tabloda döndürür."""
        params = {
            "durumlar": sorted(FINISHED_STATUSES) + list(UPCOMING_STATUSES),
            "lig_ids": league_ids or [-1],
            "lig_ids_bos": not league_ids,
        }
        with self._engine.connect() as conn:
            df = pd.read_sql(_MATCH_SQL, conn, params=params)
        df["tarih"] = pd.to_datetime(df["tarih"], utc=True)
        num_cols = df.columns.difference(["mac_id", "lig_id", "sezon", "tarih", "durum", "sonuc",
                                          "ev_sahibi_id", "deplasman_id"])
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        return df


def build_and_store_features(
    engine: Engine, league_ids: list[int] | None = None, config: FeatureConfig | None = None,
) -> FeatureSet:
    """Ham veriden öznitelikleri hesaplar ve `mac_ozellikleri` tablosunu yeniler."""
    matches = MatchDataLoader(engine).load(league_ids)
    if matches.empty:
        return FeatureSet(frame=pd.DataFrame(columns=["mac_id", "tarih", "sonuc", "lig_id"]))
    fs = FeatureBuilder(config).build(matches)
    fs.frame.to_sql(FEATURE_TABLE, engine, if_exists="replace", index=False, chunksize=2_000, method="multi")
    return fs
