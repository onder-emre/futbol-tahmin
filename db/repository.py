"""Veritabanı erişim katmanı: upsert ve pipeline'ın ihtiyaç duyduğu sorgular."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import Connection, Engine, MetaData, Table, bindparam, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from api.transformers import FINAL_STATUSES, FINISHED_STATUSES, FixtureFrames

_CHUNK_SIZE = 1_000


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """NaN/NA → None, numpy tipleri → Python tipleri (psycopg2 uyumlu)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


class FootballRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._metadata = MetaData()
        self._tables: dict[str, Table] = {}

    # --------------------------------------------------------------- generic
    def table(self, name: str) -> Table:
        if name not in self._tables:
            self._tables[name] = Table(name, self._metadata, autoload_with=self._engine)
        return self._tables[name]

    def upsert_dataframe(
        self,
        conn: Connection,
        df: pd.DataFrame,
        table_name: str,
        conflict_cols: Sequence[str],
        update_cols: Sequence[str] | None = None,
    ) -> int:
        """INSERT ... ON CONFLICT (conflict_cols) DO UPDATE.

        update_cols verilmezse DataFrame'deki tüm anahtar-dışı kolonlar güncellenir.
        DataFrame'de olmayan kolonlara (ör. istatistik_cekim_zamani) dokunulmaz.
        """
        if df.empty:
            return 0
        table = self.table(table_name)
        update_cols = list(update_cols) if update_cols is not None else [
            c for c in df.columns if c not in conflict_cols
        ]
        records = dataframe_to_records(df)
        for start in range(0, len(records), _CHUNK_SIZE):
            stmt = pg_insert(table).values(records[start:start + _CHUNK_SIZE])
            if update_cols:
                stmt = stmt.on_conflict_do_update(
                    index_elements=list(conflict_cols),
                    set_={c: stmt.excluded[c] for c in update_cols},
                )
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=list(conflict_cols))
            conn.execute(stmt)
        return len(records)

    # ----------------------------------------------------------- domain write
    def save_fixture_frames(self, frames: FixtureFrames, stats_requested_ids: Iterable[int] = ()) -> dict[str, int]:
        """Lig → takım → maç → istatistik sırasıyla tek transaction'da yazar (FK sırası)."""
        counts = {"ligler": 0, "takimlar": 0, "maclar": 0, "istatistikler": 0}
        with self._engine.begin() as conn:
            counts["ligler"] = self.upsert_dataframe(conn, frames.leagues, "ligler", ["lig_id"])
            counts["takimlar"] = self.upsert_dataframe(conn, frames.teams, "takimlar", ["takim_id"])
            counts["maclar"] = self.upsert_dataframe(conn, frames.matches, "maclar", ["mac_id"])
            counts["istatistikler"] = self.upsert_dataframe(
                conn, frames.statistics, "istatistikler", ["mac_id", "takim_id"]
            )
            self._mark_stats_fetched(conn, frames, stats_requested_ids)
        return counts

    def _mark_stats_fetched(self, conn: Connection, frames: FixtureFrames, requested_ids: Iterable[int]) -> None:
        """İstatistiği gelen maçları işaretle. İstatistiği boş gelenleri, maç 3 günden eskiyse
        yine işaretle (API'de hiç yok demektir, kotayı boşa harcamayalım); yeni maçlar sonra tekrar denenir."""
        requested = set(requested_ids)
        if not requested:
            return
        with_stats = set(frames.statistics["mac_id"].tolist()) if not frames.statistics.empty else set()
        cutoff = datetime.now(timezone.utc) - timedelta(days=3)
        old_matches = set(frames.matches.loc[frames.matches["tarih"] < cutoff, "mac_id"].tolist())
        to_mark = requested & (with_stats | old_matches)
        if not to_mark:
            return
        maclar = self.table("maclar")
        conn.execute(
            update(maclar)
            .where(maclar.c.mac_id.in_(bindparam("ids", expanding=True)))
            .values(istatistik_cekim_zamani=text("NOW()")),
            {"ids": sorted(to_mark)},
        )

    def save_predictions(self, predictions: pd.DataFrame) -> int:
        """Tahminleri upsert eder (bildirim_gonderildi bayrağına dokunmaz)."""
        cols = ["mac_id", "model_versiyonu", "olasilik_1", "olasilik_x", "olasilik_2", "tahmin", "guven"]
        df = predictions[cols].copy()
        for c in ("olasilik_1", "olasilik_x", "olasilik_2", "guven"):
            df[c] = df[c].astype(float).round(4)
        with self._engine.begin() as conn:
            return self.upsert_dataframe(conn, df, "tahminler", ["mac_id"])

    def get_match_labels(self, match_ids: Sequence[int]) -> pd.DataFrame:
        """Tahmin çıktısı için takım/lig adları."""
        sql = text("""
            SELECT m.mac_id, m.tarih, l.ad AS lig, th.ad AS ev_sahibi, ta.ad AS deplasman
            FROM maclar m
            JOIN ligler l ON l.lig_id = m.lig_id
            JOIN takimlar th ON th.takim_id = m.ev_sahibi_id
            JOIN takimlar ta ON ta.takim_id = m.deplasman_id
            WHERE m.mac_id IN :ids
        """).bindparams(bindparam("ids", expanding=True))
        with self._engine.connect() as conn:
            return pd.read_sql(sql, conn, params={"ids": list(match_ids) or [-1]})

    def load_feature_table(self, table_name: str = "mac_ozellikleri") -> pd.DataFrame:
        with self._engine.connect() as conn:
            return pd.read_sql(text(f'SELECT * FROM "{table_name}"'), conn)

    def save_sync_state(self, league_id: int, season: int, match_count: int, completed: bool) -> None:
        df = pd.DataFrame([{
            "lig_id": league_id, "sezon": season, "mac_sayisi": match_count,
            "tamamlandi": completed, "son_cekim_zamani": datetime.now(timezone.utc),
        }])
        with self._engine.begin() as conn:
            self.upsert_dataframe(conn, df, "veri_cekim_durumu", ["lig_id", "sezon"])

    # ------------------------------------------------------------ domain read
    def is_season_completed(self, league_id: int, season: int) -> bool:
        t = self.table("veri_cekim_durumu")
        with self._engine.connect() as conn:
            return bool(conn.execute(
                select(t.c.tamamlandi).where(t.c.lig_id == league_id, t.c.sezon == season)
            ).scalar())

    def get_match_snapshot(self, league_id: int, season: int) -> pd.DataFrame:
        """Değişiklik tespiti için mevcut maçların özet durumu."""
        sql = text("""
            SELECT mac_id, durum, ev_gol, dep_gol, tarih
            FROM maclar WHERE lig_id = :lig AND sezon = :sezon
        """)
        with self._engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"lig": league_id, "sezon": season})
        df["tarih"] = pd.to_datetime(df["tarih"], utc=True)
        for col in ("ev_gol", "dep_gol"):
            df[col] = df[col].astype("Int64")
        return df

    def get_fixture_ids_missing_stats(
        self, league_ids: Sequence[int], seasons: Sequence[int], limit: int | None = None
    ) -> list[int]:
        """Bitmiş ama istatistiği henüz çekilmemiş maçlar (en yeniden eskiye)."""
        maclar = self.table("maclar")
        stmt = (
            select(maclar.c.mac_id)
            .where(
                maclar.c.istatistik_cekim_zamani.is_(None),
                maclar.c.durum.in_(sorted(FINISHED_STATUSES)),
                maclar.c.lig_id.in_(list(league_ids)),
                maclar.c.sezon.in_(list(seasons)),
            )
            .order_by(maclar.c.tarih.desc())
        )
        if limit:
            stmt = stmt.limit(limit)
        with self._engine.connect() as conn:
            return list(conn.execute(stmt).scalars())

    @staticmethod
    def all_final(statuses: Iterable[str]) -> bool:
        statuses = list(statuses)
        return bool(statuses) and all(s in FINAL_STATUSES for s in statuses)
