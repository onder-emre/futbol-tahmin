"""FAZ 2 — Öznitelik mühendisliği.

Her maç için, maçtan ÖNCEKİ bilgiyle şunlar hesaplanır:
  • Genel form        : takımın son N maçı (iç saha + deplasman karışık)
  • Saha formu        : ev sahibinin son N İÇ SAHA maçı, deplasmanın son N DEPLASMAN maçı
  • H2H               : iki takımın kendi aralarındaki son N maç
  • Dinlenme günü     : son maçtan bu yana geçen gün

VERİ SIZINTISI (data leakage) KORUMASI
  Oynanmış maçlarda her istatistik serisi önce .shift(1) ile bir maç kaydırılır, sonra
  .rolling(N) uygulanır → k. maçın öznitelikleri yalnızca k-N … k-1 maçlarından gelir;
  maçın kendi skoru/xG'si/topla oynaması asla kendi satırına girmez.
  Henüz oynanmamış maçlar (tahmin edilecekler) için, o tarihten KESİN ÖNCE oynanan son
  maça kadarki rolling değeri merge_asof(allow_exact_matches=False) ile alınır.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from api.transformers import FINISHED_STATUSES

META_COLUMNS = ["mac_id", "lig_id", "sezon", "tarih", "durum", "ev_sahibi_id", "deplasman_id"]
TARGET_COLUMNS = ["sonuc", "ev_gol", "dep_gol"]

# (çıktı adı, kaynak kolon, toplama yöntemi)
FORM_SPECS: list[tuple[str, str, str]] = [
    ("atilan_gol_ort", "atilan_gol", "mean"),
    ("yenilen_gol_ort", "yenilen_gol", "mean"),
    ("xg_ort", "xg", "mean"),
    ("xg_yenilen_ort", "xg_yenilen", "mean"),
    ("topla_oynama_ort", "topla_oynama", "mean"),
    ("topla_oynama_trend", "topla_oynama", "slope"),   # maç başına % değişim (eğim)
    ("sut_ort", "sut", "mean"),
    ("isabetli_sut_ort", "isabetli_sut", "mean"),
    ("puan_ort", "puan", "mean"),
    ("mac_sayisi", "atilan_gol", "count"),
]

H2H_SPECS: list[tuple[str, str, str]] = [
    ("dusuk_kazanma_orani", "dusuk_kazandi", "mean"),
    ("beraberlik_orani", "beraberlik", "mean"),
    ("yuksek_kazanma_orani", "yuksek_kazandi", "mean"),
    ("dusuk_gol_farki_ort", "dusuk_gol_farki", "mean"),
    ("toplam_gol_ort", "toplam_gol", "mean"),
    ("mac_sayisi", "toplam_gol", "count"),
]

# Ev - deplasman fark öznitelikleri (genel formdan)
DIFF_METRICS = ["atilan_gol_ort", "yenilen_gol_ort", "xg_ort", "xg_yenilen_ort", "topla_oynama_ort", "puan_ort"]


@dataclass(frozen=True)
class FeatureConfig:
    window: int = 5
    h2h_window: int = 5
    min_periods: int = 1


@dataclass
class FeatureSet:
    frame: pd.DataFrame
    feature_columns: list[str] = field(default_factory=list)
    target_column: str = "sonuc"

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "FeatureSet":
        """Veritabanındaki `mac_ozellikleri` tablosundan FeatureSet oluşturur."""
        frame = frame.copy()
        frame["tarih"] = pd.to_datetime(frame["tarih"], utc=True)
        features = [c for c in frame.columns if c not in META_COLUMNS + TARGET_COLUMNS]
        return cls(frame=frame.sort_values(["tarih", "mac_id"]).reset_index(drop=True), feature_columns=features)

    @property
    def train(self) -> pd.DataFrame:
        """Sonucu belli (oynanmış) maçlar."""
        return self.frame[self.frame["sonuc"].notna()]

    @property
    def upcoming(self) -> pd.DataFrame:
        """Tahmin edilecek (oynanmamış) maçlar."""
        return self.frame[self.frame["sonuc"].isna()]


def _slope(values: np.ndarray) -> float:
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(values))[mask]
    return float(np.polyfit(x, values[mask], 1)[0])


def _roll(series: pd.Series, how: str, window: int, min_periods: int) -> pd.Series:
    r = series.rolling(window, min_periods=min_periods)
    if how == "mean":
        return r.mean()
    if how == "count":
        return r.count()
    if how == "slope":
        return r.apply(_slope, raw=True)
    raise ValueError(how)


class FeatureBuilder:
    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    # ================================================================ public
    def build(self, matches: pd.DataFrame) -> FeatureSet:
        m = self._prepare(matches)
        long = self._to_team_long(m)

        general = self._rolling_block(long, ["takim_id"], FORM_SPECS, self.config.window, ["mac_id", "takim_id"])
        venue = self._rolling_block(long, ["takim_id", "saha"], FORM_SPECS, self.config.window, ["mac_id", "takim_id"])
        rest = self._rest_days(long)
        team = (general.add_prefix("genel_").rename(columns={"genel_mac_id": "mac_id", "genel_takim_id": "takim_id"})
                .merge(venue.add_prefix("saha_").rename(columns={"saha_mac_id": "mac_id", "saha_takim_id": "takim_id"}),
                       on=["mac_id", "takim_id"])
                .merge(rest, on=["mac_id", "takim_id"]))

        out = m[META_COLUMNS + TARGET_COLUMNS].copy()
        for side, id_col in (("ev", "ev_sahibi_id"), ("dep", "deplasman_id")):
            side_feats = team.rename(columns={"takim_id": id_col}).set_index(["mac_id", id_col]).add_prefix(f"{side}_")
            out = out.merge(side_feats, left_on=["mac_id", id_col], right_index=True, how="left")

        out = out.merge(self._h2h(m), on="mac_id", how="left")
        for metric in DIFF_METRICS:
            out[f"fark_genel_{metric}"] = out[f"ev_genel_{metric}"] - out[f"dep_genel_{metric}"]
            out[f"fark_saha_{metric}"] = out[f"ev_saha_{metric}"] - out[f"dep_saha_{metric}"]

        count_cols = [c for c in out.columns if c.endswith("mac_sayisi")]
        out[count_cols] = out[count_cols].fillna(0)
        out = out.sort_values(["tarih", "mac_id"]).reset_index(drop=True)
        features = [c for c in out.columns if c not in META_COLUMNS + TARGET_COLUMNS]
        return FeatureSet(frame=out, feature_columns=features)

    # =============================================================== private
    @staticmethod
    def _prepare(matches: pd.DataFrame) -> pd.DataFrame:
        m = matches.copy()
        m["tarih"] = pd.to_datetime(m["tarih"], utc=True)
        m["oynandi"] = m["durum"].isin(FINISHED_STATUSES) & m["ev_gol"].notna() & m["dep_gol"].notna()
        # Oynanmamış maçta sonuç bilgisi olmamalı (güvenlik)
        m.loc[~m["oynandi"], ["ev_gol", "dep_gol"]] = np.nan
        m["sonuc"] = np.select(
            [m["ev_gol"] > m["dep_gol"], m["ev_gol"] < m["dep_gol"], m["ev_gol"] == m["dep_gol"]],
            ["H", "A", "D"], default=None,
        )
        m.loc[~m["oynandi"], "sonuc"] = None
        return m.sort_values(["tarih", "mac_id"]).reset_index(drop=True)

    @staticmethod
    def _to_team_long(m: pd.DataFrame) -> pd.DataFrame:
        """Her maçı takım bakış açısıyla 2 satıra açar."""
        def side(own: str, opp: str, id_col: str) -> pd.DataFrame:
            df = pd.DataFrame({
                "mac_id": m["mac_id"], "tarih": m["tarih"], "oynandi": m["oynandi"], "saha": own,
                "takim_id": m[id_col],
                "atilan_gol": m[f"{own}_gol"], "yenilen_gol": m[f"{opp}_gol"],
                "xg": m.get(f"{own}_xg"), "xg_yenilen": m.get(f"{opp}_xg"),
                "topla_oynama": m.get(f"{own}_topla_oynama"),
                "sut": m.get(f"{own}_sut"), "isabetli_sut": m.get(f"{own}_isabetli_sut"),
            })
            df["puan"] = np.select([df["atilan_gol"] > df["yenilen_gol"], df["atilan_gol"] == df["yenilen_gol"]],
                                   [3.0, 1.0], default=0.0)
            df.loc[~df["oynandi"], "puan"] = np.nan
            return df

        long = pd.concat([side("ev", "dep", "ev_sahibi_id"), side("dep", "ev", "deplasman_id")], ignore_index=True)
        num = ["atilan_gol", "yenilen_gol", "xg", "xg_yenilen", "topla_oynama", "sut", "isabetli_sut", "puan"]
        long[num] = long[num].astype("float64")
        return long.sort_values(["tarih", "mac_id"]).reset_index(drop=True)

    def _rolling_block(
        self, df: pd.DataFrame, keys: list[str], specs: list[tuple[str, str, str]], window: int, id_cols: list[str],
    ) -> pd.DataFrame:
        """keys gruplarında rolling öznitelik üretir.

        Oynanmış satırlar  : shift(1) → rolling(window)          (maçın kendisi hariç)
        Oynanmamış satırlar: tarihten kesin önceki son 'rolling (shift'siz)' değer (merge_asof)
        """
        mp = self.config.min_periods
        played = df[df["oynandi"]].sort_values(["tarih", "mac_id"])
        grouped = played.groupby(keys, sort=False)

        pre = pd.DataFrame(index=played.index)
        post = pd.DataFrame(index=played.index)
        for name, src, how in specs:
            pre[name] = grouped[src].transform(lambda s, h=how: _roll(s.shift(1), h, window, mp))
            post[name] = grouped[src].transform(lambda s, h=how: _roll(s, h, window, mp))

        played_out = pd.concat([played[id_cols], pre], axis=1)

        upcoming = df[~df["oynandi"]]
        if upcoming.empty:
            return played_out.reset_index(drop=True)

        right = pd.concat([played[keys + ["tarih"]], post], axis=1).sort_values("tarih")
        left_cols = list(dict.fromkeys(id_cols + keys + ["tarih"]))
        up = pd.merge_asof(
            upcoming[left_cols].sort_values("tarih"), right,
            on="tarih", by=keys, allow_exact_matches=False, direction="backward",
        )
        up_out = up[id_cols + [s[0] for s in specs]]
        return pd.concat([played_out, up_out], ignore_index=True)

    @staticmethod
    def _rest_days(long: pd.DataFrame) -> pd.DataFrame:
        """Takımın bu maçtan kesin önce oynadığı son maçtan bu yana geçen gün."""
        played = long.loc[long["oynandi"], ["takim_id", "tarih"]].sort_values("tarih")
        played = played.assign(son_mac_tarihi=played["tarih"])
        left = long[["mac_id", "takim_id", "tarih"]].sort_values("tarih")
        merged = pd.merge_asof(left, played, on="tarih", by="takim_id", allow_exact_matches=False)
        merged["dinlenme_gun"] = (merged["tarih"] - merged["son_mac_tarihi"]).dt.total_seconds() / 86_400
        return merged[["mac_id", "takim_id", "dinlenme_gun"]]

    def _h2h(self, m: pd.DataFrame) -> pd.DataFrame:
        """İki takımın aralarındaki son N maç, mevcut maçın EV SAHİBİ bakış açısından."""
        h = m[["mac_id", "tarih", "oynandi", "ev_sahibi_id", "deplasman_id", "ev_gol", "dep_gol"]].copy()
        h["dusuk_id"] = h[["ev_sahibi_id", "deplasman_id"]].min(axis=1)
        h["cift"] = h["dusuk_id"].astype(str) + "-" + h[["ev_sahibi_id", "deplasman_id"]].max(axis=1).astype(str)

        ev_dusuk = h["ev_sahibi_id"] == h["dusuk_id"]
        dusuk_gol = np.where(ev_dusuk, h["ev_gol"], h["dep_gol"])
        yuksek_gol = np.where(ev_dusuk, h["dep_gol"], h["ev_gol"])
        played = h["oynandi"]
        h["dusuk_kazandi"] = np.where(played, (dusuk_gol > yuksek_gol).astype(float), np.nan)
        h["yuksek_kazandi"] = np.where(played, (dusuk_gol < yuksek_gol).astype(float), np.nan)
        h["beraberlik"] = np.where(played, (dusuk_gol == yuksek_gol).astype(float), np.nan)
        h["dusuk_gol_farki"] = dusuk_gol - yuksek_gol
        h["toplam_gol"] = h["ev_gol"] + h["dep_gol"]

        r = self._rolling_block(h, ["cift"], H2H_SPECS, self.config.h2h_window, ["mac_id"])
        r = r.merge(h[["mac_id", "ev_sahibi_id", "dusuk_id"]], on="mac_id")
        ev_is_low = r["ev_sahibi_id"] == r["dusuk_id"]
        return pd.DataFrame({
            "mac_id": r["mac_id"],
            "h2h_mac_sayisi": r["mac_sayisi"],
            "h2h_ev_kazanma_orani": np.where(ev_is_low, r["dusuk_kazanma_orani"], r["yuksek_kazanma_orani"]),
            "h2h_beraberlik_orani": r["beraberlik_orani"],
            "h2h_dep_kazanma_orani": np.where(ev_is_low, r["yuksek_kazanma_orani"], r["dusuk_kazanma_orani"]),
            "h2h_ev_gol_farki_ort": np.where(ev_is_low, r["dusuk_gol_farki_ort"], -r["dusuk_gol_farki_ort"]),
            "h2h_toplam_gol_ort": r["toplam_gol_ort"],
        })
