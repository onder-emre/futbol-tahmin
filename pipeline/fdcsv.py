"""football-data.co.uk (GitHub aynası) → PostgreSQL veri çekme.

API-Football Free planı güncel sezonu ve maç istatistiklerini kapatıyor; ayrıca
Türkiye'de football-data.co.uk ISP düzeyinde engelli. Bu modül, aynı veriyi
GitHub üzerinden otomatik yansıtan `footballcsv/... ` benzeri bir aynadan çeker:
anahtar yok, kota yok, güncel sezon + geçmiş + gerçek istatistik (şut, korner,
faul, kart) var.

Kaynak CSV sütunları (football-data.co.uk ana format):
    Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,
    HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR, ...(bahis oranları)
"""
from __future__ import annotations

import hashlib
import io
import logging
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger(__name__)

# Varsayılan ayna (GitHub Action ile football-data.co.uk'den otomatik güncellenir).
DEFAULT_BASE = "https://raw.githubusercontent.com/Char2mant/futbol-veri-aynasi/main/data/fd"

# football-data.co.uk lig kodu -> (lig_id [API-Football ile aynı], ad, ulke)
LEAGUE_MAP: dict[str, tuple[int, str, str]] = {
    "T1":  (203, "Süper Lig",      "Türkiye"),
    "E0":  (39,  "Premier League", "İngiltere"),
    "SP1": (140, "La Liga",        "İspanya"),
    "I1":  (135, "Serie A",        "İtalya"),
    "D1":  (78,  "Bundesliga",     "Almanya"),
    "F1":  (61,  "Ligue 1",        "Fransa"),
}

# Maç saatleri football-data.co.uk'de Birleşik Krallık yerel saatidir.
_SRC_TZ = ZoneInfo("Europe/London")


def season_folder(start_year: int) -> str:
    """2026 -> '2627' (2026/27 sezonu klasörü)."""
    yy = start_year % 100
    return f"{yy:02d}{(yy + 1) % 100:02d}"


def _team_id(name: str) -> int:
    """İsimden kararlı (çalışmalar arası değişmeyen) pozitif INTEGER id."""
    return zlib.crc32(name.strip().casefold().encode("utf-8")) & 0x7FFFFFFF


def _match_id(div: str, date: str, home: str, away: str) -> int:
    key = f"{div}|{date}|{home}|{away}".casefold().encode("utf-8")
    # 60-bit -> BIGINT sınırının altında, 32-bit'ten çok daha düşük çakışma olasılığı
    return int(hashlib.sha1(key).hexdigest()[:15], 16)


def _to_utc(date_str: str, time_str: str | None) -> datetime | None:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            d = datetime.strptime(date_str.strip(), fmt).date()
            break
        except (ValueError, AttributeError):
            d = None
    if d is None:
        return None
    hh, mm = 15, 0  # saat yoksa makul varsayılan (öznitelikler güne duyarlı)
    if isinstance(time_str, str) and ":" in time_str:
        try:
            hh, mm = (int(x) for x in time_str.strip().split(":")[:2])
        except ValueError:
            pass
    local = datetime(d.year, d.month, d.day, hh, mm, tzinfo=_SRC_TZ)
    return local.astimezone(timezone.utc)


def _int(v) -> int | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
            return None
        return int(float(v))
    except (ValueError, TypeError):
        return None


@dataclass
class FdFrames:
    ligler: pd.DataFrame
    takimlar: pd.DataFrame
    maclar: pd.DataFrame
    istatistikler: pd.DataFrame


class FootballDataCsvSource:
    """Aynadan CSV indirir ve DB şemasına uygun DataFrame'lere dönüştürür."""

    def __init__(self, base_url: str = DEFAULT_BASE, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "futbol-tahmin/1.0"})

    def fetch_csv(self, start_year: int, div: str) -> pd.DataFrame | None:
        url = f"{self.base_url}/{season_folder(start_year)}/{div}.csv"
        try:
            r = self._session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            log.warning("%s indirilemedi: %s", url, exc)
            return None
        if r.status_code != 200 or not r.content:
            log.info("%s yok (HTTP %s)", url, r.status_code)
            return None
        df = pd.read_csv(io.BytesIO(r.content), encoding="utf-8-sig",
                         on_bad_lines="skip", low_memory=False)
        df = df[df.get("HomeTeam").notna() & df.get("FTHG").notna()] if "HomeTeam" in df else df.iloc[0:0]
        log.debug("%s -> %d oynanmış maç", url, len(df))
        return df

    def build(self, leagues: list[str], start_years: list[int]) -> FdFrames:
        teams: dict[int, dict] = {}
        matches: list[dict] = []
        stats: list[dict] = []
        now = datetime.now(timezone.utc)

        for year in start_years:
            for div in leagues:
                if div not in LEAGUE_MAP:
                    continue
                lig_id, _, ulke = LEAGUE_MAP[div]
                df = self.fetch_csv(year, div)
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    home, away = str(row["HomeTeam"]).strip(), str(row["AwayTeam"]).strip()
                    if not home or not away or home == away:
                        continue
                    tarih = _to_utc(str(row["Date"]), row.get("Time"))
                    if tarih is None:
                        continue
                    hid, aid = _team_id(home), _team_id(away)
                    teams.setdefault(hid, {"takim_id": hid, "ad": home, "ulke": ulke})
                    teams.setdefault(aid, {"takim_id": aid, "ad": away, "ulke": ulke})
                    mid = _match_id(div, str(row["Date"]), home, away)
                    matches.append({
                        "mac_id": mid, "lig_id": lig_id, "sezon": year, "hafta": None,
                        "tarih": tarih, "hakem": None, "stadyum": None, "durum": "FT",
                        "ev_sahibi_id": hid, "deplasman_id": aid,
                        "ev_gol": _int(row.get("FTHG")), "dep_gol": _int(row.get("FTAG")),
                        "ilk_yari_ev_gol": _int(row.get("HTHG")), "ilk_yari_dep_gol": _int(row.get("HTAG")),
                        "istatistik_cekim_zamani": now,
                    })
                    hs, as_ = _int(row.get("HS")), _int(row.get("AS"))
                    hst, ast = _int(row.get("HST")), _int(row.get("AST"))
                    for tid, sut, isabetli, korner, faul, sari, kirmizi in [
                        (hid, hs, hst, _int(row.get("HC")), _int(row.get("HF")), _int(row.get("HY")), _int(row.get("HR"))),
                        (aid, as_, ast, _int(row.get("AC")), _int(row.get("AF")), _int(row.get("AY")), _int(row.get("AR"))),
                    ]:
                        stats.append({
                            "mac_id": mid, "takim_id": tid,
                            "xg": None, "topla_oynama": None,
                            "toplam_sut": sut, "isabetli_sut": isabetli,
                            "isabetsiz_sut": (sut - isabetli) if sut is not None and isabetli is not None else None,
                            "engellenen_sut": None, "ceza_sahasi_ici_sut": None, "ceza_sahasi_disi_sut": None,
                            "korner": korner, "faul": faul, "ofsayt": None,
                            "sari_kart": sari, "kirmizi_kart": kirmizi,
                            "kaleci_kurtarisi": None, "toplam_pas": None, "isabetli_pas": None,
                            "pas_isabet_yuzdesi": None,
                        })
                log.info("Lig %s sezon %s: %d maç", div, year, len(df))

        ligler = pd.DataFrame(
            [{"lig_id": v[0], "ad": v[1], "ulke": v[2]} for k, v in LEAGUE_MAP.items() if k in leagues]
        )
        takimlar = pd.DataFrame(list(teams.values()))
        maclar = pd.DataFrame(matches)
        istatistikler = pd.DataFrame(stats)
        return FdFrames(ligler, takimlar, maclar, istatistikler)


# API-Football lig id -> football-data.co.uk Div kodu (günlük iş bu eşlemeyi kullanır)
LIG_ID_TO_DIV = {lig_id: code for code, (lig_id, _, _) in LEAGUE_MAP.items()}

_CONFLICT = {
    "ligler": ["lig_id"], "takimlar": ["takim_id"],
    "maclar": ["mac_id"], "istatistikler": ["mac_id", "takim_id"],
}


def write_frames(repo, engine, frames: FdFrames, truncate: bool = False) -> dict[str, int]:
    """FdFrames'i veritabanına yazar (upsert). truncate=True ise önce temizler.

    Tablo yansıtması transaction'dan ÖNCE ısıtılır — TRUNCATE kilidi dururken
    ayrı bağlantıdan yansıtma yapmak kendi kendine kilitlenmeye yol açardı.
    """
    from sqlalchemy import text

    for t in ("ligler", "takimlar", "maclar", "istatistikler"):
        repo.table(t)
    with engine.begin() as conn:
        if truncate:
            conn.execute(text("TRUNCATE tahminler, istatistikler, maclar, takimlar, ligler, "
                              "veri_cekim_durumu RESTART IDENTITY CASCADE"))
        repo.upsert_dataframe(conn, frames.ligler, "ligler", _CONFLICT["ligler"])
        repo.upsert_dataframe(conn, frames.takimlar, "takimlar", _CONFLICT["takimlar"])
        repo.upsert_dataframe(conn, frames.maclar, "maclar", _CONFLICT["maclar"])
        repo.upsert_dataframe(conn, frames.istatistikler, "istatistikler", _CONFLICT["istatistikler"])
    return {"lig": len(frames.ligler), "takim": len(frames.takimlar),
            "mac": len(frames.maclar), "istatistik": len(frames.istatistikler)}
