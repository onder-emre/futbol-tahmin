"""Gelecek (oynanmamış) maçların fikstürü — TheSportsDB üzerinden.

API-Football Free planı yarın ve sonrasını kapatıyor; football-data.co.uk
Türkiye'de engelli. TheSportsDB'nin ücretsiz anahtarı ('123') ise takip edilen
6 ligin sıradaki maçlarını veriyor (Süper Lig dahil). Bu modül o fikstürü çeker,
takım isimlerini aynadan gelen (DB'deki) isimlerle eşleştirir ve DB'ye durum='NS'
olarak yazar; böylece öznitelik/tahmin boru hattı gelecek maçlar için çalışır.
"""
from __future__ import annotations

import logging
import time
import unicodedata
from datetime import datetime, timezone
from difflib import get_close_matches

import pandas as pd
import requests

from config.settings import current_season
from pipeline.fdcsv import _team_id  # isimden kararlı takim_id (aynı fonksiyon)

log = logging.getLogger(__name__)

TSDB_KEY = "123"  # resmi ücretsiz test anahtarı
TSDB_URL = "https://www.thesportsdb.com/api/v1/json/{key}/eventsnextleague.php?id={lid}"

# API-Football lig id -> TheSportsDB lig id
TSDB_LEAGUE = {203: 4339, 39: 4328, 140: 4335, 135: 4332, 78: 4331, 61: 4334}

# Normalizasyonun çözemediği bilinen isim farkları: normalize(tsdb_adı) -> normalize(db_adı)
ALIAS = {
    "manutd": "manunited", "manchesterutd": "manunited",
    "manchestercity": "mancity", "newcastleutd": "newcastle",
    "wolverhamptonwanderers": "wolves", "wolverhampton": "wolves",
    "tottenhamhotspur": "tottenham", "nottinghamforest": "nottmforest",
    "brightonhovealbion": "brighton", "westhamunited": "westham",
    "leedsunited": "leeds", "sheffieldunited": "sheffieldutd",
    "atleticomadrid": "athmadrid", "athleticclub": "athbilbao",
    "athleticbilbao": "athbilbao", "realsociedad": "sociedad",
    "realbetis": "betis", "celtavigo": "celta", "deportivoalaves": "alaves",
    "rayovallecano": "vallecano", "realvalladolid": "valladolid",
    "intermilan": "inter", "internazionale": "inter", "acmilan": "milan",
    "hellasverona": "verona", "borussiadortmund": "dortmund",
    "borussiamgladbach": "mgladbach", "bayernmunich": "bayernmunich",
    "bayerleverkusen": "leverkusen", "eintrachtfrankfurt": "eintfrankfurt",
    "parissaintgermain": "parissg", "psg": "parissg",
    "olympiquemarseille": "marseille", "olympiquelyon": "lyon",
    "asmonaco": "monaco", "staderennais": "rennes", "ognicemontpellier": "nice",
    "ogcnice": "nice",
}


# Takım adlarında anlam taşımayan, iki kaynakta farklı yazılabilen ekler.
_STOP = {"fc", "afc", "sc", "sk", "cf", "as", "ac", "ca", "calcio", "club",
         "kulubu", "spor", "futbol", "u21", "u23", "1", "cd", "ud", "rc", "sd"}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.casefold().replace("-", " ").replace(".", " ")
    tokens = [t for t in s.split() if t and t not in _STOP]
    s = "".join("".join(ch for ch in t if ch.isalnum()) for t in tokens)
    return ALIAS.get(s, s)


class LiveFixtureSource:
    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": "futbol-tahmin/1.0"})

    def _fetch_league(self, tsdb_lid: int) -> list[dict]:
        """Paylaşımlı ücretsiz anahtar sık 429 verir → artan beklemeyle yeniden dene."""
        url = TSDB_URL.format(key=TSDB_KEY, lid=tsdb_lid)
        for attempt in range(4):
            try:
                r = self._s.get(url, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                return (r.json() or {}).get("events") or []
            except (requests.RequestException, ValueError) as exc:
                log.warning("TheSportsDB lig %s (deneme %d) hata: %s", tsdb_lid, attempt + 1, exc)
                time.sleep(2 * (attempt + 1))
        log.warning("TheSportsDB lig %s çekilemedi (429/hata)", tsdb_lid)
        return []

    def upcoming_matches(self, repo, league_ids, days_ahead: int = 7) -> pd.DataFrame:
        """Önümüzdeki `days_ahead` gün içindeki maçları, DB takım id'lerine eşleyerek döndürür."""
        now = datetime.now(timezone.utc)
        season = current_season()
        rows: list[dict] = []
        unmatched: list[str] = []

        # DB'deki güncel-sezon takım isimleri (lig bazında) → normalize eşleme tablosu
        with repo.engine.connect() as c:
            team_df = pd.read_sql(
                "select distinct l.lig_id, t.takim_id, t.ad "
                "from takimlar t "
                "join maclar m on (m.ev_sahibi_id=t.takim_id or m.deplasman_id=t.takim_id) "
                "join ligler l on l.lig_id=m.lig_id", c)
        norm_by_league: dict[int, dict[str, int]] = {}
        for _, t in team_df.iterrows():
            norm_by_league.setdefault(int(t["lig_id"]), {})[_norm(t["ad"])] = int(t["takim_id"])

        def resolve(name: str, lid: int) -> int | None:
            table = norm_by_league.get(lid, {})
            n = _norm(name)
            if n in table:
                return table[n]
            close = get_close_matches(n, list(table), n=1, cutoff=0.86)
            return table[close[0]] if close else None

        for i, lid in enumerate(league_ids):
            tsdb = TSDB_LEAGUE.get(lid)
            if not tsdb:
                continue
            if i:
                time.sleep(1.5)  # paylaşımlı anahtarı yormamak için istekler arası bekleme
            for ev in self._fetch_league(tsdb):
                ts = ev.get("strTimestamp")
                if not ts:
                    continue
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if not (now <= dt <= now + pd.Timedelta(days=days_ahead)):
                    continue
                home, away = ev.get("strHomeTeam"), ev.get("strAwayTeam")
                hid, aid = resolve(home, lid), resolve(away, lid)
                if hid is None:
                    unmatched.append(f"{lid}:{home}")
                if aid is None:
                    unmatched.append(f"{lid}:{away}")
                if hid is None or aid is None or hid == aid:
                    continue
                rows.append({
                    "mac_id": _team_id(f"{ev.get('idEvent')}") | (1 << 40),  # TSDB olayından kararlı id
                    "lig_id": lid, "sezon": season, "hafta": ev.get("intRound"),
                    "tarih": dt, "hakem": None, "stadyum": ev.get("strVenue"),
                    "durum": "NS", "ev_sahibi_id": hid, "deplasman_id": aid,
                    "ev_gol": None, "dep_gol": None,
                    "ilk_yari_ev_gol": None, "ilk_yari_dep_gol": None,
                })
        if unmatched:
            log.warning("Eşleşmeyen takımlar (atlandı): %s", sorted(set(unmatched)))
        return pd.DataFrame(rows)
