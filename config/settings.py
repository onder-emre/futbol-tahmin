"""Uygulama ayarları. Tüm gizli bilgiler .env dosyasından okunur."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# API-Football lig ID'leri
LEAGUES: dict[int, str] = {
    203: "Süper Lig",
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}


def current_season(today: date | None = None) -> int:
    """Avrupa sezonları Temmuz'da başlar: Kasım 2026 → 2026 sezonu (2026/27)."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise RuntimeError(f"'{name}' ortam değişkeni tanımlı değil (.env dosyasını kontrol et).")
    return value


def _env_int_list(name: str, default: list[int]) -> tuple[int, ...]:
    raw = os.getenv(name)
    if not raw:
        return tuple(default)
    return tuple(int(x) for x in raw.replace(" ", "").split(",") if x)


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    name: str
    user: str
    password: str

    @property
    def url(self) -> URL:
        return URL.create(
            "postgresql+psycopg2",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.name,
        )


@dataclass(frozen=True)
class ApiSettings:
    key: str
    base_url: str = "https://v3.football.api-sports.io"
    requests_per_minute: int = 10       # Free plan: 10/dk
    daily_quota_reserve: int = 5        # Günlük kotada bu kadar istek her zaman yedekte kalır
    timeout_seconds: int = 30
    max_retries: int = 3


@dataclass(frozen=True)
class PipelineSettings:
    league_ids: tuple[int, ...]
    start_season: int
    end_season: int

    @property
    def seasons(self) -> list[int]:
        return list(range(self.start_season, self.end_season + 1))


@dataclass(frozen=True)
class Settings:
    db: DatabaseSettings
    api: ApiSettings
    pipeline: PipelineSettings

    @classmethod
    def from_env(cls) -> "Settings":
        season_now = current_season()
        return cls(
            db=DatabaseSettings(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                name=os.getenv("DB_NAME", "futbol_tahmin"),
                user=os.getenv("DB_USER", "postgres"),
                password=_env("DB_PASSWORD"),
            ),
            api=ApiSettings(
                key=_env("API_FOOTBALL_KEY"),
                requests_per_minute=int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "10")),
                daily_quota_reserve=int(os.getenv("API_DAILY_QUOTA_RESERVE", "5")),
            ),
            pipeline=PipelineSettings(
                league_ids=_env_int_list("LEAGUE_IDS", list(LEAGUES)),
                start_season=int(os.getenv("START_SEASON", str(season_now - 9))),
                end_season=int(os.getenv("END_SEASON", str(season_now))),
            ),
        )
