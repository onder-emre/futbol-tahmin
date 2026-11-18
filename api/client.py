"""API-Football v3 istemcisi: hız sınırı, yeniden deneme ve günlük kota koruması."""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import requests

from config.settings import ApiSettings

logger = logging.getLogger(__name__)


class ApiFootballError(Exception):
    """API'nin döndürdüğü genel hata."""


class QuotaExhaustedError(ApiFootballError):
    """Günlük kota bitti (veya yedek sınıra ulaşıldı). Pipeline durmalı."""


class ApiFootballClient:
    MAX_IDS_PER_REQUEST = 20  # /fixtures?ids= en fazla 20 ID kabul eder
    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, settings: ApiSettings, session: requests.Session | None = None) -> None:
        self._settings = settings
        self._session = session or requests.Session()
        self._session.headers.update({"x-apisports-key": settings.key})
        self._min_interval = 60.0 / max(settings.requests_per_minute, 1)
        self._last_request_at = 0.0
        self.requests_made = 0
        self.daily_remaining: int | None = None
        self.request_budget: int | None = None  # Bu çalıştırma için opsiyonel üst sınır

    # ------------------------------------------------------------------ public
    def get_fixtures(self, league_id: int, season: int) -> list[dict[str, Any]]:
        """Bir lig-sezonun tüm maçları (istatistiksiz). Tek istek."""
        return self._get("/fixtures", {"league": league_id, "season": season})

    def get_fixtures_by_date(self, day: str, timezone: str = "Europe/Istanbul") -> list[dict[str, Any]]:
        """Bir günün TÜM liglerdeki fikstürü (tek istek). day: 'YYYY-MM-DD'."""
        return self._get("/fixtures", {"date": day, "timezone": timezone})

    def get_fixtures_by_ids(self, fixture_ids: Sequence[int]) -> list[dict[str, Any]]:
        """En fazla 20 maçı istatistikleriyle birlikte tek istekte döndürür.

        /fixtures/statistics maç başına 1 istek harcar; ids parametresi
        events/lineups/statistics/players dahil 20 maçı tek istekte verir → 20x kota tasarrufu.
        """
        if not fixture_ids:
            return []
        if len(fixture_ids) > self.MAX_IDS_PER_REQUEST:
            raise ValueError(f"En fazla {self.MAX_IDS_PER_REQUEST} ID gönderilebilir.")
        return self._get("/fixtures", {"ids": "-".join(str(i) for i in fixture_ids)})

    def get_status(self) -> dict[str, Any]:
        """Hesap/kota bilgisi (/status kotadan düşmez)."""
        resp = self._session.get(f"{self._settings.base_url}/status", timeout=self._settings.timeout_seconds)
        resp.raise_for_status()
        return resp.json().get("response", {})

    # ----------------------------------------------------------------- private
    def _get(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self._ensure_quota()
        url = f"{self._settings.base_url}{endpoint}"

        for attempt in range(1, self._settings.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self._settings.timeout_seconds)
            except requests.RequestException as exc:
                self._backoff(attempt, f"ağ hatası: {exc}")
                continue

            self.requests_made += 1
            self._read_quota_headers(resp)

            if resp.status_code in self._RETRYABLE_STATUS:
                self._backoff(attempt, f"HTTP {resp.status_code}")
                continue
            resp.raise_for_status()

            payload = resp.json()
            errors = payload.get("errors")
            if errors:  # API hataları 200 ile döner; dict veya list olabilir
                if isinstance(errors, dict) and "rateLimit" in errors:
                    self._backoff(attempt, "dakikalık limit")
                    continue
                if isinstance(errors, dict) and "requests" in errors:
                    raise QuotaExhaustedError(str(errors["requests"]))
                raise ApiFootballError(f"{endpoint} {params}: {errors}")

            logger.debug("%s %s → %d kayıt", endpoint, params, payload.get("results", 0))
            return payload.get("response", [])

        raise ApiFootballError(f"{endpoint} {params}: {self._settings.max_retries} denemede başarısız.")

    def _ensure_quota(self) -> None:
        if self.request_budget is not None and self.requests_made >= self.request_budget:
            raise QuotaExhaustedError(f"Bu çalıştırma için istek bütçesi ({self.request_budget}) doldu.")
        if self.daily_remaining is not None and self.daily_remaining <= self._settings.daily_quota_reserve:
            raise QuotaExhaustedError(
                f"Günlük kalan istek {self.daily_remaining}, yedek sınır {self._settings.daily_quota_reserve}."
            )

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _read_quota_headers(self, resp: requests.Response) -> None:
        remaining = resp.headers.get("x-ratelimit-requests-remaining")
        if remaining is not None and remaining.lstrip("-").isdigit():
            self.daily_remaining = int(remaining)

    @staticmethod
    def _backoff(attempt: int, reason: str) -> None:
        delay = min(2 ** attempt * 5, 60)
        logger.warning("İstek tekrar denenecek (%s), deneme %d, %ds bekleniyor.", reason, attempt, delay)
        time.sleep(delay)
