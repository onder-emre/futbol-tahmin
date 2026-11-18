"""Tahminleri Telegram MarkdownV2 mesajına dönüştürür (ağ erişimi yok → kolay test).

Örnek çıktı:
    ⚽ Günün Güvenilir Tahminleri
    📅 18.11.2026 Çarşamba
    🇹🇷 Süper Lig
    🕘 20:00 │ Galatasaray - Fenerbahçe
    🟢 %65 | ⚪ %20 | 🔴 %15
    🎯 Galatasaray kazanır (MS 1) ⭐⭐
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from telegram.helpers import escape_markdown

from models.predictor import to_percentages

TELEGRAM_LIMIT = 4096
_SAFE_LIMIT = 3800

LEAGUE_FLAGS = {
    "Süper Lig": "🇹🇷", "Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "La Liga": "🇪🇸",
    "Serie A": "🇮🇹", "Bundesliga": "🇩🇪", "Ligue 1": "🇫🇷",
}
WEEKDAYS_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]


def esc(text: object) -> str:
    return escape_markdown(str(text), version=2)


def bold(text: object) -> str:
    return f"*{esc(text)}*"


def confidence_stars(p: float) -> str:
    return "⭐⭐⭐" if p >= 0.70 else "⭐⭐" if p >= 0.62 else "⭐"


@dataclass(frozen=True)
class MessageFormatter:
    threshold: float = 0.55
    timezone: str = "Europe/Istanbul"

    def select_confident(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Sadece en yüksek olasılığı eşik ve üzerinde olan 'güvenilir' tahminler."""
        return predictions[predictions["guven"] >= self.threshold]

    def build(self, predictions: pd.DataFrame, day: pd.Timestamp, model_version: str) -> list[str]:
        """predictions kolonları: mac_id, tarih, lig, ev_sahibi, deplasman,
        olasilik_1, olasilik_x, olasilik_2, tahmin, guven. Dönüş: 4096 sınırına göre bölünmüş mesajlar."""
        confident = self.select_confident(predictions).sort_values(["lig", "tarih"])
        header = self._header(day)
        footer = self._footer(total=len(predictions), shown=len(confident), model_version=model_version)

        if confident.empty:
            body = (f"😴 Bugün {esc(len(predictions))} maç incelendi, "
                    f"%{esc(round(self.threshold * 100))} eşiğini geçen güvenilir tahmin yok\\.")
            return [f"{header}\n\n{body}\n\n{footer}"]

        blocks: list[str] = []
        for lig, grp in confident.groupby("lig", sort=False):
            blocks.append(f"{LEAGUE_FLAGS.get(lig, '🏆')} {bold(lig)}")
            blocks.extend(self._match_block(r) for _, r in grp.iterrows())
        return self._paginate(header, blocks, footer)

    # ------------------------------------------------------------ parts
    def _header(self, day: pd.Timestamp) -> str:
        return (f"⚽ {bold('Günün Güvenilir Tahminleri')}\n"
                f"📅 {esc(day.strftime('%d.%m.%Y'))} {esc(WEEKDAYS_TR[day.weekday()])}\n"
                f"{esc('━' * 22)}")

    def _match_block(self, r: pd.Series) -> str:
        p1, px, p2 = to_percentages([r["olasilik_1"], r["olasilik_x"], r["olasilik_2"]])
        kickoff = pd.Timestamp(r["tarih"]).tz_convert(self.timezone).strftime("%H:%M")
        verdict = {
            "1": f"{r['ev_sahibi']} kazanır",
            "X": "Beraberlik",
            "2": f"{r['deplasman']} kazanır",
        }[r["tahmin"]]
        market = f"(MS {r['tahmin']})"
        return (
            f"🕘 {esc(kickoff)} │ {bold(r['ev_sahibi'])} {esc('-')} {bold(r['deplasman'])}\n"
            f"🟢 %{p1} \\| ⚪ %{px} \\| 🔴 %{p2}\n"
            f"🎯 {bold(verdict)} {esc(market)} {confidence_stars(r['guven'])}"
        )

    def _footer(self, total: int, shown: int, model_version: str) -> str:
        return (
            f"{esc('━' * 22)}\n"
            f"🟢 Ev sahibi  ⚪ Beraberlik  🔴 Deplasman\n"
            f"📊 {esc(f'{total} maçtan {shown} güvenilir tahmin (eşik %{round(self.threshold * 100)})')}\n"
            f"🤖 Model: {esc(model_version)}\n"
            f"_{esc('İstatistiksel model çıktısıdır, kesinlik içermez.')}_"
        )

    @staticmethod
    def _paginate(header: str, blocks: list[str], footer: str) -> list[str]:
        messages, current = [], header
        for block in blocks:
            if len(current) + len(block) + 2 > _SAFE_LIMIT:
                messages.append(current)
                current = block
            else:
                current += "\n\n" + block
        if len(current) + len(footer) + 2 > _SAFE_LIMIT:
            messages.append(current)
            current = footer
        else:
            current += "\n\n" + footer
        messages.append(current)
        return messages
