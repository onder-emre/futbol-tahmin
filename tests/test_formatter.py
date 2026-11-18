import re

import pandas as pd
import pytest

from bot.formatter import MessageFormatter, confidence_stars

DAY = pd.Timestamp("2026-11-18", tz="Europe/Istanbul")
SPECIAL = set(r"_*[]()~`>#+-=|{}.!")


def preds(rows):
    base = dict(lig="Süper Lig", tarih=pd.Timestamp("2026-11-18 17:00", tz="UTC"))
    out = []
    for i, (home, away, p1, px, p2) in enumerate(rows):
        probs = {"1": p1, "X": px, "2": p2}
        best = max(probs, key=probs.get)
        out.append({**base, "mac_id": i, "ev_sahibi": home, "deplasman": away,
                    "olasilik_1": p1, "olasilik_x": px, "olasilik_2": p2, "tahmin": best, "guven": probs[best]})
    return pd.DataFrame(out)


def assert_valid_markdown_v2(text: str):
    """Kaçırılmamış özel karakter sadece biçim işaretleri (* ve _) olabilir, ve bunlar çift sayıda olmalı."""
    unescaped = re.sub(r"\\.", "", text)
    bad = {c for c in unescaped if c in SPECIAL} - {"*", "_"}
    assert not bad, f"Kaçırılmamış karakterler: {bad}"
    assert unescaped.count("*") % 2 == 0 and unescaped.count("_") % 2 == 0


def test_spec_example_format():
    msg = MessageFormatter().build(preds([("Galatasaray", "Fenerbahçe", 0.65, 0.20, 0.15)]), DAY, "xgboost_v1")[0]
    assert "🟢 %65 \\| ⚪ %20 \\| 🔴 %15" in msg
    assert "*Galatasaray kazanır*" in msg
    assert "🇹🇷 *Süper Lig*" in msg
    assert "20:00" in msg            # 17:00 UTC → 20:00 İstanbul
    assert "Çarşamba" in msg
    assert_valid_markdown_v2(msg)


def test_only_confident_predictions_are_listed():
    df = preds([("Galatasaray", "Fenerbahçe", 0.65, 0.20, 0.15),
                ("Beşiktaş", "Trabzonspor", 0.40, 0.30, 0.30),      # %40 → listelenmez
                ("Göztepe", "Samsunspor", 0.20, 0.25, 0.55)])       # tam eşik → listelenir
    msg = "\n".join(MessageFormatter(threshold=0.55).build(df, DAY, "m"))
    assert "Galatasaray" in msg and "Samsunspor kazanır" in msg
    assert "Beşiktaş" not in msg
    assert "3 maçtan 2 güvenilir tahmin" in msg


def test_no_confident_message():
    msg = MessageFormatter().build(preds([("A", "B", 0.40, 0.30, 0.30)]), DAY, "m")[0]
    assert "güvenilir tahmin yok" in msg
    assert_valid_markdown_v2(msg)


def test_special_characters_in_team_names_are_escaped():
    df = preds([("St. Pauli", "1. FC Köln (U-21)", 0.70, 0.20, 0.10)])
    df["lig"] = "Bundesliga"
    msg = MessageFormatter().build(df, DAY, "random_forest_2026.11")[0]
    assert "St\\. Pauli" in msg and "\\(U\\-21\\)" in msg
    assert "🇩🇪" in msg
    assert_valid_markdown_v2(msg)


def test_long_lists_are_split_under_telegram_limit():
    df = preds([(f"Takım {i}", f"Rakip {i}", 0.80, 0.10, 0.10) for i in range(120)])
    messages = MessageFormatter().build(df, DAY, "m")
    assert len(messages) > 1
    assert all(len(m) <= 4096 for m in messages)
    assert sum(m.count("🎯") for m in messages) == 120
    for m in messages:
        assert_valid_markdown_v2(m)


@pytest.mark.parametrize("p, stars", [(0.56, "⭐"), (0.65, "⭐⭐"), (0.75, "⭐⭐⭐")])
def test_confidence_stars(p, stars):
    assert confidence_stars(p) == stars
