import numpy as np
import pandas as pd
import pytest

from models.features import FeatureBuilder, FeatureConfig

RNG = np.random.default_rng(42)


def make_matches(n_rounds: int = 6, teams=(1, 2, 3, 4), upcoming: int = 2) -> pd.DataFrame:
    """Çift devreli lig: her tur her ikili bir kez, ev/deplasman dönüşümlü."""
    rows, mid = [], 100
    day = pd.Timestamp("2024-08-01", tz="UTC")
    pairs = [(a, b) for i, a in enumerate(teams) for b in teams[i + 1:]]
    for r in range(n_rounds):
        for j, (a, b) in enumerate(pairs):
            home, away = (a, b) if r % 2 == 0 else (b, a)
            gh, ga = RNG.integers(0, 4, size=2)
            rows.append(dict(
                mac_id=mid, lig_id=39, sezon=2024, tarih=day + pd.Timedelta(days=7 * r, hours=j), durum="FT",
                ev_sahibi_id=home, deplasman_id=away, ev_gol=float(gh), dep_gol=float(ga),
                ev_xg=float(RNG.uniform(0, 3)), dep_xg=float(RNG.uniform(0, 3)),
                ev_topla_oynama=float(RNG.uniform(30, 70)), dep_topla_oynama=np.nan,
                ev_sut=float(RNG.integers(5, 20)), dep_sut=float(RNG.integers(5, 20)),
                ev_isabetli_sut=3.0, dep_isabetli_sut=2.0,
            ))
            mid += 1
    last = rows[-1]["tarih"]
    for k in range(upcoming):
        a, b = pairs[k]
        rows.append(dict(mac_id=mid, lig_id=39, sezon=2024, tarih=last + pd.Timedelta(days=7, hours=k), durum="NS",
                         ev_sahibi_id=a, deplasman_id=b, ev_gol=np.nan, dep_gol=np.nan))
        mid += 1
    df = pd.DataFrame(rows)
    df["dep_topla_oynama"] = 100 - df["ev_topla_oynama"]
    return df


def team_history(df: pd.DataFrame, team: int, before: pd.Timestamp, venue: str | None = None) -> pd.DataFrame:
    """Bir takımın, verilen tarihten önce oynadığı maçlar (kendi bakış açısıyla)."""
    p = df[(df["durum"] == "FT") & (df["tarih"] < before)]
    home = p[p["ev_sahibi_id"] == team].assign(atilan=lambda x: x["ev_gol"], xg=lambda x: x["ev_xg"])
    away = p[p["deplasman_id"] == team].assign(atilan=lambda x: x["dep_gol"], xg=lambda x: x["dep_xg"])
    parts = {"ev": [home], "dep": [away], None: [home, away]}[venue]
    return pd.concat(parts).sort_values("tarih")


@pytest.fixture(scope="module")
def data():
    raw = make_matches()
    return raw, FeatureBuilder(FeatureConfig(window=5)).build(raw)


def test_general_form_equals_mean_of_previous_five(data):
    raw, fs = data
    for _, row in fs.train.iloc[20:].iterrows():
        hist = team_history(raw, row["ev_sahibi_id"], row["tarih"]).tail(5)
        assert row["ev_genel_atilan_gol_ort"] == pytest.approx(hist["atilan"].mean())
        assert row["ev_genel_xg_ort"] == pytest.approx(hist["xg"].mean())
        assert row["ev_genel_mac_sayisi"] == len(hist)


def test_venue_form_uses_only_home_or_away_games(data):
    raw, fs = data
    for _, row in fs.train.iloc[20:].iterrows():
        h = team_history(raw, row["ev_sahibi_id"], row["tarih"], venue="ev").tail(5)
        a = team_history(raw, row["deplasman_id"], row["tarih"], venue="dep").tail(5)
        assert row["ev_saha_atilan_gol_ort"] == pytest.approx(h["atilan"].mean()) if len(h) else np.isnan(row["ev_saha_atilan_gol_ort"])
        assert row["dep_saha_atilan_gol_ort"] == pytest.approx(a["atilan"].mean()) if len(a) else np.isnan(row["dep_saha_atilan_gol_ort"])


def test_no_leakage_own_match_stats_do_not_affect_own_features(data):
    raw, fs = data
    target_id = raw.loc[30, "mac_id"]
    tampered = raw.copy()
    tampered.loc[30, ["ev_gol", "dep_gol", "ev_xg", "dep_xg", "ev_topla_oynama", "ev_sut"]] = [99, 0, 50, 0, 99, 99]
    fs2 = FeatureBuilder(FeatureConfig(window=5)).build(tampered)

    cols = fs.feature_columns
    before = fs.frame.set_index("mac_id")
    after = fs2.frame.set_index("mac_id")
    # Maçın kendi öznitelikleri ve öncesindeki tüm maçlar DEĞİŞMEMELİ
    earlier_or_same = before.index[before["tarih"] <= before.loc[target_id, "tarih"]]
    pd.testing.assert_frame_equal(before.loc[earlier_or_same, cols], after.loc[earlier_or_same, cols])
    # Sonraki maçlar değişmeli (yani bilgi ileriye doğru akıyor)
    later = before.index[before["tarih"] > before.loc[target_id, "tarih"]]
    assert not before.loc[later, cols].equals(after.loc[later, cols])


def test_first_match_has_no_history(data):
    _, fs = data
    first = fs.frame.iloc[0]
    assert first["ev_genel_mac_sayisi"] == 0
    assert np.isnan(first["ev_genel_atilan_gol_ort"])
    assert first["h2h_mac_sayisi"] == 0


def test_upcoming_match_uses_all_played_history(data):
    raw, fs = data
    up = fs.upcoming
    assert len(up) == 2 and up["sonuc"].isna().all()
    for _, row in up.iterrows():
        hist = team_history(raw, row["ev_sahibi_id"], row["tarih"]).tail(5)
        assert row["ev_genel_atilan_gol_ort"] == pytest.approx(hist["atilan"].mean())
        assert row["ev_genel_mac_sayisi"] == 5


def test_h2h_from_home_team_perspective(data):
    raw, fs = data
    row = fs.frame.iloc[-1]  # yaklaşan maç
    home, away = row["ev_sahibi_id"], row["deplasman_id"]
    p = raw[(raw["durum"] == "FT") & (raw["tarih"] < row["tarih"])]
    meet = p[((p.ev_sahibi_id == home) & (p.deplasman_id == away)) | ((p.ev_sahibi_id == away) & (p.deplasman_id == home))].tail(5)
    home_goals = np.where(meet.ev_sahibi_id == home, meet.ev_gol, meet.dep_gol)
    away_goals = np.where(meet.ev_sahibi_id == home, meet.dep_gol, meet.ev_gol)
    assert row["h2h_mac_sayisi"] == len(meet)
    assert row["h2h_ev_kazanma_orani"] == pytest.approx((home_goals > away_goals).mean())
    assert row["h2h_dep_kazanma_orani"] == pytest.approx((home_goals < away_goals).mean())
    assert row["h2h_ev_gol_farki_ort"] == pytest.approx((home_goals - away_goals).mean())


def test_possession_trend_is_slope(data):
    raw, fs = data
    row = fs.train.iloc[-1]
    hist = team_history(raw, row["ev_sahibi_id"], row["tarih"]).tail(5)
    poss = np.where(hist["ev_sahibi_id"] == row["ev_sahibi_id"], hist["ev_topla_oynama"], hist["dep_topla_oynama"])
    assert row["ev_genel_topla_oynama_trend"] == pytest.approx(np.polyfit(np.arange(5), poss, 1)[0])


def test_feature_columns_contain_no_raw_match_data(data):
    _, fs = data
    forbidden = {"ev_gol", "dep_gol", "sonuc", "ev_xg", "dep_xg", "ev_topla_oynama", "dep_topla_oynama", "ev_sut"}
    assert forbidden.isdisjoint(fs.feature_columns)
    assert all(c.startswith(("ev_", "dep_", "h2h_", "fark_")) for c in fs.feature_columns)
