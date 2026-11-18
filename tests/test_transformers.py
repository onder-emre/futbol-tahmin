import math
from datetime import datetime

import pandas as pd
import pytest

from api.transformers import FixtureTransformer, parse_stat_value
from db.repository import FootballRepository, dataframe_to_records
from pipeline.ingestion import select_changed_matches


def _fixture(fid, home, away, gh, ga, status="FT", date="2024-08-16T19:00:00+00:00", stats=None):
    fx = {
        "fixture": {"id": fid, "referee": "M. Oliver", "date": date,
                    "venue": {"id": 1, "name": "Old Trafford"},
                    "status": {"long": "Match Finished", "short": status, "elapsed": 90}},
        "league": {"id": 39, "name": "Premier League", "country": "England",
                   "logo": "l.png", "season": 2024, "round": "Regular Season - 1"},
        "teams": {"home": {"id": home, "name": f"T{home}", "logo": "h.png", "winner": True},
                  "away": {"id": away, "name": f"T{away}", "logo": "a.png", "winner": False}},
        "goals": {"home": gh, "away": ga},
        "score": {"halftime": {"home": 0 if gh is not None else None, "away": 0 if ga is not None else None},
                  "fulltime": {"home": gh, "away": ga}},
    }
    if stats is not None:
        fx["statistics"] = stats
    return fx


STATS = [
    {"team": {"id": 33}, "statistics": [
        {"type": "Ball Possession", "value": "61%"}, {"type": "Total Shots", "value": 14},
        {"type": "Corner Kicks", "value": 7}, {"type": "Yellow Cards", "value": None},
        {"type": "expected_goals", "value": "2.43"}, {"type": "Passes %", "value": "87%"},
        {"type": "Bilinmeyen", "value": 3},
    ]},
    {"team": {"id": 36}, "statistics": [
        {"type": "Ball Possession", "value": "39%"}, {"type": "Total Shots", "value": 10},
        {"type": "Red Cards", "value": 1}, {"type": "expected_goals", "value": "0.45"},
    ]},
]


@pytest.mark.parametrize("raw, expected", [("55%", 55.0), ("1.23", 1.23), (7, 7.0), (None, math.nan), ("-", math.nan)])
def test_parse_stat_value(raw, expected):
    result = parse_stat_value(raw)
    assert (math.isnan(result) and math.isnan(expected)) or result == expected


def test_transform_without_statistics():
    frames = FixtureTransformer().transform([_fixture(1, 33, 36, 1, 0), _fixture(2, 40, 33, None, None, "NS")])
    assert list(frames.leagues["lig_id"]) == [39]
    assert set(frames.teams["takim_id"]) == {33, 36, 40}
    assert (frames.teams["ulke"] == "England").all()
    m = frames.matches.set_index("mac_id")
    assert m.loc[1, "ev_gol"] == 1 and m.loc[1, "stadyum"] == "Old Trafford"
    assert pd.isna(m.loc[2, "ev_gol"])
    assert str(m["tarih"].dt.tz) == "UTC"
    assert frames.statistics.empty


def test_transform_flattens_statistics_wide():
    frames = FixtureTransformer().transform([_fixture(1, 33, 36, 2, 0, stats=STATS)])
    s = frames.statistics.set_index("takim_id")
    assert s.loc[33, "topla_oynama"] == 61.0
    assert s.loc[33, "xg"] == 2.43
    assert s.loc[33, "korner"] == 7
    assert pd.isna(s.loc[33, "sari_kart"])
    assert s.loc[36, "kirmizi_kart"] == 1
    assert "Bilinmeyen" not in frames.statistics.columns


def test_records_are_psycopg2_friendly():
    frames = FixtureTransformer().transform([_fixture(1, 33, 36, 2, 0, stats=STATS), _fixture(2, 40, 33, None, None, "NS")])
    for df in (frames.matches, frames.statistics, frames.teams):
        for rec in dataframe_to_records(df):
            for v in rec.values():
                assert v is None or isinstance(v, (int, float, str, bool, datetime)), (v, type(v))
                assert not type(v).__module__.startswith("numpy"), (v, type(v))


def test_select_changed_matches():
    api = FixtureTransformer().transform([
        _fixture(1, 33, 36, 1, 0),                    # aynı → yazılmamalı
        _fixture(2, 40, 33, 2, 2),                    # NS → FT oldu → yazılmalı
        _fixture(3, 50, 51, None, None, "NS"),        # DB'de yok → yazılmalı
    ]).matches
    db = pd.DataFrame({
        "mac_id": [1, 2], "durum": ["FT", "NS"],
        "ev_gol": pd.array([1, None], dtype="Int64"), "dep_gol": pd.array([0, None], dtype="Int64"),
        "tarih": pd.to_datetime(["2024-08-16T19:00:00+00:00"] * 2, utc=True),
    })
    assert sorted(select_changed_matches(api, db)["mac_id"]) == [2, 3]


def test_all_final():
    assert FootballRepository.all_final(["FT", "PEN", "CANC"])
    assert not FootballRepository.all_final(["FT", "NS"])
    assert not FootballRepository.all_final([])
