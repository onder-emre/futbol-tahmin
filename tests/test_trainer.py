import numpy as np
import pandas as pd
import pytest

from models.features import FeatureBuilder, FeatureSet
from models.predictor import MatchPredictor, ModelArtifact, ModelStore, format_prediction, to_percentages
from models.trainer import ModelTrainer, TrainConfig, ranked_probability_score

FAST = dict(n_splits=3, xgb_params=dict(n_estimators=300, learning_rate=0.05, max_depth=3),
            rf_params=dict(n_estimators=100, min_samples_leaf=20))


def simulate_league(n_teams=16, seasons=3, upcoming=4, seed=7) -> pd.DataFrame:
    """Gizli takım güçleriyle (Poisson) gerçekçi lig simülasyonu → modelin öğrenebileceği sinyal var."""
    rng = np.random.default_rng(seed)
    attack, defence = rng.normal(0, 0.35, n_teams), rng.normal(0, 0.35, n_teams)
    teams = np.arange(1, n_teams + 1)
    rows, mid, day = [], 1, pd.Timestamp("2021-08-01", tz="UTC")
    for s in range(seasons):
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(fixtures)
        for k, (h, a) in enumerate(fixtures):
            lam_h = np.exp(0.25 + attack[h - 1] - defence[a - 1])
            lam_a = np.exp(attack[a - 1] - defence[h - 1])
            xg_h, xg_a = rng.gamma(4, lam_h / 4), rng.gamma(4, lam_a / 4)
            poss = float(np.clip(50 + 15 * (attack[h - 1] - attack[a - 1]) + rng.normal(0, 5), 25, 75))
            rows.append(dict(
                mac_id=mid, lig_id=203, sezon=2021 + s, tarih=day + pd.Timedelta(hours=8 * k + 3000 * s),
                durum="FT", ev_sahibi_id=h, deplasman_id=a,
                ev_gol=float(rng.poisson(lam_h)), dep_gol=float(rng.poisson(lam_a)),
                ev_xg=xg_h, dep_xg=xg_a, ev_topla_oynama=poss, dep_topla_oynama=100 - poss,
                ev_sut=float(rng.poisson(12 * lam_h)), dep_sut=float(rng.poisson(12 * lam_a)),
                ev_isabetli_sut=float(rng.poisson(4 * lam_h)), dep_isabetli_sut=float(rng.poisson(4 * lam_a)),
            ))
            mid += 1
    last = rows[-1]["tarih"]
    for k in range(upcoming):
        rows.append(dict(mac_id=mid, lig_id=203, sezon=2021 + seasons, tarih=last + pd.Timedelta(days=3, hours=k),
                         durum="NS", ev_sahibi_id=teams[k], deplasman_id=teams[-k - 1], ev_gol=np.nan, dep_gol=np.nan))
        mid += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def feature_set() -> FeatureSet:
    return FeatureBuilder().build(simulate_league())


@pytest.fixture(scope="module")
def trained(feature_set):
    return ModelTrainer(TrainConfig(algorithm="auto", **FAST)).train(feature_set)


def test_time_split_has_no_overlap(feature_set):
    trainer = ModelTrainer(TrainConfig(**FAST))
    _, _, dates = trainer.prepare(feature_set)
    tr, te = trainer.time_split(dates)
    assert dates.iloc[tr].max() < dates.iloc[te].min()
    assert set(dates.iloc[tr].dt.date).isdisjoint(set(dates.iloc[te].dt.date))  # aynı gün bölünmez
    assert 0.1 < len(te) / len(dates) < 0.2


def test_training_beats_baseline(trained):
    m = trained.metrics
    assert trained.algorithm in ("xgboost", "random_forest")
    assert set(trained.cv_scores) == {"xgboost", "random_forest"}
    assert m["log_loss"] < m["baseline_log_loss"]
    assert m["rps"] < m["baseline_rps"]
    assert "Log loss" in trained.report()


@pytest.mark.parametrize("algorithm", ["xgboost", "random_forest"])
def test_predict_proba_on_upcoming(feature_set, algorithm, tmp_path):
    result = ModelTrainer(TrainConfig(algorithm=algorithm, **FAST)).train(feature_set)
    store = ModelStore(tmp_path)
    store.save(ModelArtifact.from_training(result))
    artifact = store.load()  # disk'ten geri yükleme
    preds = MatchPredictor(artifact).predict(feature_set.upcoming)
    assert len(preds) == 4
    probs = preds[["olasilik_1", "olasilik_x", "olasilik_2"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0) and (probs > 0).all()
    assert set(preds["tahmin"]) <= {"1", "X", "2"}
    assert preds["model_versiyonu"].str.startswith(algorithm).all()


def test_format_prediction_matches_spec():
    assert format_prediction("Galatasaray", "Fenerbahçe", 0.65, 0.20, 0.15) == \
        "Galatasaray Kazanır %65, Beraberlik %20, Fenerbahçe Kazanır %15"


@pytest.mark.parametrize("probs", [[0.333, 0.333, 0.334], [0.455, 0.275, 0.27], [0.996, 0.002, 0.002]])
def test_percentages_always_sum_to_100(probs):
    assert sum(to_percentages(probs)) == 100


def test_rps_extremes():
    y = np.array([0, 2])
    assert ranked_probability_score(y, np.array([[1, 0, 0], [0, 0, 1.0]])) == 0
    assert ranked_probability_score(y, np.array([[0, 0, 1], [1, 0, 0.0]])) == 1
