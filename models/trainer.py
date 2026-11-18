"""FAZ 3 — Maç sonucu sınıflandırma modeli (XGBoost / Random Forest).

Sınıflar: 0 → "1" (ev sahibi kazanır), 1 → "X" (beraberlik), 2 → "2" (deplasman kazanır)

Zaman bazlı değerlendirme (rastgele train_test_split YOK):
  • Holdout: en yeni %test_fraction maç TEST, öncesi EĞİTİM. Aynı gün maçlar bölünmez.
  • Eğitim içinde TimeSeriesSplit ile çapraz doğrulama → algoritma seçimi.
  • XGBoost erken durdurma da eğitim setinin son diliminde (yine zamana göre) yapılır;
    test seti hiçbir model kararında kullanılmaz.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from models.features import FeatureSet

logger = logging.getLogger(__name__)

RESULT_TO_CLASS = {"H": 0, "D": 1, "A": 2}
CLASS_LABELS = ("1", "X", "2")
ALGORITHMS = ("xgboost", "random_forest")


@dataclass(frozen=True)
class TrainConfig:
    algorithm: str = "auto"              # "xgboost" | "random_forest" | "auto" (CV ile seç)
    test_fraction: float = 0.15
    n_splits: int = 5
    min_history: int = 3                 # Her iki takımın da en az bu kadar geçmiş maçı olmalı
    early_stopping_fraction: float = 0.15
    random_state: int = 42
    xgb_params: dict = field(default_factory=lambda: dict(
        n_estimators=1500, learning_rate=0.03, max_depth=4, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    ))
    rf_params: dict = field(default_factory=lambda: dict(
        n_estimators=500, min_samples_leaf=20, max_features="sqrt",
    ))


@dataclass
class TrainingResult:
    model: object
    algorithm: str
    feature_columns: list[str]
    metrics: dict
    cv_scores: dict
    feature_importance: pd.Series
    train_period: tuple[str, str]
    test_period: tuple[str, str]
    # Ek pazarlar (maç sonucuna ek): KG = karşılıklı gol, ÜA = üst/alt 2.5 gol
    kg_model: object = None
    ua_model: object = None
    extra_metrics: dict = field(default_factory=dict)
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))

    def report(self) -> str:
        m = self.metrics
        lines = [
            f"Algoritma          : {self.algorithm}",
            f"Eğitim dönemi      : {self.train_period[0]} → {self.train_period[1]}  ({m['n_train']} maç)",
            f"Test dönemi        : {self.test_period[0]} → {self.test_period[1]}  ({m['n_test']} maç)",
            "",
            f"{'Metrik':<20}{'Model':>10}{'Baz (sınıf oranı)':>20}",
            f"{'Log loss ↓':<20}{m['log_loss']:>10.4f}{m['baseline_log_loss']:>20.4f}",
            f"{'RPS ↓':<20}{m['rps']:>10.4f}{m['baseline_rps']:>20.4f}",
            f"{'Doğruluk ↑':<20}{m['accuracy']:>10.3f}{m['baseline_accuracy']:>20.3f}  (baz: hep '1')",
            "",
            "CV (TimeSeriesSplit) ortalama log loss: "
            + ", ".join(f"{k}={v:.4f}" for k, v in self.cv_scores.items()),
            "",
            "Karışıklık matrisi (satır=gerçek, sütun=tahmin; 1/X/2):",
            str(pd.DataFrame(m["confusion_matrix"], index=CLASS_LABELS, columns=CLASS_LABELS)),
            "",
            "En önemli 10 öznitelik:",
            self.feature_importance.head(10).round(4).to_string(),
        ]
        if self.extra_metrics:
            lines += ["", "Ek pazarlar (holdout doğruluğu):"]
            if "kg_accuracy" in self.extra_metrics:
                lines.append(f"  KG (karşılıklı gol var/yok) : {self.extra_metrics['kg_accuracy']:.3f}")
            if "ua_accuracy" in self.extra_metrics:
                lines.append(f"  Üst/Alt 2.5 gol             : {self.extra_metrics['ua_accuracy']:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------- metrics
def ranked_probability_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Futbol tahmininde standart metrik: 1/X/2 sıralı olduğu için '2' yerine 'X' demek daha az hatalıdır."""
    onehot = np.eye(proba.shape[1])[y_true]
    cum_diff = np.cumsum(proba, axis=1) - np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum(cum_diff[:, :-1] ** 2, axis=1) / (proba.shape[1] - 1)))


def evaluate(y_true: np.ndarray, proba: np.ndarray, y_train: np.ndarray) -> dict:
    prior = np.bincount(y_train, minlength=3) / len(y_train)
    prior_proba = np.tile(prior, (len(y_true), 1))
    return {
        "n_test": int(len(y_true)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "rps": ranked_probability_score(y_true, proba),
        "accuracy": float(accuracy_score(y_true, proba.argmax(axis=1))),
        "baseline_log_loss": float(log_loss(y_true, prior_proba, labels=[0, 1, 2])),
        "baseline_rps": ranked_probability_score(y_true, prior_proba),
        "baseline_accuracy": float(np.mean(y_true == 0)),
        "confusion_matrix": confusion_matrix(y_true, proba.argmax(axis=1), labels=[0, 1, 2]).tolist(),
    }


# ---------------------------------------------------------------- trainer
class ModelTrainer:
    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()

    # ------------------------------------------------------------ data prep
    def _filtered(self, fs: FeatureSet) -> pd.DataFrame:
        """Yeterli geçmişi olan eğitim maçları, zaman sırasıyla (prepare ve ek pazarlar aynı satırları kullanır)."""
        df = fs.train.sort_values(["tarih", "mac_id"])
        enough = (df["ev_genel_mac_sayisi"] >= self.config.min_history) & \
                 (df["dep_genel_mac_sayisi"] >= self.config.min_history)
        return df[enough].reset_index(drop=True)

    def prepare(self, fs: FeatureSet) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
        df = self._filtered(fs)
        X = df[fs.feature_columns].astype("float64").reset_index(drop=True)
        y = df["sonuc"].map(RESULT_TO_CLASS).to_numpy(dtype=int)
        return X, y, df["tarih"].reset_index(drop=True)

    def _fit_binary(self, X: pd.DataFrame, y: np.ndarray):
        """KG / Üst-Alt için ikili Random Forest (eksik değerleri doldurarak)."""
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(**self.config.rf_params, n_jobs=-1, random_state=self.config.random_state),
        )
        return model.fit(X, y)

    def time_split(self, dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """En yeni maçlar test. Kesim noktası bir gün başına çekilir → aynı gün iki tarafa bölünmez."""
        cut_idx = int(len(dates) * (1 - self.config.test_fraction))
        cutoff = dates.iloc[cut_idx].normalize()
        train_mask = (dates < cutoff).to_numpy()
        return np.where(train_mask)[0], np.where(~train_mask)[0]

    # ---------------------------------------------------------- estimators
    def _fit(self, algorithm: str, X: pd.DataFrame, y: np.ndarray):
        if algorithm == "xgboost":
            return self._fit_xgb(X, y)
        if algorithm == "random_forest":
            model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                RandomForestClassifier(**self.config.rf_params, n_jobs=-1, random_state=self.config.random_state),
            )
            return model.fit(X, y)
        raise ValueError(f"Bilinmeyen algoritma: {algorithm}")

    def _fit_xgb(self, X: pd.DataFrame, y: np.ndarray) -> XGBClassifier:
        """Eğitim setinin son diliminde (zamana göre) erken durdurma → en iyi ağaç sayısıyla tüm veride yeniden eğit."""
        base = dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss", tree_method="hist",
                    random_state=self.config.random_state, n_jobs=-1, **self.config.xgb_params)
        split = int(len(X) * (1 - self.config.early_stopping_fraction))
        probe = XGBClassifier(**base, early_stopping_rounds=50)
        probe.fit(X.iloc[:split], y[:split], eval_set=[(X.iloc[split:], y[split:])], verbose=False)
        final = XGBClassifier(**{**base, "n_estimators": probe.best_iteration + 1})
        return final.fit(X, y, verbose=False)

    # ------------------------------------------------------------ workflow
    def cross_validate(self, algorithm: str, X: pd.DataFrame, y: np.ndarray) -> float:
        scores = []
        for tr, va in TimeSeriesSplit(n_splits=self.config.n_splits).split(X):
            model = self._fit(algorithm, X.iloc[tr], y[tr])
            scores.append(log_loss(y[va], model.predict_proba(X.iloc[va]), labels=[0, 1, 2]))
        return float(np.mean(scores))

    def train(self, fs: FeatureSet) -> TrainingResult:
        X, y, dates = self.prepare(fs)
        if len(X) < 200:
            raise ValueError(f"Eğitim için yetersiz veri: {len(X)} maç (en az 200 gerekir).")
        tr, te = self.time_split(dates)
        X_tr, y_tr = X.iloc[tr], y[tr]
        logger.info("Eğitim: %d maç, test: %d maç", len(tr), len(te))

        candidates = ALGORITHMS if self.config.algorithm == "auto" else (self.config.algorithm,)
        cv_scores = {alg: self.cross_validate(alg, X_tr, y_tr) for alg in candidates}
        algorithm = min(cv_scores, key=cv_scores.get)
        logger.info("CV log loss: %s → seçilen: %s", cv_scores, algorithm)

        # 1) Test performansını ölç (model sadece eğitim döneminden öğrenir)
        model = self._fit(algorithm, X_tr, y_tr)
        metrics = evaluate(y[te], model.predict_proba(X.iloc[te]), y_tr)
        metrics["n_train"] = int(len(tr))

        # 2) Canlı tahmin için son modeli TÜM geçmiş veriyle eğit (en güncel form bilgisi dahil)
        final_model = self._fit(algorithm, X, y)

        # 3) Ek pazarlar: KG (karşılıklı gol) ve Üst/Alt 2.5 gol — aynı öznitelikler, ikili modeller
        df = self._filtered(fs)
        goller = df["ev_gol"].to_numpy() + df["dep_gol"].to_numpy()
        kg = ((df["ev_gol"] > 0) & (df["dep_gol"] > 0)).astype(int).to_numpy()
        ua = (goller > 2.5).astype(int)
        extra: dict = {}
        kg_model = ua_model = None
        try:
            # holdout doğruluğu (sadece raporlama; model eğitim döneminden öğrenir)
            kg_probe = self._fit_binary(X_tr, kg[tr])
            extra["kg_accuracy"] = float(accuracy_score(kg[te], kg_probe.predict(X.iloc[te])))
            ua_probe = self._fit_binary(X_tr, ua[tr])
            extra["ua_accuracy"] = float(accuracy_score(ua[te], ua_probe.predict(X.iloc[te])))
            # canlı için tüm veriyle yeniden eğit
            kg_model = self._fit_binary(X, kg)
            ua_model = self._fit_binary(X, ua)
        except Exception:  # ek pazarlar başarısız olsa bile ana model (MS) teslim edilir
            logger.exception("Ek pazar (KG/ÜA) modelleri eğitilemedi, yalnızca MS döner")

        return TrainingResult(
            model=final_model,
            algorithm=algorithm,
            feature_columns=list(X.columns),
            metrics=metrics,
            cv_scores=cv_scores,
            feature_importance=self._importance(final_model, list(X.columns)),
            train_period=(str(dates.iloc[tr[0]].date()), str(dates.iloc[tr[-1]].date())),
            test_period=(str(dates.iloc[te[0]].date()), str(dates.iloc[te[-1]].date())),
            kg_model=kg_model, ua_model=ua_model, extra_metrics=extra,
        )

    @staticmethod
    def _importance(model, columns: list[str]) -> pd.Series:
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        return pd.Series(est.feature_importances_, index=columns).sort_values(ascending=False)
