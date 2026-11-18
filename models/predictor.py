"""Model saklama ve olasılık bazlı tahmin (predict_proba)."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT
from models.trainer import CLASS_LABELS, TrainingResult

ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts"


@dataclass
class ModelArtifact:
    model: object
    algorithm: str
    feature_columns: list[str]
    metrics: dict
    version: str
    kg_model: object = None   # karşılıklı gol (var/yok) ikili modeli
    ua_model: object = None   # üst/alt 2.5 gol ikili modeli

    @classmethod
    def from_training(cls, result: TrainingResult) -> "ModelArtifact":
        return cls(result.model, result.algorithm, result.feature_columns, result.metrics,
                   version=f"{result.algorithm}_{result.trained_at}",
                   kg_model=getattr(result, "kg_model", None),
                   ua_model=getattr(result, "ua_model", None))


class ModelStore:
    def __init__(self, directory: Path = ARTIFACT_DIR) -> None:
        self.directory = directory

    def save(self, artifact: ModelArtifact) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"model_{artifact.version}.joblib"
        joblib.dump(artifact, path)
        shutil.copyfile(path, self.directory / "latest.joblib")
        return path

    def load(self, version: str = "latest") -> ModelArtifact:
        name = "latest.joblib" if version == "latest" else f"model_{version}.joblib"
        path = self.directory / name
        if not path.exists():
            raise FileNotFoundError(f"Model bulunamadı: {path}. Önce: python -m scripts.train_model")
        return joblib.load(path)


def to_percentages(probs: np.ndarray) -> list[int]:
    """Olasılıkları toplamı tam 100 olan tam sayılara çevirir (en büyük kalan yöntemi)."""
    raw = np.asarray(probs, dtype=float) * 100
    floors = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - floors))[: 100 - floors.sum()]:
        floors[i] += 1
    return floors.tolist()


def format_prediction(home: str, away: str, p_home: float, p_draw: float, p_away: float) -> str:
    """Örn: 'Galatasaray Kazanır %65, Beraberlik %20, Fenerbahçe Kazanır %15'"""
    h, d, a = to_percentages([p_home, p_draw, p_away])
    return f"{home} Kazanır %{h}, Beraberlik %{d}, {away} Kazanır %{a}"


class MatchPredictor:
    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """features: FeatureSet.frame satırları (mac_id + öznitelik kolonları).
        Çıktı: mac_id, olasilik_1, olasilik_x, olasilik_2, tahmin ('1'/'X'/'2'), guven."""
        if features.empty:
            return pd.DataFrame(columns=["mac_id", "olasilik_1", "olasilik_x", "olasilik_2", "tahmin", "guven"])
        X = features.reindex(columns=self.artifact.feature_columns).astype("float64")
        proba = self.artifact.model.predict_proba(X)
        out = pd.DataFrame({
            "mac_id": features["mac_id"].to_numpy(),
            "olasilik_1": proba[:, 0], "olasilik_x": proba[:, 1], "olasilik_2": proba[:, 2],
        })
        out["tahmin"] = np.array(CLASS_LABELS)[proba.argmax(axis=1)]
        out["guven"] = proba.max(axis=1)
        out["model_versiyonu"] = self.artifact.version

        # Ek pazarlar (varsa): KG = karşılıklı gol, ÜA = üst/alt 2.5 gol
        kg_model = getattr(self.artifact, "kg_model", None)
        ua_model = getattr(self.artifact, "ua_model", None)
        if kg_model is not None:
            p_var = kg_model.predict_proba(X)[:, 1]
            out["kg_olasilik"] = p_var
            out["kg_tahmin"] = np.where(p_var >= 0.5, "Var", "Yok")
        if ua_model is not None:
            p_ust = ua_model.predict_proba(X)[:, 1]
            out["ua_olasilik"] = p_ust
            out["ua_tahmin"] = np.where(p_ust >= 0.5, "Üst", "Alt")
        return out
