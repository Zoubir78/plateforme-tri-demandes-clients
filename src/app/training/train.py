from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.app.services.gcs_service import upload_file_to_gcs
from src.app.core.config import get_settings

settings = get_settings()

upload_file_to_gcs(
    bucket_name=settings.GCS_BUCKET_NAME,
    local_path="models/model.joblib",
    blob_name="models/model_v1.joblib",
)

def _to_single_label(value: Any) -> str | None:
    """
    Convertit :
    - "['AchatMagasin']" -> "AchatMagasin"
    - ['AchatMagasin'] -> "AchatMagasin"
    - "AchatMagasin" -> "AchatMagasin"
    - "" / None -> None
    """
    if value is None or pd.isna(value):
        return None

    # Cas liste/tuple/set Python
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if not items:
            return None
        return str(items[0]).strip() or None

    text = str(value).strip()
    if not text:
        return None

    # Cas chaîne de type "['AchatMagasin']"
    if text.startswith("[") or text.startswith("(") or text.startswith("{"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                parsed = list(parsed)
                if not parsed:
                    return None
                return str(parsed[0]).strip() or None
            return str(parsed).strip() or None
        except (ValueError, SyntaxError):
            # fallback si le format est cassé
            cleaned = text.strip("[](){}").split(",")[0].strip()
            return cleaned or None

    return text


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "category" not in df.columns:
        raise ValueError("La colonne 'category' est absente du fichier d'entraînement.")

    out = df.copy()
    out["category"] = out["category"].apply(_to_single_label)
    out = out.dropna(subset=["clean_text", "category"])

    # On garde seulement les textes non vides
    out["clean_text"] = out["clean_text"].astype(str).str.strip()
    out = out[out["clean_text"] != ""]

    return out.reset_index(drop=True)


def train_model(data_path: str, model_path: str) -> None:
    df = pd.read_csv(data_path)
    df = prepare_labels(df)

    if "clean_text" not in df.columns:
        raise ValueError("La colonne 'clean_text' est absente du fichier d'entraînement.")

    X = df["clean_text"]
    y = df["category"]

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X, y)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modèle entraîné et sauvegardé dans : {model_path}")


if __name__ == "__main__":
    train_model("data/processed/train.csv", "models/model.joblib")