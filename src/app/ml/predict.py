from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TicketModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        self.model = joblib.load(path)

    def predict(self, text: str) -> str:
        if self.model is None:
            self.load()

        clean_text = normalize_text(text)
        if not clean_text:
            raise ValueError("Le texte d'entrée est vide ou invalide.")

        return self.model.predict([clean_text])[0]

    def predict_many(self, texts: list[str]) -> list[str]:
        if self.model is None:
            self.load()

        clean_texts = [normalize_text(text) for text in texts]
        clean_texts = [text for text in clean_texts if text]

        if not clean_texts:
            raise ValueError("Aucun texte valide à prédire.")

        return list(self.model.predict(clean_texts))
    
    def predict_with_proba(self, text: str):
        if self.model is None:
            self.load()

        clean_text = normalize_text(text)
        probs = self.model.predict_proba([clean_text])[0]
        idx = probs.argmax()

        return self.model.classes_[idx], float(probs[idx])