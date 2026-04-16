from __future__ import annotations

import os
from functools import lru_cache
from dotenv import load_dotenv

# charge .env automatiquement
load_dotenv()


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "tri-demandes-api")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/model.joblib")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "logreg-v1")

    GCP_PROJECT_ID: str | None = os.getenv("GCP_PROJECT_ID")
    GCS_BUCKET_NAME: str | None = os.getenv("GCS_BUCKET_NAME")
    GOOGLE_APPLICATION_CREDENTIALS: str | None = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    def validate(self):
        # valide seulement si vraiment nécessaire
        pass

def get_settings():
    return Settings()

settings = get_settings()