from __future__ import annotations

from datetime import datetime

from src.app.ingestion.extract import extract_from_signalconso_api
from src.app.ingestion.transform import transform_dataframe
from src.app.services.gcs_service import upload_file_to_gcs
from src.app.training.train import train_model
from src.app.core.config import get_settings


API_URL = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/signalconso/records"


def run_pipeline(log) -> dict:
    """
    Pipeline complet exécutable depuis Streamlit
    log = fonction Streamlit st.write / placeholder
    """

    settings = get_settings()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    log("🚀 Démarrage pipeline")

    # ---------------- EXTRACT ----------------
    log("📥 Extraction des données")
    raw_df = extract_from_signalconso_api(API_URL, limit=10000)

    raw_path = "data/raw/signalconso.csv"
    raw_df.to_csv(raw_path, index=False)

    upload_file_to_gcs(
        settings.GCS_BUCKET_NAME,
        raw_path,
        f"raw/signalconso_{today}.csv",
    )

    log(f"✔ RAW upload OK ({len(raw_df)} lignes)")

    # ---------------- TRANSFORM ----------------
    log("🧹 Transformation des données")
    clean_df = transform_dataframe(raw_df)

    clean_path = "data/processed/signalconso_clean.csv"
    clean_df.to_csv(clean_path, index=False)

    upload_file_to_gcs(
        settings.GCS_BUCKET_NAME,
        clean_path,
        f"processed/signalconso_clean_{today}.csv",
    )

    log(f"✔ CLEAN upload OK ({len(clean_df)} lignes)")

    # ---------------- TRAIN ----------------
    log("🤖 Entraînement du modèle")
    model_path = "models/model.joblib"

    train_model(
        data_path=clean_path,
        model_path=model_path,
    )

    upload_file_to_gcs(
        settings.GCS_BUCKET_NAME,
        model_path,
        f"models/model_{today}.joblib",
    )

    log("✔ MODÈLE entraîné et uploadé")

    return {
        "raw_rows": len(raw_df),
        "clean_rows": len(clean_df),
        "model_path": model_path,
    }