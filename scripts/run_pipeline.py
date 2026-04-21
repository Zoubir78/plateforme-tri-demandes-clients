from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import gettempdir

from src.app.core.config import get_settings
from src.app.ingestion.extract import extract_from_signalconso_api
from src.app.ingestion.transform import transform_dataframe
from src.app.services.gcs_service import upload_file_to_gcs

API_URL = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/signalconso/records"


def main() -> None:
    settings = get_settings()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Fichiers temporaires compatibles cloud
    tmp_dir = Path(gettempdir())
    raw_path = tmp_dir / "signalconso.csv"
    clean_path = tmp_dir / "signalconso_clean.csv"

    # --------- EXTRACT ---------
    raw_df = extract_from_signalconso_api(API_URL, limit=10000)
    raw_df.to_csv(raw_path, index=False)

    # --------- TRANSFORM ---------
    clean_df = transform_dataframe(raw_df)
    clean_df.to_csv(clean_path, index=False)

    # --------- UPLOAD RAW ---------
    upload_file_to_gcs(
        bucket_name=settings.GCS_BUCKET_NAME,
        local_path=str(raw_path),
        blob_name=f"raw/signalconso_{today}.csv",
    )

    # --------- UPLOAD PROCESSED ---------
    upload_file_to_gcs(
        bucket_name=settings.GCS_BUCKET_NAME,
        local_path=str(clean_path),
        blob_name=f"processed/signalconso_clean_{today}.csv",
    )

    print("Upload terminé vers GCS")
    print(f"Lignes brutes : {len(raw_df)}")
    print(f"Lignes propres : {len(clean_df)}")


if __name__ == "__main__":
    main()