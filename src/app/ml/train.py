from __future__ import annotations

import os
import pandas as pd

from src.app.core.config import get_settings
from src.app.services.gcs_service import (
    get_latest_blob,
    download_blob_to_file,
)
from src.app.training.train import train_model  # ton script existant

TMP_FILE = "tmp/latest_dataset.csv"


def main():
    settings = get_settings()

    bucket = settings.GCS_BUCKET_NAME

    # --------- 1. Trouver dernier dataset ---------
    latest_blob = get_latest_blob(
        bucket_name=bucket,
        prefix="processed/"
    )

    if latest_blob is None:
        raise Exception("Aucun dataset trouvé dans GCS")

    print(f"Dataset utilisé : {latest_blob}")

    # --------- 2. Télécharger ---------
    os.makedirs("tmp", exist_ok=True)

    download_blob_to_file(
        bucket_name=bucket,
        blob_name=latest_blob,
        destination_path=TMP_FILE,
    )

    # --------- 3. Entraîner ---------
    model_path = "models/model.joblib"

    train_model(
        data_path=TMP_FILE,
        model_path=model_path
    )

    print("Entraînement terminé")


if __name__ == "__main__":
    main()