from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path

from src.app.api.schemas.prediction import PredictionRequest, PredictionResponse
from src.app.core.config import get_settings
from src.app.ml.predict import TicketModel, normalize_text
from src.app.services.gcs_service import upload_json_to_gcs, download_json_from_gcs
from src.app.services.gcs_service import find_prediction_in_bucket

router = APIRouter()

settings = get_settings()

model = TicketModel(settings.MODEL_PATH)
model.load()

MODEL_VERSION = settings.MODEL_VERSION

# --------- CREATE PREDICTION ---------
@router.post("", response_model=PredictionResponse)
def create_prediction(request: PredictionRequest):
    try:
        prediction_id = str(uuid4())

        clean_text = normalize_text(request.text)
        prediction, confidence = model.predict_with_proba(request.text)

        now = datetime.utcnow()

        blob_path = (
            f"predictions/{now.year}/{now.month:02d}/{now.day:02d}/{prediction_id}.json"
        )

        data = {
            "id": prediction_id,
            "input_text": request.text,
            "clean_text": clean_text,
            "predicted_category": prediction,
            "confidence": float(confidence),
            "model_version": MODEL_VERSION,
            "created_at": now.isoformat(),
        }
        # Sauvegarde dans GCS
        upload_json_to_gcs(
            bucket_name=settings.GCS_BUCKET_NAME,
            blob_name=blob_path,
            data=data,
        )

        return PredictionResponse(**data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------- GET PREDICTION ---------
@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: str):
    data = find_prediction_in_bucket(
        bucket_name=settings.GCS_BUCKET_NAME,
        prediction_id=prediction_id,
    )

    if data is None:
        raise HTTPException(status_code=404, detail="Prédiction introuvable")

    return PredictionResponse(**data)