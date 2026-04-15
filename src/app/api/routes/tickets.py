from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.app.ml.predict import TicketModel

router = APIRouter()

model = TicketModel("models/model.joblib")
model.load()


# --------- SCHEMAS ---------
class TicketRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    predicted_category: str


class PredictionWithConfidenceResponse(BaseModel):
    text: str
    predicted_category: str
    confidence: float


# --------- ENDPOINT SIMPLE ---------
@router.post("/predict", response_model=PredictionResponse)
def predict_ticket(request: TicketRequest):
    try:
        prediction = model.predict(request.text)

        return PredictionResponse(
            text=request.text,
            predicted_category=prediction,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------- ENDPOINT AVEC CONFIANCE ---------
@router.post("/predict_proba", response_model=PredictionWithConfidenceResponse)
def predict_ticket_with_confidence(request: TicketRequest):
    try:
        confidence_threshold = 0.5
        prediction, confidence = model.predict_with_proba(request.text)
        confidence = round(confidence, 2)
        if confidence < confidence_threshold:
            predicted_label = "A VERIFIER"
        else:
            predicted_label = prediction

        return PredictionWithConfidenceResponse(
            text=request.text,
            predicted_category=predicted_label,
            confidence=confidence,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))