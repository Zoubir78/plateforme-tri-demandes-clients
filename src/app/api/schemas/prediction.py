from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class PredictionResponse(BaseModel):
    id: str
    input_text: str
    clean_text: str
    predicted_category: str
    confidence: float
    model_version: str