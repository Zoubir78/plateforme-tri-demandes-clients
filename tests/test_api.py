from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    
def test_predict_ticket_endpoint() -> None:
    payload = {
        "subject": "Problème de facture",
        "description": "Je n'arrive pas à télécharger ma facture.",
    }
    response = client.post("/tickets/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "priority" in data
    assert "confidence" in data