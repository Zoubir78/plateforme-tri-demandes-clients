from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "platforme-tri-demandes-clients"}