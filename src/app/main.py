from fastapi import FastAPI
from src.app.api.routes.health import router as health_router
from src.app.api.routes.tickets import router as tickets_router
from src.app.config import settings

app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(health_router)
app.include_router(tickets_router, prefix="/tickets", tags=["tickets"])

@app.get("/")
def root() -> dict:
  return {
    "message": "Bienvenue sur la plateforme intelligente de tri des demandes clients"
}