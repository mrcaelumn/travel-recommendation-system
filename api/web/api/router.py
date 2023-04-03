from fastapi.routing import APIRouter

from api.web.api import docs, echo, monitoring, predictions

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(docs.router)
api_router.include_router(echo.router, prefix="/echo", tags=["echo"])
api_router.include_router(predictions.router, prefix="/predict", tags=["predict"])
