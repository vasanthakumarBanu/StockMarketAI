"""
Health and Status Routes
"""

from fastapi import APIRouter
from src.stockmarketai.service import get_service
from src.api.schemas import HealthResponse

router = APIRouter(tags=["Health & Status"])


@router.get("/health", response_model=HealthResponse, summary="Check API Health")
@router.get("/api/v1/health", response_model=HealthResponse, summary="Check API Health (v1)")
def get_health():
    """Returns the operational status and uptime of the StockMarketAI service."""
    service = get_service()
    return service.get_status()


@router.get("/api/v1/status", response_model=HealthResponse, summary="Get System Status")
def get_status():
    """Detailed status including loaded models and cached symbols."""
    service = get_service()
    return service.get_status()


@router.get("/api/v1/models", summary="Get Trained Models Information")
def get_models():
    """Returns details of all trained models, evaluation metrics, and feature lists."""
    service = get_service()
    return {
        "models_summary": service.results_summary,
        "feature_columns": service.feature_columns,
        "active_targets": list(service.models.keys()),
        "has_direction_classifier": service.classification_model is not None,
        "status": "ready"
    }
