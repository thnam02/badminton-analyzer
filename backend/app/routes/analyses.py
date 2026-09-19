"""Product-facing analysis history and aggregate result endpoints (Phase D)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.analysis_aggregator import (
    compare_analyses,
    get_analysis_result,
    list_analyses,
)

router = APIRouter(tags=["analyses"])


@router.get("/analyses")
def get_analyses(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    """List recent local analyses from the file-backed index / outputs."""
    items = list_analyses(limit=limit)
    return {"count": len(items), "analyses": items}


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> dict:
    """Return the product-facing aggregate for one analysis (no recompute)."""
    try:
        return get_analysis_result(analysis_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/compare")
def get_compare(
    left: str = Query(..., description="Previous / left analysis id"),
    right: str = Query(..., description="Current / right analysis id"),
) -> dict:
    """Compare two compatible local analyses on shared metrics."""
    try:
        return compare_analyses(left, right)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
