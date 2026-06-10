"""
SQL Injection Detection API — Request & Response Schemas
=========================================================
Defines all Pydantic models used by app.py for request
validation and response serialisation.
"""

from pydantic import BaseModel, Field
from typing import Any, Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────────────────────────────────────

class SingleQueryRequest(BaseModel):
    """Body for POST /detect — classify one query string."""
    query: str = Field(
        ...,
        min_length=1,
        description="Raw query string to classify (URL params, form values, etc.)",
        examples=["SELECT * FROM users WHERE id=1 OR 1=1--"],
    )
    threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Classification threshold. P(SQLi) >= threshold → flagged as attack. "
            "Lower = more sensitive (fewer missed attacks, more false alarms). "
            "Default 0.35 is tuned for production traffic (99%+ benign). "
            "⚠  Tune this via evaluate.py Step 6b before deploying."
        ),
    )


class BatchQueryRequest(BaseModel):
    """Body for POST /detect/batch — classify multiple queries in one call."""
    queries: List[str] = Field(
        ...,
        min_length=1,
        description="List of raw query strings to classify.",
        examples=[["SELECT * FROM users", "' OR 1=1--", "normal search term"]],
    )
    threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Classification threshold applied to all queries in the batch.",
    )


class ScanRequestParams(BaseModel):
    """
    Body for POST /scan — extract and classify all parameters
    from a raw HTTP request before it reaches your application.
    """
    url: Optional[str] = Field(
        default=None,
        description="Full URL including query string, e.g. /search?q=foo&id=1",
    )
    form_data: Optional[dict[str, Any]] = Field(
        default=None,
        description="POST form fields as a flat key→value dict.",
        examples=[{"username": "admin", "password": "' OR 1=1--"}],
    )
    json_body: Optional[Any] = Field(
        default=None,
        description="JSON request body as a dict (nested values are flattened).",
    )
    headers: Optional[dict[str, Any]] = Field(
        default=None,
        description="Request headers to inspect (User-Agent, Referer, etc.)",
    )
    threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Classification threshold applied to every extracted parameter.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class DetectionResult(BaseModel):
    """Single-query classification result."""
    query:      str
    cleaned:    str
    label:      int                      # 0 = Benign, 1 = SQLi
    prediction: str                      # "Benign" or "SQLi"
    confidence: float                    # P(SQLi) — 0.0 to 1.0
    threshold:  float
    risk_level: str                      # "Low" / "Medium" / "High"


class BatchDetectionResult(BaseModel):
    """Batch classification result."""
    total:       int
    sqli_count:  int
    benign_count: int
    sqli_rate:   float                   # fraction of queries flagged
    results:     List[DetectionResult]


class ParamScanResult(BaseModel):
    """
    Result from POST /scan — one entry per extracted parameter.
    If any parameter is flagged, 'blocked' is True and the
    request should be rejected before reaching your application.
    """
    blocked:      bool                   # True if ANY parameter is SQLi
    sqli_params:  List[str]              # names of flagged parameters
    total_params: int
    results:      List[dict]             # per-param DetectionResult dicts


class HealthResponse(BaseModel):
    status:         str
    model:          str
    threshold:      float
    version:        str
