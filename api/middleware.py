"""
SQL Injection Detection API — Middleware Helpers
=================================================
Drop-in middleware for the three most common Python web frameworks.
Each middleware extracts all user-supplied parameters from an incoming
request and runs them through the detection model before the request
reaches your application code.

Usage:
    Flask   → register_flask_middleware(app)
    FastAPI → app.add_middleware(SQLiDetectionMiddleware)
    Django  → add 'api.middleware.DjangoSQLiMiddleware' to MIDDLEWARE

⚠  CLASS IMBALANCE REMINDER
    In production, 99%+ of traffic is benign. The default threshold
    of 0.35 is already tuned for this. Monitor your false positive
    rate for the first week after deployment and adjust if needed.
"""

import json
import logging
from urllib.parse import urlparse, parse_qs
from typing import Callable, Optional

logger = logging.getLogger("sqli_detector")


# ─────────────────────────────────────────────────────────────────────────────
# Shared parameter extractor
# Works on any raw HTTP request data — used by all three middlewares.
# ─────────────────────────────────────────────────────────────────────────────

SCANNABLE_HEADERS = {"user-agent", "referer", "x-forwarded-for", "origin"}

def extract_params(
    url:       Optional[str]  = None,
    form_data: Optional[dict] = None,
    json_body: Optional[dict] = None,
    headers:   Optional[dict] = None,
) -> dict:
    """
    Extract all user-supplied string values from an HTTP request.
    Returns a dict of {param_name: value} for every scannable field.

    Scans:
      - URL query string parameters (?id=1&name=foo)
      - POST form fields
      - JSON body values (nested dicts are flattened)
      - A safe subset of HTTP headers
    """
    params = {}

    # URL query string
    if url:
        parsed = urlparse(url)
        for key, values in parse_qs(parsed.query).items():
            for i, val in enumerate(values):
                param_key = key if len(values) == 1 else f"{key}[{i}]"
                params[f"url:{param_key}"] = val

    # Form fields
    if form_data:
        for key, val in form_data.items():
            params[f"form:{key}"] = str(val)

    # JSON body (flatten nested structure)
    if json_body:
        for key, val in _flatten_dict(json_body).items():
            params[f"json:{key}"] = str(val)

    # Headers (only the subset likely to carry injection)
    if headers:
        for header, val in headers.items():
            if header.lower() in SCANNABLE_HEADERS:
                params[f"header:{header}"] = str(val)

    return params


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict into dot-separated keys."""
    result = {}
    for key, val in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            result.update(_flatten_dict(val, full_key))
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    result.update(_flatten_dict(item, f"{full_key}[{i}]"))
                else:
                    result[f"{full_key}[{i}]"] = str(item)
        else:
            result[full_key] = str(val)
    return result


def scan_params(params: dict, pipeline: dict, threshold: float) -> dict:
    """
    Run all extracted parameters through the model.
    Returns a summary dict with blocked status and per-param results.
    """
    from src.predict import predict

    results      = []
    sqli_params  = []

    for name, value in params.items():
        if not value or not value.strip():
            continue
        result = predict(value, pipeline, threshold=threshold)
        result["param_name"] = name
        results.append(result)
        if result["label"] == 1:
            sqli_params.append(name)
            logger.warning(
                f"SQLi detected | param={name} | "
                f"confidence={result['confidence']} | "
                f"risk={result['risk_level']} | "
                f"query={value[:120]!r}"
            )

    return {
        "blocked":      len(sqli_params) > 0,
        "sqli_params":  sqli_params,
        "total_params": len(results),
        "results":      results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flask middleware
# ─────────────────────────────────────────────────────────────────────────────

def register_flask_middleware(app, pipeline: dict, threshold: float = 0.35):
    """
    Register SQLi detection as a Flask before_request hook.
    Blocks any request where at least one parameter is flagged.

    Usage:
        from api.middleware import register_flask_middleware
        from src.predict    import load_pipeline

        pipeline = load_pipeline()
        register_flask_middleware(app, pipeline, threshold=0.35)
    """
    from flask import request, abort, jsonify

    @app.before_request
    def _sqli_check():
        params = extract_params(
            url=request.url,
            form_data=request.form.to_dict() if request.form else None,
            json_body=request.get_json(silent=True),
            headers=dict(request.headers),
        )
        scan = scan_params(params, pipeline, threshold)
        if scan["blocked"]:
            logger.warning(f"Request blocked | path={request.path} | "
                           f"flagged_params={scan['sqli_params']}")
            return jsonify({
                "error":       "Forbidden",
                "reason":      "SQL injection attempt detected",
                "flagged":     scan["sqli_params"],
                "risk_levels": [r["risk_level"] for r in scan["results"]
                                if r["label"] == 1],
            }), 403


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI / Starlette middleware
# ─────────────────────────────────────────────────────────────────────────────

class SQLiDetectionMiddleware:
    """
    ASGI middleware for FastAPI / Starlette.
    Intercepts every request and blocks any that contain SQLi.

    Usage:
        from fastapi           import FastAPI
        from api.middleware    import SQLiDetectionMiddleware
        from src.predict       import load_pipeline

        pipeline = load_pipeline()
        app      = FastAPI()
        app.add_middleware(SQLiDetectionMiddleware,
                           pipeline=pipeline, threshold=0.35)
    """

    def __init__(self, app, pipeline: dict, threshold: float = 0.35):
        self.app       = app
        self.pipeline  = pipeline
        self.threshold = threshold

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from urllib.parse import parse_qs
        from starlette.requests  import Request
        from starlette.responses import JSONResponse

        request = Request(scope, receive)

        # Extract query string
        query_string = scope.get("query_string", b"").decode("utf-8")
        url_params   = {}
        for key, values in parse_qs(query_string).items():
            for i, val in enumerate(values):
                url_params[f"url:{key}" if len(values)==1 else f"url:{key}[{i}]"] = val

        # Form / JSON body
        form_data = None
        json_body = None
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                json_body = await request.json()
            except Exception:
                pass
        elif "application/x-www-form-urlencoded" in content_type or \
             "multipart/form-data" in content_type:
            try:
                form = await request.form()
                form_data = dict(form)
            except Exception:
                pass

        all_params = {**url_params}
        if form_data:
            all_params.update({f"form:{k}": str(v) for k, v in form_data.items()})
        if json_body and isinstance(json_body, dict):
            all_params.update({f"json:{k}": str(v)
                               for k, v in _flatten_dict(json_body).items()})
        for header in SCANNABLE_HEADERS:
            val = request.headers.get(header)
            if val:
                all_params[f"header:{header}"] = val

        scan = scan_params(all_params, self.pipeline, self.threshold)

        if scan["blocked"]:
            path = scope.get("path", "")
            logger.warning(f"Request blocked | path={path} | "
                           f"flagged_params={scan['sqli_params']}")
            response = JSONResponse(
                status_code=403,
                content={
                    "error":   "Forbidden",
                    "reason":  "SQL injection attempt detected",
                    "flagged": scan["sqli_params"],
                },
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


# ─────────────────────────────────────────────────────────────────────────────
# Django middleware
# ─────────────────────────────────────────────────────────────────────────────

class DjangoSQLiMiddleware:
    """
    Django WSGI middleware for SQLi detection.

    Add to settings.py:
        MIDDLEWARE = [
            'api.middleware.DjangoSQLiMiddleware',
            ...
        ]

    The pipeline is loaded once when Django starts.
    Set SQLI_THRESHOLD in settings.py to override the default (0.35).
    """

    _pipeline = None

    def __init__(self, get_response: Callable):
        self.get_response = get_response
        if DjangoSQLiMiddleware._pipeline is None:
            from src.predict import load_pipeline
            DjangoSQLiMiddleware._pipeline = load_pipeline()

        try:
            from django.conf import settings
            self.threshold = getattr(settings, "SQLI_THRESHOLD", 0.35)
        except Exception:
            self.threshold = 0.35

    def __call__(self, request):
        from django.http import JsonResponse

        # Build param dict from Django request
        params = extract_params(
            url=request.get_full_path(),
            form_data=request.POST.dict() if request.method == "POST" else None,
            headers=dict(request.headers),
        )

        # JSON body
        content_type = request.content_type or ""
        if "application/json" in content_type:
            try:
                json_body = json.loads(request.body)
                if isinstance(json_body, dict):
                    params.update({f"json:{k}": str(v)
                                   for k, v in _flatten_dict(json_body).items()})
            except Exception:
                pass

        scan = scan_params(params, self._pipeline, self.threshold)

        if scan["blocked"]:
            logger.warning(f"Request blocked | path={request.path} | "
                           f"flagged_params={scan['sqli_params']}")
            return JsonResponse(
                {
                    "error":   "Forbidden",
                    "reason":  "SQL injection attempt detected",
                    "flagged": scan["sqli_params"],
                },
                status=403,
            )

        return self.get_response(request)