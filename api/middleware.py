"""
SQL Injection Detection middleware helpers.

This module lets the trained ML detector run before application route code.
It supports two integration styles:

- in-process middleware for Flask, FastAPI/Starlette, and Django
- shared extraction/scanning helpers used by the standalone FastAPI scanner
"""

import json
import logging
import os
from email import policy
from email.parser import BytesParser
from typing import Callable, Iterable, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("sqli_detector")

DEFAULT_THRESHOLD = 0.35
DEFAULT_SKIP_PATHS = {
    "/docs",
    "/openapi.json",
    "/redoc",
    "/health",
    "/favicon.ico",
}
SCANNABLE_HEADERS = {"user-agent", "referer", "x-forwarded-for", "origin"}


def get_threshold(default: float = DEFAULT_THRESHOLD) -> float:
    """Read SQLI_THRESHOLD from the environment with a safe fallback."""
    raw_value = os.getenv("SQLI_THRESHOLD")
    if raw_value is None:
        return default

    try:
        threshold = float(raw_value)
    except ValueError:
        logger.warning("Invalid SQLI_THRESHOLD=%r; using %.2f", raw_value, default)
        return default

    if not 0.0 <= threshold <= 1.0:
        logger.warning("SQLI_THRESHOLD=%r outside 0..1; using %.2f", raw_value, default)
        return default

    return threshold


def extract_params(
    url: Optional[str] = None,
    form_data: Optional[dict] = None,
    json_body=None,
    headers: Optional[dict] = None,
) -> dict:
    """
    Extract user-controlled string values from an HTTP request.

    Scans URL query values, form fields, JSON body values, and a small allowlist
    of headers that commonly contain attacker-controlled content.
    """
    params = {}

    if url:
        parsed = urlparse(url)
        query = parsed.query if parsed.query else url.lstrip("?")
        for key, values in parse_qs(query, keep_blank_values=True).items():
            for i, val in enumerate(values):
                param_key = key if len(values) == 1 else f"{key}[{i}]"
                params[f"url:{param_key}"] = val

    if form_data:
        for key, val in form_data.items():
            for item_key, item_value in _iter_named_values(str(key), val):
                params[f"form:{item_key}"] = _stringify_value(item_value)

    if json_body is not None:
        for key, val in _flatten_json(json_body).items():
            params[f"json:{key}"] = _stringify_value(val)

    if headers:
        for header, val in headers.items():
            if header.lower() in SCANNABLE_HEADERS:
                params[f"header:{header}"] = _stringify_value(val)

    return params


def _iter_named_values(key: str, value):
    """Yield one or more named scalar values, preserving repeated fields."""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            yield key, value[0]
            return
        for index, item in enumerate(value):
            yield f"{key}[{index}]", item
        return

    yield key, value


def _flatten_json(data, prefix: str = "") -> dict:
    """Flatten JSON-like objects into stable parameter names."""
    result = {}

    if isinstance(data, dict):
        for key, val in data.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_json(val, full_key))
        return result

    if isinstance(data, list):
        for index, item in enumerate(data):
            item_key = f"{prefix}[{index}]" if prefix else f"[{index}]"
            result.update(_flatten_json(item, item_key))
        return result

    result[prefix or "$"] = _stringify_value(data)
    return result


def _stringify_value(value) -> str:
    if value is None:
        return ""
    return str(value)


def scan_params(params: dict, pipeline: dict, threshold: float) -> dict:
    """Run extracted parameters through the ML model."""
    from src.predict import predict

    results = []
    sqli_params = []

    for name, value in params.items():
        value = _stringify_value(value)
        if not value.strip():
            continue

        result = predict(value, pipeline, threshold=threshold)
        result["param_name"] = name
        results.append(result)

        if result["label"] == 1:
            sqli_params.append(name)
            logger.warning(
                "sqli_detected param=%s confidence=%s risk=%s sample=%r",
                name,
                result["confidence"],
                result["risk_level"],
                value[:120],
            )

    return {
        "blocked": len(sqli_params) > 0,
        "sqli_params": sqli_params,
        "total_params": len(results),
        "results": results,
    }


def register_flask_middleware(
    app,
    pipeline: dict,
    threshold: Optional[float] = None,
    skip_paths: Optional[Iterable[str]] = None,
):
    """Register SQLi detection as a Flask before_request hook."""
    from flask import jsonify, request

    threshold = get_threshold() if threshold is None else threshold
    skip_paths = set(DEFAULT_SKIP_PATHS if skip_paths is None else skip_paths)

    @app.before_request
    def _sqli_check():
        if request.path in skip_paths:
            return None

        params = extract_params(
            url=request.url,
            form_data=request.form.to_dict(flat=False) if request.form else None,
            json_body=request.get_json(silent=True),
            headers=dict(request.headers),
        )
        scan = scan_params(params, pipeline, threshold)

        if scan["blocked"]:
            logger.warning(
                "request_blocked framework=flask path=%s flagged=%s",
                request.path,
                scan["sqli_params"],
            )
            return jsonify(
                {
                    "error": "Forbidden",
                    "reason": "SQL injection attempt detected",
                    "flagged": scan["sqli_params"],
                    "risk_levels": [
                        r["risk_level"] for r in scan["results"] if r["label"] == 1
                    ],
                }
            ), 403

        return None


class SQLiDetectionMiddleware:
    """
    ASGI middleware for FastAPI and Starlette.

    The middleware consumes the incoming body once, scans JSON or URL-encoded
    form fields, then replays the exact same body to downstream route handlers.
    That replay step is what keeps normal request parsing working after the
    security scan runs.
    """

    def __init__(
        self,
        app,
        pipeline: dict,
        threshold: Optional[float] = None,
        skip_paths: Optional[Iterable[str]] = None,
        max_body_bytes: int = 1_000_000,
    ):
        self.app = app
        self.pipeline = pipeline
        self.threshold = get_threshold() if threshold is None else threshold
        self.skip_paths = set(DEFAULT_SKIP_PATHS if skip_paths is None else skip_paths)
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        if path in self.skip_paths:
            await self.app(scope, receive, send)
            return

        from starlette.datastructures import Headers
        from starlette.responses import JSONResponse

        body, body_too_large = await _read_body(receive, self.max_body_bytes)
        replay_receive = _make_replay_receive(body)

        if body_too_large:
            logger.warning(
                "request_body_rejected path=%s size_over_limit=%s",
                path,
                self.max_body_bytes,
            )
            response = JSONResponse(
                status_code=413,
                content={
                    "error": "Payload Too Large",
                    "reason": "Request body exceeds SQLi scanner size limit",
                    "max_body_bytes": self.max_body_bytes,
                },
            )
            await response(scope, _make_replay_receive(body), send)
            return

        headers = Headers(scope=scope)
        params = _extract_asgi_params(scope, headers, body, self.max_body_bytes)
        scan = scan_params(params, self.pipeline, self.threshold)

        if scan["blocked"]:
            logger.warning(
                "request_blocked framework=asgi path=%s method=%s flagged=%s",
                path,
                scope.get("method", ""),
                scan["sqli_params"],
            )
            response = JSONResponse(
                status_code=403,
                content={
                    "error": "Forbidden",
                    "reason": "SQL injection attempt detected",
                    "flagged": scan["sqli_params"],
                },
            )
            await response(scope, _make_replay_receive(body), send)
            return

        await self.app(scope, replay_receive, send)


async def _read_body(receive, max_body_bytes: int) -> tuple[bytes, bool]:
    """Read the ASGI request body up to max_body_bytes."""
    chunks = []
    more_body = True
    total_size = 0

    while more_body:
        message = await receive()
        chunk = message.get("body", b"")
        total_size += len(chunk)
        if total_size > max_body_bytes:
            return b"".join(chunks), True
        chunks.append(chunk)
        more_body = message.get("more_body", False)

    return b"".join(chunks), False


def _make_replay_receive(body: bytes):
    """Create an ASGI receive callable that replays a previously read body."""
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def _extract_asgi_params(scope, headers, body: bytes, max_body_bytes: int) -> dict:
    query_string = scope.get("query_string", b"").decode("utf-8", errors="replace")
    params = extract_params(url=f"?{query_string}" if query_string else None)

    content_type = headers.get("content-type", "")
    if body and len(body) <= max_body_bytes:
        body_text = body.decode("utf-8", errors="replace")
        if "application/json" in content_type:
            try:
                json_body = json.loads(body_text)
                params.update(extract_params(json_body=json_body))
            except json.JSONDecodeError:
                logger.info("request_body_json_parse_failed path=%s", scope.get("path", ""))
        elif "application/x-www-form-urlencoded" in content_type:
            form_data = parse_qs(body_text, keep_blank_values=True)
            params.update(extract_params(form_data=form_data))
        elif "multipart/form-data" in content_type:
            form_data = _parse_multipart_form_data(content_type, body)
            params.update(extract_params(form_data=form_data))
    elif body:
        logger.info(
            "request_body_scan_skipped path=%s size=%s max_size=%s",
            scope.get("path", ""),
            len(body),
            max_body_bytes,
        )

    header_params = extract_params(headers=dict(headers))
    params.update(header_params)
    return params


def _parse_multipart_form_data(content_type: str, body: bytes) -> dict:
    """Parse text fields from multipart/form-data and skip file content."""
    raw_message = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8") + body
    message = BytesParser(policy=policy.default).parsebytes(raw_message)
    form_data = {}

    if not message.is_multipart():
        return form_data

    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue

        field_name = part.get_param("name", header="content-disposition")
        filename = part.get_param("filename", header="content-disposition")
        if not field_name or filename is not None:
            continue

        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        value = payload.decode(charset, errors="replace")
        _append_multi_value(form_data, field_name, value)

    return form_data


def _append_multi_value(data: dict, key: str, value: str) -> None:
    if key not in data:
        data[key] = value
    elif isinstance(data[key], list):
        data[key].append(value)
    else:
        data[key] = [data[key], value]


class DjangoSQLiMiddleware:
    """Django middleware for SQLi detection."""

    _pipeline = None

    def __init__(self, get_response: Callable):
        self.get_response = get_response
        if DjangoSQLiMiddleware._pipeline is None:
            from src.predict import load_pipeline

            DjangoSQLiMiddleware._pipeline = load_pipeline()

        try:
            from django.conf import settings

            self.threshold = getattr(settings, "SQLI_THRESHOLD", get_threshold())
            self.skip_paths = set(getattr(settings, "SQLI_SKIP_PATHS", DEFAULT_SKIP_PATHS))
        except Exception:
            self.threshold = get_threshold()
            self.skip_paths = set(DEFAULT_SKIP_PATHS)

    def __call__(self, request):
        from django.http import JsonResponse

        if request.path in self.skip_paths:
            return self.get_response(request)

        params = extract_params(
            url=request.get_full_path(),
            form_data=dict(request.POST.lists()) if request.method == "POST" else None,
            headers=dict(request.headers),
        )

        if "application/json" in (request.content_type or ""):
            try:
                json_body = json.loads(request.body)
                params.update(extract_params(json_body=json_body))
            except json.JSONDecodeError:
                logger.info("django_json_parse_failed path=%s", request.path)

        scan = scan_params(params, self._pipeline, self.threshold)

        if scan["blocked"]:
            logger.warning(
                "request_blocked framework=django path=%s flagged=%s",
                request.path,
                scan["sqli_params"],
            )
            return JsonResponse(
                {
                    "error": "Forbidden",
                    "reason": "SQL injection attempt detected",
                    "flagged": scan["sqli_params"],
                },
                status=403,
            )

        return self.get_response(request)
