import asyncio
import json
import unittest

import api.middleware as middleware
from api.middleware import SQLiDetectionMiddleware


async def echo_body_app(scope, receive, send):
    body = b""
    more_body = True

    while more_body:
        message = await receive()
        body += message.get("body", b"")
        more_body = message.get("more_body", False)

    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


async def call_asgi(app, *, path="/submit", body=b"", headers=None, query_string=b""):
    headers = headers or []
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": query_string,
        "headers": headers,
    }
    sent = []
    received = False

    async def receive():
        nonlocal received
        if received:
            return {"type": "http.request", "body": b"", "more_body": False}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)

    status = next(msg["status"] for msg in sent if msg["type"] == "http.response.start")
    response_body = b"".join(
        msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body"
    )
    return status, response_body


def fake_scan_params(params, pipeline, threshold):
    flagged = [
        name
        for name, value in params.items()
        if "or 1=1" in str(value).lower()
    ]
    return {
        "blocked": bool(flagged),
        "sqli_params": flagged,
        "total_params": len(params),
        "results": [
            {
                "param_name": name,
                "label": 1 if name in flagged else 0,
                "risk_level": "High" if name in flagged else "Low",
            }
            for name in params
        ],
    }


class SQLiDetectionMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.original_scan_params = middleware.scan_params
        middleware.scan_params = fake_scan_params

    def tearDown(self):
        middleware.scan_params = self.original_scan_params

    def test_blocks_sqli_json_body(self):
        body = json.dumps({"username": "admin", "password": "' OR 1=1--"}).encode()
        app = SQLiDetectionMiddleware(echo_body_app, pipeline={}, threshold=0.35)

        status, response_body = asyncio.run(
            call_asgi(
                app,
                body=body,
                headers=[(b"content-type", b"application/json")],
            )
        )

        payload = json.loads(response_body)
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"], "Forbidden")
        self.assertIn("json:password", payload["flagged"])

    def test_replays_benign_json_body_to_downstream_app(self):
        body = json.dumps({"search": "normal product lookup"}).encode()
        app = SQLiDetectionMiddleware(echo_body_app, pipeline={}, threshold=0.35)

        status, response_body = asyncio.run(
            call_asgi(
                app,
                body=body,
                headers=[(b"content-type", b"application/json")],
            )
        )

        self.assertEqual(status, 200)
        self.assertEqual(response_body, body)

    def test_skip_path_bypasses_scan(self):
        calls = {"count": 0}

        def counting_scan(params, pipeline, threshold):
            calls["count"] += 1
            return fake_scan_params(params, pipeline, threshold)

        middleware.scan_params = counting_scan
        body = json.dumps({"password": "' OR 1=1--"}).encode()
        app = SQLiDetectionMiddleware(
            echo_body_app,
            pipeline={},
            threshold=0.35,
            skip_paths={"/health"},
        )

        status, response_body = asyncio.run(
            call_asgi(
                app,
                path="/health",
                body=body,
                headers=[(b"content-type", b"application/json")],
            )
        )

        self.assertEqual(status, 200)
        self.assertEqual(response_body, body)
        self.assertEqual(calls["count"], 0)

    def test_rejects_body_larger_than_scan_limit(self):
        body = json.dumps({"payload": "x" * 128}).encode()
        app = SQLiDetectionMiddleware(
            echo_body_app,
            pipeline={},
            threshold=0.35,
            max_body_bytes=16,
        )

        status, response_body = asyncio.run(
            call_asgi(
                app,
                body=body,
                headers=[(b"content-type", b"application/json")],
            )
        )

        payload = json.loads(response_body)
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"], "Payload Too Large")
        self.assertEqual(payload["max_body_bytes"], 16)


if __name__ == "__main__":
    unittest.main()
