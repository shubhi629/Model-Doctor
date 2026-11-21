"""ASGI middleware: request IDs, request-size limits and access logging."""

import json
import logging
import re
import time
import uuid
from typing import Callable

from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import MB, MULTIPART_OVERHEAD_BYTES, settings
from app.core.errors import error_body
from app.core.log_config import request_id_var

logger = logging.getLogger("modeldoctor.access")
error_logger = logging.getLogger("modeldoctor")

REQUEST_ID_HEADER = "x-request-id"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _header(scope: Scope, name: str) -> str:
    for key, value in scope.get("headers", []):
        if key.decode("latin-1").lower() == name:
            return value.decode("latin-1")
    return ""


def _too_large_message(limit: int) -> str:
    return f"Request body exceeds the {limit / MB:g} MB limit."


def _body_limit(scope: Scope) -> int:
    if _header(scope, "content-type").lower().startswith("multipart/"):
        return settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
    return settings.max_json_bytes


class RequestContextMiddleware:
    """
    - Uses the client's X-Request-ID when it is safe (letters, digits, . _ -; up to 64
      characters), otherwise generates one, and echoes it on every response.
    - Rejects bodies above the configured limit with 413 (checked from Content-Length
      and while streaming, so chunked uploads cannot bypass it).
    - Logs one line per request: method, path, status, duration. Never the body or query.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        client_id = _header(scope, REQUEST_ID_HEADER)
        request_id = client_id if _VALID_REQUEST_ID.match(client_id) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        scope.setdefault("state", {})["request_id"] = request_id
        start = time.perf_counter()
        status = {"code": 500}
        response_started = {"value": False}
        limit = _body_limit(scope)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                response_started["value"] = True
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        received = {"bytes": 0}

        async def limited_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                received["bytes"] += len(message.get("body", b""))
                if received["bytes"] > limit:
                    # FastAPI re-raises HTTPException from body parsing; the error handler renders it.
                    raise HTTPException(413, _too_large_message(limit))
            return message

        try:
            declared = _header(scope, "content-length")
            if declared.isdigit() and int(declared) > limit:
                await self._send_error(send_with_id, 413, _too_large_message(limit), "too_large", request_id)
                return
            await self.app(scope, limited_receive, send_with_id)
        except Exception:
            # Handled here rather than by Starlette's outer ServerErrorMiddleware so the
            # response still carries the request ID and the log line is correlated.
            error_logger.exception("Unhandled server error")
            if response_started["value"]:
                raise
            await self._send_error(
                send_with_id, 500, "Internal server error. Check the server logs for details.",
                "internal_error", request_id,
            )
        finally:
            logger.info(
                "request completed",
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status["code"],
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                },
            )
            request_id_var.reset(token)

    @staticmethod
    async def _send_error(send: Callable, status: int, detail: str, code: str, request_id: str) -> None:
        body = json.dumps(error_body(detail, code, request_id)).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})
