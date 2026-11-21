"""Consistent JSON error responses. Stack traces are never returned to clients.

Every error body is {"detail": str, "code": str, "request_id": str}. Unexpected
exceptions become a generic 500 in app.core.middleware.RequestContextMiddleware.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ml.inference import ModelLoadError

logger = logging.getLogger("modeldoctor")

_HTTP_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "too_large",
    415: "unsupported_media_type",
    422: "validation_error",
}


class ApiError(Exception):
    """Application error carrying an HTTP status code and a client-safe message."""

    def __init__(self, status_code: int, detail: str, code: str = "error"):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code


def not_found(detail: str) -> ApiError:
    return ApiError(404, detail, "not_found")


def bad_request(detail: str) -> ApiError:
    return ApiError(400, detail, "bad_request")


def conflict(detail: str) -> ApiError:
    return ApiError(409, detail, "conflict")


def unprocessable(detail: str) -> ApiError:
    return ApiError(422, detail, "validation_error")


def service_unavailable(detail: str, code: str = "model_unavailable") -> ApiError:
    return ApiError(503, detail, code)


def error_body(detail: str, code: str, request_id: Optional[str]) -> Dict[str, Any]:
    return {"detail": detail, "code": code, "request_id": request_id or "-"}


def _request_id(request: Request) -> Optional[str]:
    return request.scope.get("state", {}).get("request_id")


def _respond(request: Request, status: int, detail: str, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=error_body(detail, code, _request_id(request)))


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _respond(request, exc.status_code, exc.detail, exc.code)

    @app.exception_handler(ModelLoadError)
    async def _model_load_error(request: Request, exc: ModelLoadError):
        logger.error("Model could not be loaded: %s", exc)
        return _respond(request, 503, f"The model is unavailable: {exc} Retrain the model.", "model_unavailable")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        return _respond(request, exc.status_code, str(exc.detail), _HTTP_CODES.get(exc.status_code, "http_error"))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        if any(err.get("type") == "json_invalid" for err in errors):
            return _respond(request, 400, "Request body is not valid JSON.", "malformed_json")
        messages = []
        for err in errors:
            loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
            messages.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
        return _respond(request, 422, "; ".join(messages) or "Invalid request", "validation_error")
