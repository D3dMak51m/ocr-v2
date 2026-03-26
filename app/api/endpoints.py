import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import httpx
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings
from core.schemas import (
    OcrRequest,
    AirflowTask,
    ApiResponse,
    ResponseStatus,
    ErrorDetail,
)
from core.processor import run_ocr
from core.exceptions import FileTooLargeError, OcrBaseException

logger = logging.getLogger(__name__)
router = APIRouter()
token_auth_scheme = HTTPBearer()


def verify_token(http_auth: HTTPAuthorizationCredentials = Depends(token_auth_scheme)):
    """Dependency to verify the API token."""
    if http_auth.credentials != settings.API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
        )


def _build_file_too_large_http_exception(exc: FileTooLargeError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=ErrorDetail(code=exc.code, message=str(exc)).model_dump(),
    )


async def _execute_ocr_request(
    request: OcrRequest,
    *,
    enforce_public_size_limit: bool,
) -> ApiResponse:
    max_file_size_mb = settings.MAX_SYNC_FILE_SIZE_MB if enforce_public_size_limit else None

    if enforce_public_size_limit and request.file_size_mb > settings.MAX_SYNC_FILE_SIZE_MB:
        raise _build_file_too_large_http_exception(
            FileTooLargeError(
                f"File size exceeds {settings.MAX_SYNC_FILE_SIZE_MB} MB. "
                "Please use the /queue_inference endpoint for large files."
            )
        )

    try:
        result = await run_ocr(request, max_file_size_mb=max_file_size_mb)
        return ApiResponse(
            request_id=request.request_id, status=ResponseStatus.SUCCESS, data=result
        )
    except FileTooLargeError as e:
        logger.error(
            "OCR request %s exceeded size limit: %s",
            request.request_id,
            e,
        )
        raise _build_file_too_large_http_exception(e)
    except OcrBaseException as e:
        logger.error(
            f"OCR processing failed for request {request.request_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(code=e.code, message=str(e)).model_dump(),
        )
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred for request {request.request_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(code="INTERNAL_SERVER_ERROR", message=str(e)).model_dump(),
        )


@router.post(
    "/inference",
    response_model=ApiResponse,
    summary="Extract text from a document or image",
    dependencies=[Depends(verify_token)],
)
async def text_extraction(request: OcrRequest) -> ApiResponse:
    """
    Performs OCR on a file specified by URL or local path.

    - **Supported extensions:** jpg, png, pdf, doc, docx, ppt, pptx, etc.
    - Files larger than 50MB should use the `/queue_inference` endpoint.
    """
    return await _execute_ocr_request(request, enforce_public_size_limit=True)


@router.post(
    "/internal/inference",
    response_model=ApiResponse,
    include_in_schema=False,
    dependencies=[Depends(verify_token)],
)
async def internal_text_extraction(request: OcrRequest) -> ApiResponse:
    """
    Internal OCR endpoint for orchestrators that are allowed to process files
    beyond the public synchronous size cap.
    """
    return await _execute_ocr_request(request, enforce_public_size_limit=False)


@router.post(
    "/queue_inference",
    response_model=ApiResponse,
    summary="Queue a large file for processing via Airflow",
    dependencies=[Depends(verify_token)],
)
async def create_airflow_task(request: AirflowTask) -> ApiResponse:
    """
    Triggers an Airflow DAG to process a large file asynchronously.
    """
    dag_id = "airflow_large" if request.file_size_mb > 5 else "airflow_dag"
    airflow_url = f"{settings.AIRFLOW_BASE_URL}/api/v1/dags/{dag_id}/dagRuns"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                airflow_url,
                json={"conf": request.model_dump()},
                auth=(settings.AIRFLOW_USER, settings.AIRFLOW_PASSWORD),
            )
            response.raise_for_status()

        try:
            airflow_payload = response.json()
        except ValueError:
            airflow_payload = response.text or None

        logger.info(
            f"Successfully triggered Airflow DAG '{dag_id}' for request {request.request_id}"
        )
        return ApiResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={"status": "received", "airflow_response": airflow_payload},
        )
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Airflow returned HTTP {e.response.status_code} for request {request.request_id}"
        )
        error_detail = ErrorDetail(
            code="AIRFLOW_TRIGGER_FAILED",
            message=f"Airflow returned HTTP {e.response.status_code}",
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ApiResponse(
                request_id=request.request_id,
                status=ResponseStatus.ERROR,
                error=error_detail,
            ).model_dump(exclude_none=True),
        )
    except httpx.RequestError as e:
        logger.error(
            f"Failed to connect to Airflow for request {request.request_id}: {e}"
        )
        error_detail = ErrorDetail(
            code="AIRFLOW_TRIGGER_FAILED",
            message=f"Failed to communicate with Airflow: {e}",
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ApiResponse(
                request_id=request.request_id,
                status=ResponseStatus.ERROR,
                error=error_detail,
            ).model_dump(exclude_none=True),
        )
