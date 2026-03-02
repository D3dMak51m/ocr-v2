import logging
import requests
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from requests.auth import HTTPBasicAuth

from config import settings
from core.schemas import (
    OcrRequest,
    AirflowTask,
    ApiResponse,
    ResponseStatus,
    ErrorDetail,
)
from core.processor import run_ocr
from core.exceptions import OcrBaseException

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


@router.post(
    "/inference",
    response_model=ApiResponse,
    summary="Extract text from a document or image",
    dependencies=[Depends(verify_token)],
)
def text_extraction(request: OcrRequest) -> ApiResponse:
    """
    Performs OCR on a file specified by URL or local path.

    - **Supported extensions:** jpg, png, pdf, doc, docx, ppt, pptx, etc.
    - Files larger than 50MB should use the `/queue_inference` endpoint.
    """
    if request.file_size_mb > 50:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 50MB. Please use the /queue_inference \
                endpoint for large files.",
        )

    try:
        result = run_ocr(request)
        return ApiResponse(
            request_id=request.request_id, status=ResponseStatus.SUCCESS, data=result
        )
    except OcrBaseException as e:
        # Catch custom exceptions and format them nicely
        logger.error(
            f"OCR processing failed for request \
                     {request.request_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(code=e.code, message=str(e)).model_dump(),
        )
    except Exception as e:
        # Catch any unexpected errors
        logger.exception(
            f"An unexpected error occurred for request \
                {request.request_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(code="INTERNAL_SERVER_ERROR", message=str(e)).model_dump(),
        )


@router.post(
    "/queue_inference",
    response_model=ApiResponse,
    summary="Queue a large file for processing via Airflow",
    dependencies=[Depends(verify_token)],
)
def create_airflow_task(request: AirflowTask) -> ApiResponse:
    """
    Triggers an Airflow DAG to process a large file asynchronously.
    """
    dag_id = "airflow_large" if request.file_size_mb > 5 else "airflow_dag"
    airflow_url = f"{settings.AIRFLOW_BASE_URL}/api/v1/dags/{dag_id}/dagRuns"

    try:
        response = requests.post(
            airflow_url,
            json={"conf": request.model_dump()},
            auth=HTTPBasicAuth(settings.AIRFLOW_USER, settings.AIRFLOW_PASSWORD),
            timeout=30,
        )
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

        logger.info(
            f"Successfully triggered Airflow DAG \
                '{dag_id}' for request {request.request_id}"
        )
        return ApiResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={"status": "received", "airflow_response": response.json()},
        )
    except requests.RequestException as e:
        logger.error(
            f"Failed to trigger Airflow DAG for request \
                {request.request_id}. Error: {e}"
        )
        error_detail = ErrorDetail(
            code="AIRFLOW_TRIGGER_FAILED",
            message=f"Failed to communicate with Airflow: {e}",
        )
        # Return 502 if the gateway (Airflow) is down or failing
        return ApiResponse(
            request_id=request.request_id,
            status=ResponseStatus.ERROR,
            error=error_detail,
        )
