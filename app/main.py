import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError, HTTPException

from api import endpoints
from core.schemas import ApiResponse, ResponseStatus, ErrorDetail
from config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL.upper())
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OCR Service API",
    description="A service to extract text from various document types.",
    version="1.0.0",
)

# --- Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request,
                                       exc: RequestValidationError):
    """Handles Pydantic validation errors."""
    error_detail = ErrorDetail(code="VALIDATION_ERROR", message=str(exc))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ApiResponse(
            request_id="N/A", status=ResponseStatus.ERROR, error=error_detail
        ).dict(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handles FastAPI's built-in HTTPExceptions to conform to ApiResponse."""
    try:
        # Assumes detail is a dict from our custom error model
        error_dict = exc.detail
        error_detail = ErrorDetail(
            code=error_dict.get('code', 'HTTP_ERROR'),
            message=error_dict.get('message', str(exc.detail)))
    except (TypeError, AttributeError):
        # Fallback for standard HTTPException details
        error_detail = ErrorDetail(code="HTTP_ERROR", message=str(exc.detail))
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse(
            request_id="N/A", status=ResponseStatus.ERROR, error=error_detail
        ).dict(exclude_none=True),
    )

# --- Routers ---
app.include_router(endpoints.router, prefix="/api/v2", tags=["OCR"])


@app.get("/", tags=["Health Check"])
async def read_root():
    """Root endpoint to check if the service is running."""
    return {"message": "Hello from FastAPI, OCR service is running!"}