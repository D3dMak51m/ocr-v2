from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict
from enum import Enum

# --- Request Models ---


class OcrRequest(BaseModel):
    url: Optional[str] = None
    local_path: Optional[str] = None
    request_id: str
    file_size_mb: float

    class Config:
        schema_extra = {
            "example": {
                "url": "https://www.orimi.com/pdf-test.pdf",
                "local_path": "",
                "request_id": "b1b7b6e0-9b0a-4b0e-8b0a-9b0a4b0e8b0a",
                "file_size_mb": 1.5,
            }
        }


class AirflowTask(BaseModel):
    url: Optional[str] = None
    local_path: Optional[str] = None
    request_id: str
    file_size_mb: float


# --- OCR Result Data Models ---


class ImageOcrResult(BaseModel):
    filename: Optional[str] = None
    text: Optional[str] = None
    encoding: Optional[str] = None
    encoding_conf: Optional[float] = None


class DocOcrResult(BaseModel):
    text: Optional[str] = None
    images: List[ImageOcrResult] = Field(default_factory=list)
    service: Optional[str] = None


# --- Standard API Response Structure ---


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class ErrorDetail(BaseModel):
    code: str
    message: str


class ApiResponse(BaseModel):
    request_id: str
    status: ResponseStatus
    data: Optional[Union[DocOcrResult, Dict]] = None
    error: Optional[ErrorDetail] = None
