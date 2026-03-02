from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union, Dict, Any
from enum import Enum

class OcrRequest(BaseModel):
    url: Optional[str] = None
    local_path: Optional[str] = None
    request_id: str
    file_size_mb: float
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://www.orimi.com/pdf-test.pdf",
                "local_path": "",
                "request_id": "b1b7b6e0-9b0a-4b0e-8b0a-9b0a4b0e8b0a",
                "file_size_mb": 1.5,
            }
        }
    )

class AirflowTask(BaseModel):
    url: Optional[str] = None
    local_path: Optional[str] = None
    request_id: str
    file_size_mb: float

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectedStamp(BaseModel):
    label: str
    confidence: float
    box: BoundingBox

class ImageOcrResult(BaseModel):
    filename: Optional[str] = None
    text: Optional[str] = None
    stamps: List[DetectedStamp] = Field(default_factory=list)
    tables_html: List[str] = Field(default_factory=list)

class DocOcrResult(BaseModel):
    text: Optional[str] = None
    images: List[ImageOcrResult] = Field(default_factory=list)
    service: Optional[str] = None
    early_stop_triggered: bool = False

class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"

class ErrorDetail(BaseModel):
    code: str
    message: str

class ApiResponse(BaseModel):
    request_id: str
    status: ResponseStatus
    data: Optional[Union[DocOcrResult, Dict[str, Any]]] = None
    error: Optional[ErrorDetail] = None