from pydantic import BaseModel
from typing import List, Optional
from ocr_models import extensions_wrapper


class TextractRequest(BaseModel):
    url: Optional[str] = None
    local_path: Optional[str] = None
    request_id: str
    file_size_mb: float

    def __init__(self, url: str, local_path: str, request_id: str, file_size_mb: float):
        super().__init__(
            url=url,
            local_path=local_path,
            request_id=request_id,
            file_size_mb=file_size_mb,
        )

    def __str__(self):
        return (
            f"TextractRequest(url='{self.url}', local_path='{self.local_path}', "
            f"request_id='{self.request_id}', file_size_mb={self.file_size_mb})"
        )


class GeneralResponse(BaseModel):
    request_id: str
    status: str
    msg: str
    result: Optional[str] = None

    def __init__(self, request_id: str, status: str, msg: str, result: str):
        super().__init__(request_id=request_id, status=status, msg=msg, result=result)

    def __str__(self):
        return (
            f"GeneralResponse(request_id='{self.request_id}', status='{self.status}', "
            f"msg='{self.msg}', result='{self.result}')"
        )


class ImageOcrResult(BaseModel):
    status: Optional[str] = None
    text: Optional[str] = None
    filename: Optional[str] = None
    language: Optional[str] = None
    language_score: Optional[str] = None
    encoding: Optional[str] = None
    encoding_conf: Optional[str] = None

    def __init__(
        self,
        status: str,
        text: str,
        language: str,
        language_score: str,
        encoding: str,
        encoding_conf: str,
    ):
        super().__init__(
            status=status,
            text=text,
            language=language,
            language_score=language_score,
            encoding=encoding,
            encoding_conf=encoding_conf,
        )

    def __str__(self):
        return (
            f"ImageOcrResult(status='{self.status}', text='{self.text}', "
            f"language='{self.language}', language_score='{self.language_score}', "
            f"encoding='{self.encoding}', encoding_conf='{self.encoding_conf}')"
        )


class ErrorOcrResult(BaseModel):
    status: str = "error"
    message: str
    status_code: int = 500
    

class DocOcrResult(BaseModel):
    text: Optional[str] = None
    images: List[ImageOcrResult]
    service: Optional[str] = None

    def __init__(self, text: str, images: List[ImageOcrResult], service: str):
        super().__init__(text=text, images=images, service=service)

    def __str__(self):
        images_str = "\n    ".join(str(image) for image in self.images)
        return (
            f"DocOcrResult(text='{self.text}', service='{self.service}', "
            f"images=[\n    {images_str}\n])"
        )


class AirflowTask(BaseModel):
    url: Optional[str] = None
    local_path: Optional[str] = None
    request_id: str
    file_size_mb: float
    callback_url: str

    class Config:
        orm_mode = True

