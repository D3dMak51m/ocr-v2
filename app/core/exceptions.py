class OcrBaseException(Exception):
    """Base exception for the OCR service."""

    def __init__(self, message: str, code: str = "OCR_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class FileProcessingError(OcrBaseException):
    """Error during file processing."""

    def __init__(self, message: str, code: str = "FILE_PROCESSING_ERROR"):
        super().__init__(message, code)


class UnsupportedFileTypeError(OcrBaseException):
    """Error for unsupported file types."""

    def __init__(self, message: str, code: str = "UNSUPPORTED_FILE_TYPE"):
        super().__init__(message, code)


class ExternalServiceError(OcrBaseException):
    """Error related to an external service like Airflow or Tika."""

    def __init__(
        self, service_name: str, message: str,
        code: str = "EXTERNAL_SERVICE_ERROR"
    ):
        full_message = f"Error with {service_name}: {message}"
        super().__init__(full_message, code)
