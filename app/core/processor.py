import os
import tempfile
import logging
from contextlib import contextmanager

import requests
from requests.exceptions import RequestException

from core.schemas import OcrRequest, DocOcrResult, ImageOcrResult
from core.exceptions import FileProcessingError, UnsupportedFileTypeError
import core.file_types as file_types
from services import image_service, tika_service, utils

from typing import Union


logger = logging.getLogger(__name__)


@contextmanager
def temporary_file_from_url(url: str):
    """Context manager to download a file from a URL to a temporary local path."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            extension = utils.get_file_extension(url)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{extension}"
            ) as tmp_file:
                for chunk in r.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name

        logger.info(f"Successfully downloaded file from {url} to {tmp_path}")
        yield tmp_path
    except RequestException as e:
        raise FileProcessingError(
            f"Failed to download file from URL: {url}. Reason: {e}"
        )
    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info(f"Cleaned up temporary file: {tmp_path}")


def process_file_path(filepath: str) -> Union[DocOcrResult, ImageOcrResult]:
    """
    Main dispatcher to process a file based on its extension.
    """
    extension = utils.get_file_extension(filepath)
    file_type = file_types.get_file_type(extension)

    logger.info(
        f"Processing file '{os.path.basename(filepath)}' with type: {file_type}"
    )

    if file_type is None:
        raise UnsupportedFileTypeError(
            f"File extension '{extension}' is not supported."
        )

    if file_type == file_types.TYPE_IMG:
        return image_service.process_image_from_path(filepath)

    if file_type in file_types.TIKA_FILE_TYPES:
        return tika_service.process_document_with_tika(filepath, file_type)

    # Fallback for any unhandled but recognized type
    raise UnsupportedFileTypeError(
        f"No processor available for file type '{file_type}'."
    )


def run_ocr(request: OcrRequest) -> Union[DocOcrResult, ImageOcrResult]:
    """
    Entry point for running OCR on a request, either from a URL or a local path.
    """
    if request.url:
        with temporary_file_from_url(request.url) as temp_path:
            return process_file_path(temp_path)
    elif request.local_path:
        if not os.path.exists(request.local_path):
            raise FileProcessingError(
                f"Local file path does not exist: {request.local_path}"
            )
        return process_file_path(request.local_path)
    else:
        raise ValueError("OCR request must contain either a 'url' or a 'local_path'.")
