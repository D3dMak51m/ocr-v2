import asyncio
import os
import tempfile
import logging
from contextlib import contextmanager, asynccontextmanager

import httpx
import requests
from requests.exceptions import RequestException

from core.schemas import OcrRequest, DocOcrResult
from core.exceptions import FileProcessingError, UnsupportedFileTypeError
import core.file_types as file_types
from services import image_service, tika_service, utils
from typing import Union


logger = logging.getLogger(__name__)


@asynccontextmanager
async def temporary_file_from_url(url: str):
    """Асинхронное скачивание файла."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    tmp_path = None
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream('GET', url, headers=headers, timeout=60.0) as r:
                r.raise_for_status()
                extension = utils.get_file_extension(url)

                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp_file:
                    async for chunk in r.aiter_bytes(chunk_size=8192):
                        tmp_file.write(chunk)
                    tmp_path = tmp_file.name

        logger.info(f"Successfully downloaded file from {url} to {tmp_path}")
        yield tmp_path
    except Exception as e:
        raise FileProcessingError(f"Failed to download file from URL: {url}. Reason: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info(f"Cleaned up temporary file: {tmp_path}")


async def process_file_path(filepath: str) -> DocOcrResult:
    """Асинхронный роутер файлов."""
    extension = utils.get_file_extension(filepath)
    file_type = file_types.get_file_type(extension)

    logger.info(f"Processing file '{os.path.basename(filepath)}' with type: {file_type}")

    if file_type is None:
        raise UnsupportedFileTypeError(f"File extension '{extension}' is not supported.")

    if file_type == file_types.TYPE_IMG:
        # Вызываем синхронный OCR в отдельном потоке (чтобы не заблокировать сервер)
        image_ocr_result = await asyncio.to_thread(image_service.process_image_from_path, filepath)

        # Проверяем Early Stopping
        CRITICAL_STAMPS = ["secret", "xdfu", "dlp", "дсп", "секрет"]
        early_stop = any(s.label in CRITICAL_STAMPS for s in image_ocr_result.stamps)

        return DocOcrResult(
            text="",
            images=[image_ocr_result],
            service="paddle",
            early_stop_triggered=early_stop
        )

    if file_type in file_types.TIKA_FILE_TYPES:
        return await tika_service.process_document_with_tika(filepath, file_type)

    raise UnsupportedFileTypeError(f"No processor available for file type '{file_type}'.")


async def run_ocr(request: OcrRequest) -> DocOcrResult:
    """Главная точка входа (вызывается из endpoints.py)."""
    if request.url:
        async with temporary_file_from_url(request.url) as temp_path:
            return await process_file_path(temp_path)
    elif request.local_path:
        if not os.path.exists(request.local_path):
            raise FileProcessingError(f"Local file path does not exist: {request.local_path}")
        return await process_file_path(request.local_path)
    else:
        raise ValueError("OCR request must contain either a 'url' or a 'local_path'.")
