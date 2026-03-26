import asyncio
import os
import tempfile
import logging
from contextlib import asynccontextmanager

import httpx

from config import settings
from core.schemas import OcrRequest, DocOcrResult
from core.exceptions import (
    FileProcessingError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
import core.file_types as file_types
from services import image_service, tika_service, utils


logger = logging.getLogger(__name__)
BYTES_IN_MB = 1024 * 1024


def _to_size_limit_bytes(max_file_size_mb: int | float | None) -> int | None:
    if max_file_size_mb is None:
        return None
    return int(max_file_size_mb * BYTES_IN_MB)


def _validate_local_file_size(filepath: str, max_file_size_mb: int | float | None) -> None:
    max_file_size_bytes = _to_size_limit_bytes(max_file_size_mb)
    if max_file_size_bytes is None:
        return

    actual_size_bytes = os.path.getsize(filepath)
    if actual_size_bytes > max_file_size_bytes:
        raise FileTooLargeError(
            f"Local file exceeds {max_file_size_mb} MB limit: "
            f"{actual_size_bytes / BYTES_IN_MB:.2f} MB"
        )



@asynccontextmanager
async def temporary_file_from_url(url: str, max_file_size_mb: int | float | None):
    """Асинхронное скачивание файла во временный файл."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    tmp_path = None
    max_file_size_bytes = _to_size_limit_bytes(max_file_size_mb)
    # 1. Скачиваем файл (ошибки скачивания — FileProcessingError)
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream('GET', url, headers=headers, timeout=60.0) as r:
                r.raise_for_status()
                content_length = r.headers.get("Content-Length")
                if content_length and max_file_size_bytes is not None:
                    try:
                        declared_size = int(content_length)
                        if declared_size > max_file_size_bytes:
                            raise FileTooLargeError(
                                f"Remote file exceeds {max_file_size_mb} MB limit: "
                                f"{declared_size / BYTES_IN_MB:.2f} MB"
                            )
                    except ValueError:
                        logger.warning("Invalid Content-Length header for %s: %s", url, content_length)

                extension = utils.get_file_extension(url)
                suffix = f".{extension}" if extension else ""
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_path = tmp_file.name
                    downloaded_size = 0
                    async for chunk in r.aiter_bytes(chunk_size=8192):
                        downloaded_size += len(chunk)
                        if max_file_size_bytes is not None and downloaded_size > max_file_size_bytes:
                            raise FileTooLargeError(
                                f"Remote file exceeds {max_file_size_mb} MB limit during download: "
                                f"{downloaded_size / BYTES_IN_MB:.2f} MB"
                            )
                        tmp_file.write(chunk)
        logger.info(f"Successfully downloaded file from {url} to {tmp_path}")
    except FileTooLargeError:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    except Exception as e:
        # Очищаем если частично скачали
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise FileProcessingError(f"Failed to download file from URL: {url}. Reason: {e}")

    # 2. Отдаём файл caller'у — его ошибки НЕ оборачиваем
    try:
        yield tmp_path
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info(f"Cleaned up temporary file: {tmp_path}")


async def process_file_path(filepath: str) -> DocOcrResult:
    """Асинхронный роутер файлов."""
    extension = utils.get_file_extension(filepath)
    file_type = file_types.get_file_type(extension)

    logger.info(f"Processing file '{os.path.basename(filepath)}' with type: {file_type}")

    if file_type == file_types.TYPE_IMG:
        # Вызываем синхронный OCR в отдельном потоке (чтобы не заблокировать сервер)
        image_ocr_result = await asyncio.to_thread(image_service.process_image_from_path, filepath)

        return DocOcrResult(
            text="",
            images=[image_ocr_result],
            service="paddle",
        )

    if file_type in file_types.TIKA_FILE_TYPES:
        return await tika_service.process_document_with_tika(filepath, file_type)

    raise UnsupportedFileTypeError(f"No processor available for file type '{file_type}'.")


async def run_ocr(
    request: OcrRequest,
    max_file_size_mb: int | float | None = settings.MAX_SYNC_FILE_SIZE_MB,
) -> DocOcrResult:
    """Главная точка входа (вызывается из endpoints.py)."""
    if request.url:
        async with temporary_file_from_url(request.url, max_file_size_mb) as temp_path:
            return await process_file_path(temp_path)
    elif request.local_path:
        if not os.path.exists(request.local_path):
            raise FileProcessingError(f"Local file path does not exist: {request.local_path}")
        _validate_local_file_size(request.local_path, max_file_size_mb)
        return await process_file_path(request.local_path)
    else:
        raise FileProcessingError("OCR request must contain either a 'url' or a 'local_path'.")
