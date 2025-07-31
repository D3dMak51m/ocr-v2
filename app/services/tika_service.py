import os
import io
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import List

import requests
import fitz # PyMuPDF
from PIL import Image
from requests.exceptions import RequestException

from config import settings
from core import file_types
from core.exceptions import ExternalServiceError
from core.schemas import DocOcrResult, ImageOcrResult
from services import image_service, utils

# Initialize logger
logger = logging.getLogger(__name__)


def _tika_get_text_content(filepath: str) -> str:
    """Extracts plain text content from a document using Apache Tika."""
    tika_url = f"{settings.TIKA_SERVER_URL}/tika"
    logger.info(f"Sending file to Tika for text extraction: {tika_url}")
    try:
        with open(filepath, "rb") as f:
            response = requests.put(
                tika_url,
                data=f,
                headers={"Accept": "text/plain", "Content-Type":
                         "application/octet-stream"},
                timeout=120  # 2-minute timeout for large files
            )
            response.raise_for_status()
            return response.text
    except RequestException as e:
        msg = f"Tika server request failed: {e}"
        logger.error(msg)
        raise ExternalServiceError("Tika", msg)
    except Exception as e:
        msg = f"An unexpected error occurred during Tika text extraction: {e}"
        logger.error(msg)
        raise ExternalServiceError("Tika", msg)


def _tika_extract_embedded_files(filepath: str) -> List[ImageOcrResult]:
    """Unpacks embedded files (like images) from a document using Tika."""
    tika_url = f"{settings.TIKA_SERVER_URL}/unpack"
    logger.info(f"Sending file to Tika for unpacking embedded files: \
                {tika_url}")
    
    ocr_results = []
    try:
        with open(filepath, "rb") as f:
            response = requests.put(
                tika_url, data=f, 
                headers={"Accept": "application/zip"}, timeout=120
            )
            response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmpzip:
            tmpzip.write(response.content)
            tmpzip.seek(0)

            with zipfile.ZipFile(tmpzip, "r") as zip_ref:
                for item_name in zip_ref.namelist():
                    # Skip metadata and other non-image files
                    if item_name.startswith('__'):
                        continue
                    try:
                        file_bytes = zip_ref.read(item_name)
                        # Process image directly from bytes
                        result = image_service.process_image_from_bytes(
                            file_bytes
                            )
                        result.filename = os.path.basename(item_name)
                        ocr_results.append(result)
                    except Exception as e:
                        logger.warning(f"Could not process embedded file \
                                       '{item_name}': {e}")
    except RequestException as e:
        msg = f"Tika server request failed during unpacking: {e}"
        logger.error(msg)
        raise ExternalServiceError("Tika", msg)

    return ocr_results


def _pdf_extract_images(pdf_path: str) -> List[ImageOcrResult]:
    """Extracts images from each page of a PDF and runs OCR on them."""
    ocr_results = []
    logger.info(f"Extracting images directly from PDF: {pdf_path}")
    try:
        with fitz.open(pdf_path) as pdf_file:
            for page_num in range(len(pdf_file)):
                image_list = pdf_file.get_page_images(page_num, full=True)
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = pdf_file.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    try:
                        # More efficient: process image directly from bytes
                        result = image_service.\
                            process_image_from_bytes(image_bytes)
                        result.filename = f"page_{page_num + 1}_img_\
                            {img_index + 1}.{image_ext}"
                        ocr_results.append(result)
                    except Exception as e:
                        logger.warning(f"Could not process image on page \
                                       {page_num+1}: {e}")
    except Exception as e:
        logger.error(f"Failed to process PDF file for image extraction: {e}")
        # Don't raise, as we might still have text from Tika
    return ocr_results


def process_document_with_tika(filepath: str, file_type: str) -> DocOcrResult:
    """
    Processes a document using Tika for text and embedded images.
    """
    # 1. Get main text content from Tika
    raw_text = _tika_get_text_content(filepath)
    formatted_text = utils.text_formatting(raw_text)

    # 2. Extract and OCR images based on file type
    image_results = []
    if file_type == file_types.TYPE_PDF:
        image_results = _pdf_extract_images(filepath)
    else:
        # For DOCX, PPTX, etc., use Tika's unpack feature
        image_results = _tika_extract_embedded_files(filepath)

    return DocOcrResult(text=formatted_text,
                        images=image_results,
                        service="tika")
