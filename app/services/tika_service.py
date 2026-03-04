import asyncio
import os
import io
import logging
import tempfile
import zipfile
from typing import List, Tuple
from zipfile import ZipFile
import httpx

import requests
import fitz # PyMuPDF
from PIL import Image
from requests.exceptions import RequestException

from config import settings
from core import file_types
from core.exceptions import ExternalServiceError
from core.schemas import DocOcrResult, ImageOcrResult
from services import image_service, utils
from services.stamp_detector import stamp_detector

# Initialize logger
logger = logging.getLogger(__name__)

async def _async_process_pdf_hybrid(filepath: str) -> Tuple[str, List[ImageOcrResult], bool]:
    """
    Умный парсинг PDF: читает блоки страницы сверху вниз.
    Нативный текст копирует, встроенные картинки отправляет в OCR и склеивает всё по порядку.
    """
    full_text_parts = []
    all_images_results = []
    early_stop_triggered = False

    # Классы, которые мы ловим нашей YOLO-моделью
    CRITICAL_STAMPS = ["stamp", "signature"]

    logger.info(f"Starting hybrid parsing for PDF: {filepath}")

    try:
        # Открываем PDF
        with fitz.open(filepath) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Получаем словарь структуры страницы (текст + картинки)
                page_dict = page.get_text("dict")
                blocks = page_dict.get("blocks", [])

                # Сортируем блоки по вертикали (сверху вниз)
                blocks.sort(key=lambda b: b["bbox"][1])

                for block in blocks:
                    # ТИП 0: Цифровой текст
                    if block["type"] == 0:
                        text_content = ""
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text_content += span.get("text", "") + " "
                            text_content += "\n"

                        cleaned_text = text_content.strip()
                        if cleaned_text:
                            full_text_parts.append(cleaned_text)

                    # ТИП 1: Встроенное изображение (или целиком отсканированная страница)
                    elif block["type"] == 1:
                        image_bytes = block.get("image")
                        if not image_bytes:
                            continue

                        # Преобразуем байты в PIL Image
                        img = Image.open(io.BytesIO(image_bytes))
                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        # 1. Проверяем картинку на штампы (Early Stop)
                        stamps = stamp_detector.detect(img)
                        for stamp in stamps:
                            if stamp.label in CRITICAL_STAMPS and stamp.confidence > 0.60:
                                logger.warning(f"CRITICAL STAMP '{stamp.label}' FOUND. EARLY STOPPING.")
                                early_stop_triggered = False # TODO: change to True after training model for finding dlp stamps
                                break

                        # 2. Сканируем картинку через PaddleOCR (в фоне, чтобы не блокировать сервер)
                        result = await asyncio.to_thread(image_service.process_image_from_pil, img)
                        result.filename = f"page_{page_num + 1}_img.png"
                        all_images_results.append(result)

                        # 3. Вставляем распознанный текст в общий поток ровно там, где была картинка!
                        if result.text:
                            full_text_parts.append(f"\n[Text from Image]\n{result.text}\n")

                        if early_stop_triggered:
                            break

                if early_stop_triggered:
                    break

        # Склеиваем всё в один последовательный документ
        combined_text = "\n\n".join(full_text_parts)
        return combined_text, all_images_results, early_stop_triggered

    except Exception as e:
        logger.error(f"Hybrid PDF processing failed: {e}")
        return "", [], False


async def _async_tika_get_text(filepath: str) -> str:
    """Асинхронно извлекает текст через Tika."""
    tika_url = f"{settings.TIKA_SERVER_URL}/tika"
    logger.info(f"Sending file to Tika for text extraction: {tika_url}")
    try:
        with open(filepath, "rb") as f:
            file_data = f.read()

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.put(
                tika_url,
                content=file_data,
                headers={
                    "Accept": "text/plain; charset=utf-8",
                    "Content-Type": "application/octet-stream",
                    "Accept-Charset": "utf-8"
                }
            )
            response.raise_for_status()

            # Пытаемся обработать кодировку
            try:
                text = response.text
            except UnicodeDecodeError:
                import chardet
                detected = chardet.detect(response.content)
                encoding = detected.get('encoding', 'utf-8')
                text = response.content.decode(encoding, errors='replace')

            return text
    except Exception as e:
        logger.error(f"Tika extraction failed: {e}")
        return ""


async def _async_tika_extract_embedded_files(filepath: str) -> List[ImageOcrResult]:
    """Асинхронно распаковывает встроенные картинки через Tika."""
    tika_url = f"{settings.TIKA_SERVER_URL}/unpack"
    ocr_results = []

    try:
        with open(filepath, "rb") as f:
            file_data = f.read()

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.put(
                tika_url,
                content=file_data,
                headers={"Accept": "application/zip"}
            )

            if response.status_code == 204 or not response.content:
                return ocr_results
            if len(response.content) < 4 or response.content[:2] != b'PK':
                return ocr_results

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmpzip:
            tmpzip.write(response.content)
            tmpzip.seek(0)

            try:
                with zipfile.ZipFile(tmpzip, "r") as zip_ref:
                    for item_name in zip_ref.namelist():
                        if item_name.startswith('__'):
                            continue
                        try:
                            file_bytes = zip_ref.read(item_name)
                            # Вызываем синхронную функцию распознавания картинки
                            result = image_service.process_image_from_bytes(file_bytes)
                            result.filename = os.path.basename(item_name)
                            ocr_results.append(result)
                        except Exception as e:
                            logger.warning(f"Could not process embedded file '{item_name}': {e}")
            except zipfile.BadZipFile:
                pass

    except Exception as e:
        logger.error(f"Tika unpacking failed: {e}")

    return ocr_results



def _tika_get_text_content(filepath: str) -> str:
    """Extracts plain text content from a document using Apache Tika."""
    tika_url = f"{settings.TIKA_SERVER_URL}/tika"
    logger.info(f"Sending file to Tika for text extraction: {tika_url}")
    try:
        with open(filepath, "rb") as f:
            response = requests.put(
                tika_url,
                data=f,
                headers={
                    "Accept": "text/plain; charset=utf-8",
                    "Content-Type": "application/octet-stream",
                    "Accept-Charset": "utf-8"
                },
                timeout=120  # 2-minute timeout for large files
            )
            response.raise_for_status()
            
            # Ensure proper encoding
            response.encoding = 'utf-8'
            
            # Try to decode with UTF-8, fallback to other encodings if needed
            try:
                text = response.text
            except UnicodeDecodeError:
                # Try to detect encoding
                import chardet
                detected = chardet.detect(response.content)
                encoding = detected.get('encoding', 'utf-8')
                logger.info(f"Detected encoding: {encoding}")
                text = response.content.decode(encoding, errors='replace')
            
            return text
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
            
            # Handle 204 No Content response (no embedded files to extract)
            if response.status_code == 204:
                logger.info(f"No embedded files found in {filepath} (HTTP 204)")
                return ocr_results
            
            response.raise_for_status()
            
            # Check if response content is empty
            if not response.content:
                logger.info(f"Empty response content from Tika for {filepath}")
                return ocr_results
            
            # Check if response is actually a ZIP file
            # ZIP files start with 'PK' (0x504B) magic bytes
            if len(response.content) < 4 or response.content[:2] != b'PK':
                logger.info(f"Response from Tika is not a ZIP file for {filepath}")
                return ocr_results
        
        logger.info(
            f"Successfully unpacked embedded files from {filepath} using Tika the content {response.content[:100]}..."
        )
        
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmpzip:
            tmpzip.write(response.content)
            tmpzip.seek(0)

            try:
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
            except zipfile.BadZipFile as e:
                logger.warning(f"Invalid ZIP file returned from Tika for {filepath}: {e}")
                return ocr_results
                
    except RequestException as e:
        msg = f"Tika server request failed during unpacking: {e}"
        logger.error(msg)
        raise ExternalServiceError("Tika", msg)

    return ocr_results


def _pdf_extract_images(pdf_path: str) -> Tuple[List[ImageOcrResult], bool]:
    ocr_results = []
    early_stop_triggered = False
    CRITICAL_STAMPS = ["secret", "xdfu", "dlp", "дсп", "секрет"]

    logger.info(f"Extracting images from PDF: {pdf_path}")
    try:
        with fitz.open(pdf_path) as pdf_file:
            for page_num in range(len(pdf_file)):
                page = pdf_file[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Early check for security classification
                stamps = stamp_detector.detect(img)
                for stamp in stamps:
                    if stamp.label in CRITICAL_STAMPS and stamp.confidence > 0.80:
                        logger.warning(f"CRITICAL STAMP FOUND '{stamp.label}' on page {page_num + 1}. EARLY STOPPING.")
                        early_stop_triggered = False # TODO: change to True after training model for finding dlp stamps
                        break

                try:
                    result = image_service.process_image_from_pil(img)
                    result.filename = f"page_{page_num + 1}.png"
                    ocr_results.append(result)
                except Exception as e:
                    logger.warning(f"Could not process page {page_num + 1}: {e}")

                if early_stop_triggered:
                    break

    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")

    return ocr_results, early_stop_triggered


def _is_excel_file(filepath: str) -> bool:
    """Check if a file is an Excel file based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in ['.xls', '.xlsx', '.xlsm', '.xlsb']


def _has_encoding_issues(text: str) -> bool:
    """Check if text has potential encoding issues."""
    if not text:
        return False
    
    # Check for common signs of encoding problems
    # Unicode replacement character
    if '\ufffd' in text:
        return True
    
    # Check for excessive special characters that might indicate encoding issues
    special_char_count = sum(1 for c in text if 127 < ord(c) < 160)
    if len(text) > 100 and special_char_count / len(text) > 0.1:
        return True
    
    return False


def _extract_excel_text_with_encoding(filepath: str) -> str:
    """
    Extract text from Excel files with proper encoding support.
    Particularly useful for files with Cyrillic or other non-ASCII text.
    """
    try:
        # Try using openpyxl for XLSX files
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.xlsx', '.xlsm', '.xlsb']:
            try:
                from openpyxl import load_workbook
                
                wb = load_workbook(filepath, data_only=True, read_only=True)
                all_text = []
                
                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    all_text.append(f"Sheet: {sheet_name}\n")
                    
                    for row in sheet.iter_rows(values_only=True):
                        row_text = []
                        for cell in row:
                            if cell is not None:
                                # Ensure proper string conversion
                                cell_str = str(cell)
                                row_text.append(cell_str)
                        if row_text:
                            all_text.append('\t'.join(row_text))
                
                wb.close()
                return '\n'.join(all_text)
                
            except ImportError:
                logger.warning("openpyxl not installed, using fallback method")
            except Exception as e:
                logger.warning(f"Error using openpyxl: {e}")
        
        # For XLS files or as fallback, try xlrd
        if ext == '.xls':
            try:
                import xlrd
                
                # xlrd has good support for various encodings
                book = xlrd.open_workbook(filepath, encoding_override=None)
                all_text = []
                
                for sheet_idx in range(book.nsheets):
                    sheet = book.sheet_by_index(sheet_idx)
                    all_text.append(f"Sheet: {sheet.name}\n")
                    
                    for row_idx in range(sheet.nrows):
                        row_text = []
                        for col_idx in range(sheet.ncols):
                            cell = sheet.cell(row_idx, col_idx)
                            if cell.value:
                                # xlrd handles encoding well
                                cell_str = str(cell.value)
                                row_text.append(cell_str)
                        if row_text:
                            all_text.append('\t'.join(row_text))
                
                return '\n'.join(all_text)
                
            except ImportError:
                logger.warning("xlrd not installed for XLS processing")
            except Exception as e:
                logger.warning(f"Error using xlrd: {e}")
        
        # Last resort: try pandas if available
        try:
            import pandas as pd
            
            # Pandas has good encoding detection
            if ext == '.xls':
                df_dict = pd.read_excel(filepath, sheet_name=None, engine='xlrd')
            else:
                df_dict = pd.read_excel(filepath, sheet_name=None, engine='openpyxl')
            
            all_text = []
            for sheet_name, df in df_dict.items():
                all_text.append(f"Sheet: {sheet_name}\n")
                # Convert DataFrame to string with proper encoding
                all_text.append(df.to_string())
            
            return '\n'.join(all_text)
            
        except ImportError:
            logger.warning("pandas not installed for Excel processing")
        except Exception as e:
            logger.warning(f"Error using pandas: {e}")
    
    except Exception as e:
        logger.error(f"Failed to extract text with encoding support: {e}")
    
    return ""


def _extract_images_from_xlsx(filepath: str) -> List[ImageOcrResult]:
    """
    Extract images from XLSX files directly.
    XLSX files are actually ZIP archives with a specific structure.
    """
    ocr_results = []
    logger.info(f"Attempting to extract images from XLSX file: {filepath}")
    
    try:
        with ZipFile(filepath, 'r') as zip_file:
            # List all files in the ZIP
            all_files = zip_file.namelist()
            
            # Images in XLSX are typically stored in xl/media/ directory
            image_files = [f for f in all_files if f.startswith('xl/media/')]
            
            if not image_files:
                logger.info(f"No images found in XLSX file: {filepath}")
                return ocr_results
            
            logger.info(f"Found {len(image_files)} images in XLSX file")
            
            for image_file in image_files:
                try:
                    image_bytes = zip_file.read(image_file)
                    # Process image directly from bytes
                    result = image_service.process_image_from_bytes(image_bytes)
                    result.filename = os.path.basename(image_file)
                    ocr_results.append(result)
                    logger.info(f"Successfully processed image: {image_file}")
                except Exception as e:
                    logger.warning(f"Could not process image '{image_file}': {e}")
                    
    except zipfile.BadZipFile:
        logger.warning(f"File {filepath} is not a valid ZIP/XLSX file")
    except Exception as e:
        logger.error(f"Error extracting images from XLSX: {e}")
    
    return ocr_results


def _extract_images_from_xls(filepath: str) -> List[ImageOcrResult]:
    """
    Extract images from XLS files using a different approach.
    For older XLS files, we may need to use specialized libraries or Tika.
    """
    ocr_results = []
    logger.info(f"Attempting to extract images from XLS file: {filepath}")
    
    # First, try Tika's approach for XLS files
    # Tika can sometimes extract images from XLS files even if it returns 204 for XLSX
    ocr_results = _tika_extract_embedded_files(filepath)
    
    if ocr_results:
        logger.info(f"Successfully extracted {len(ocr_results)} images from XLS using Tika")
        return ocr_results
    
    # If Tika didn't work, we might need additional libraries like xlrd or python-excel
    # This would require additional dependencies
    logger.info("No images extracted from XLS file using available methods")
    
    # Alternative: Try to convert XLS to XLSX using Tika and then extract
    # This requires additional implementation
    
    return ocr_results


def _extract_images_from_excel(filepath: str) -> List[ImageOcrResult]:
    """
    Main function to extract images from Excel files.
    Handles both XLS and XLSX formats.
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.xlsx', '.xlsm', '.xlsb']:
        # Modern Excel format (ZIP-based)
        return _extract_images_from_xlsx(filepath)
    elif ext == '.xls':
        # Legacy Excel format
        return _extract_images_from_xls(filepath)
    else:
        logger.warning(f"Unknown Excel format: {ext}")
        return []


async def process_document_with_tika(filepath: str, file_type: str) -> DocOcrResult:
    """Главная асинхронная функция маршрутизации документа."""

    early_stop = False
    image_results = []
    formatted_text = ""

    # Если это PDF, используем наш новый умный гибридный парсер
    if file_type == file_types.TYPE_PDF:
        raw_text, image_results, early_stop = await _async_process_pdf_hybrid(filepath)
        formatted_text = utils.text_formatting(raw_text)

        return DocOcrResult(
            text=formatted_text,
            images=image_results,
            service="hybrid_fitz_paddle",
            early_stop_triggered=early_stop
        )

    # Для остальных форматов (Excel, Word, txt) используем Tika
    raw_text = await _async_tika_get_text(filepath)

    if len(raw_text.strip()) > 100 and not _is_excel_file(filepath):
        logger.info(f"Document is native (digital). Extracted {len(raw_text)} chars via Tika.")
    else:
        logger.info("Document seems to be a scanned copy or Excel. Extracting files.")
        if _is_excel_file(filepath):
            image_results = await asyncio.to_thread(_extract_images_from_xlsx, filepath)
            if not image_results:
                image_results = await _async_tika_extract_embedded_files(filepath)
        else:
            image_results = await _async_tika_extract_embedded_files(filepath)

    formatted_text = utils.text_formatting(raw_text)

    return DocOcrResult(
        text=formatted_text,
        images=image_results,
        service="tika_paddle",
        early_stop_triggered=early_stop
    )