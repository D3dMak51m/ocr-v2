import logging
import re
import io
from PIL import Image
import pytesseract
import chardet

from core.schemas import ImageOcrResult
from services.utils import language_detector, text_formatting, enhance_ocr_image_v1
import uuid, json
import requests
import io

# Initialize logger
logger = logging.getLogger(__name__)


def _ensure_utf8(text: str) -> str:
    """
    Detects and converts text to UTF-8 if it's not already.
    This can help clean up garbled text from OCR engines.
    """
    try:
        detected = chardet.detect(text.encode())
        encoding = detected.get("encoding")
        if encoding and encoding.lower() != "utf-8":
            logger.warning(f"Detected non-UTF-8 encoding '{encoding}',\
                           attempting to convert.")
            return text.encode(encoding, errors="ignore").decode("utf-8")
    except Exception as e:
        logger.error(f"Error during UTF-8 conversion: {e}")
    return text


def _detect_image_orientation(pil_image: Image.Image) -> tuple[str, float]:
    """
    Detects script orientation and confidence using Tesseract's OSD.
    """
    try:
        osd = pytesseract.image_to_osd(
            pil_image, config="-c min_characters_to_try=5"
        )
        script = re.search(r"Script: (\w+)", osd)
        conf = re.search(r"Script confidence: (\d+\.?\d*)", osd)

        script_name = script.group(1) if script else "Unknown"
        confidence = float(conf.group(1)) if conf else 0.0

        return script_name, confidence
    except pytesseract.TesseractError as e:
        logger.warning(f"Could not perform OSD (orientation detection): {e}")
        return "N/A", 0.0
    except Exception as e:
        logger.error(f"An unexpected error occurred during OSD: {e}")
        return "N/A", 0.0


def process_image_from_pil(pil_image: Image.Image) -> ImageOcrResult:
    """
    Performs OCR on a PIL.Image object and returns a structured result.
    This is the core image processing function.
    """
    # 1. Detect image orientation and script
    encoding, conf = _detect_image_orientation(pil_image)
    logger.info(f"Image script detection: {encoding} (Confidence: {conf:.2f})")

    # 2. Extract raw text using Tesseract
    try:
        raw_text = pytesseract.image_to_string(
            image=pil_image, lang="uzb_cyrl+uzb+en",
            config="-c min_characters_to_try=5"
        )
        # raw_text = "dummy text for testing"  # Placeholder for actual OCR result
    except pytesseract.TesseractError as e:
        logger.error(f"Tesseract failed to process the image: {e}")
        # Return a result indicating failure at the OCR step
        return ImageOcrResult(text=f"OCR failed: {e}")

    # 3. Clean and format the text
    cleaned_text = text_formatting(raw_text)
    final_text = _ensure_utf8(cleaned_text)

    # 4. Detect language of the extracted text
    lang_info = language_detector(final_text) if final_text else {}
    language = lang_info.get("language")
    score = lang_info.get("score")

    # 5. Assemble the final result object
    result = ImageOcrResult(
        text=final_text,
        language=language,
        language_score=float(score) if score is not None else None,
        encoding=encoding,
        encoding_conf=float(conf) if conf is not None else None,
    )
    return result


def process_image_from_path(image_path: str) -> ImageOcrResult:
    """
    Opens an image from a file path and processes it.
    """
    logger.info(f"Processing image from path: {image_path}")

    url = " http://iron_ocr:8080/ocr"
    params = {
        "lang": "uzbek,uzbek-cyrillic"
    }



    try:
        with Image.open(image_path) as pil_img:
            # enhance the image
            enhanced_image = enhance_ocr_image_v1(pil_img, scale_factor=1.9)
            # # cv2.imwrite(
            img_buffer = io.BytesIO()
            format_ = image_path.split('.')[-1].upper()
            if format_ == "JPG":
                format_ = "JPEG"
            enhanced_image.save(img_buffer, format=format_)
            img_buffer.seek(0)

            files = {"file": img_buffer}

            
            # call iron ocr 
            resp = process_image_from_pil(enhanced_image)
            print("confidence: ", resp.encoding_conf)
            if resp.encoding == "Cyrillic" and resp.encoding_conf > 30:
                params["lang"] = "uzbek-cyrillic"

            response = requests.post(url, params=params, files=files)
            # with open(image_path, "rb") as f:
            #     files = {"file": f}
            #     response = requests.post(url, params=params, files=files)

            parsed = json.loads(response.text)
            print(resp.text)

            resp.text = parsed["text"]
            return resp
    except FileNotFoundError:
        logger.error(f"Image file not found at path: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to open or process image at {image_path}: {e}")
        raise


def process_image_from_bytes(image_bytes: bytes) -> ImageOcrResult:
    """
    Processes an image directly from a byte stream.
    """
    logger.info("Processing image from byte stream.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            return process_image_from_pil(pil_img)
    except Exception as e:
        logger.error(f"Failed to process image from bytes: {e}")
        raise
