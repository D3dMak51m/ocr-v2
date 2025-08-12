import io
import logging
import os
import re
import time
from typing import Optional, Tuple

import chardet
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

from core.schemas import ImageOcrResult
from services.utils import (
    language_detector,
    text_formatting,
    prepare_image_for_ocr,  # fast pipeline (deskew + enhance + psm)
    enhance_ocr_image_v1,  # keep for debugging if needed
)

# Initialize logger
logger = logging.getLogger(__name__)

# Toggle timing logs (default: on; share env var with utils)
# LOG_TIMINGS = os.getenv("OCR_LOG_TIMINGS", "1") not in ("0", "false", "False", "")
LOG_TIMINGS = False


def _t() -> float:
    return time.perf_counter()


def _log_ms(start: float, label: str) -> None:
    if LOG_TIMINGS:
        logger.info("[TIME] %s: %.1f ms", label, (time.perf_counter() - start) * 1000.0)


# Thresholds based on Tesseract docs: >=15.0 is "reasonably confident"
ORIENT_CONF_MIN = 1.0  # orientation_conf >= 15.0 ⇒ trust rotate
SCRIPT_CONF_STRONG = 15.0  # script_conf      >= 15.0 ⇒ trust single script

# Default language preferences
DEFAULT_LATIN_LANG = (
    "uzb"  # change to "eng" if you expect mostly English when script is Latin
)
CYRILLIC_LANG = "uzb_cyrl"
MIXED_LANGS = "uzb_cyrl+uzb+eng"  # mixed fallback


def _ensure_utf8(text: str) -> str:
    """
    Detects and converts text to UTF-8 if it's not already.
    This can help clean up garbled text from OCR engines.
    """
    t0 = _t()
    try:
        detected = chardet.detect(text.encode())
        encoding = detected.get("encoding")
        if encoding and encoding.lower() != "utf-8":
            logger.warning(
                f"Detected non-UTF-8 encoding '{encoding}', attempting to convert."
            )
            text = text.encode(encoding, errors="ignore").decode("utf-8")
    except Exception as e:
        logger.error(f"Error during UTF-8 conversion: {e}")
    _log_ms(t0, "ensure_utf8")
    return text


def _run_osd(pil_image: Image.Image) -> Optional[dict]:
    """
    Run Tesseract OSD (Orientation & Script Detection) and return a dict:
      {'page_num', 'orientation', 'rotate', 'orientation_conf', 'script', 'script_conf'}
    Uses PSM 0 per best practices.
    """
    t0 = _t()
    try:
        osd = pytesseract.image_to_osd(
            pil_image,
            output_type=Output.DICT,
            config="--psm 0 -c min_characters_to_try=5",
        )
        _log_ms(t0, "tesseract.image_to_osd")
        return osd
    except pytesseract.TesseractError as e:
        _log_ms(t0, "tesseract.image_to_osd(error)")
        logger.warning(f"OSD failed: {e}")
        return None
    except Exception as e:
        _log_ms(t0, "tesseract.image_to_osd(error)")
        logger.error(f"Unexpected OSD error: {e}")
        return None


def _apply_osd_rotation_if_confident(
    pil_image: Image.Image, osd: Optional[dict]
) -> Tuple[Image.Image, Optional[int], Optional[float], Optional[str], Optional[float]]:
    """
    If OSD is confident, rotate the image by 'rotate' degrees to correct orientation.
    Returns (possibly rotated image, rotate_degrees, orientation_conf, script, script_conf).
    """
    t0 = _t()
    if not osd:
        _log_ms(t0, "osd.apply_rotation.skip(no_osd)")
        return pil_image, None, None, None, None

    rotate = int(osd.get("rotate", 0))
    orient_conf = float(osd.get("orientation_conf", 0.0))
    script = osd.get("script")
    script_conf = float(osd.get("script_conf", 0.0))

    if rotate in (90, 180, 270) and orient_conf >= ORIENT_CONF_MIN:
        t1 = _t()
        pil_image = pil_image.rotate(rotate, expand=True)
        _log_ms(t1, f"osd.rotate({rotate}deg)")
        logger.info(
            "OSD rotation applied: %d° (conf %.2f), script=%s (%.2f)",
            rotate,
            orient_conf,
            script,
            script_conf,
        )
        _log_ms(t0, "osd.apply_rotation.total(applied)")
        return pil_image, rotate, orient_conf, script, script_conf

    logger.info(
        "OSD rotation skipped: rotate=%s, orient_conf=%.2f, script=%s (%.2f)",
        rotate,
        orient_conf,
        script,
        script_conf,
    )
    _log_ms(t0, "osd.apply_rotation.total(skipped)")
    return pil_image, rotate, orient_conf, script, script_conf


def _select_langs(script: Optional[str], script_conf: Optional[float]) -> str:
    """
    Choose Tesseract 'lang' based on OSD script info.
    - Strong Latin -> DEFAULT_LATIN_LANG (single)
    - Strong Cyrillic -> CYRILLIC_LANG (single)
    - Otherwise -> MIXED_LANGS
    """
    t0 = _t()
    if script is None or script_conf is None:
        _log_ms(t0, "select_langs(mixed_default)")
        return MIXED_LANGS

    sc = script_conf or 0.0
    name = (script or "").lower()

    if sc >= SCRIPT_CONF_STRONG:
        if "latin" in name:
            _log_ms(t0, "select_langs(latin)")
            return DEFAULT_LATIN_LANG
        if "cyrillic" in name:
            _log_ms(t0, "select_langs(cyrillic)")
            return CYRILLIC_LANG

    _log_ms(t0, "select_langs(mixed_lowconf)")
    return MIXED_LANGS


def _ocr_once(pil_image: Image.Image, lang: str, psm: int, timeout_s: int = 10) -> str:
    """
    Single Tesseract pass with tuned defaults for speed/quality.
    """
    config = (
        f"--oem 1 --psm {psm} "
        f"-c user_defined_dpi=300 "
        f"-c preserve_interword_spaces=1 "
        f"-c min_characters_to_try=5"
    )
    t0 = _t()
    text = pytesseract.image_to_string(
        image=pil_image, lang=lang, config=config, timeout=timeout_s
    )
    # text = pytesseract.image_to_string(
    #     image=pil_image, lang="uzb", config=config, timeout=timeout_s
    # )

    _log_ms(t0, f"tesseract.image_to_string(lang={lang},psm={psm})")
    return text


def _text_strength(t: str) -> int:
    """
    A quick proxy for OCR quality: count of alphanumeric characters.
    """
    return len(re.findall(r"\w", t))


def process_image_from_pil(pil_image: Image.Image) -> ImageOcrResult:
    """
    Performs OCR on a PIL.Image with OSD-based rotation and script-aware language selection,
    plus a fast deskew/enhance pipeline. Includes detailed timing logs.
    """
    t_total = _t()

    # 0) EXIF orientation fix
    t0 = _t()
    img = ImageOps.exif_transpose(pil_image)
    _log_ms(t0, "process.exif_transpose")

    # 1) OSD: orientation + script
    t1 = _t()
    osd = _run_osd(img)
    _log_ms(t1, "process.osd_total")

    # 1a) Apply rotation if confident
    t2 = _t()
    img, rotate_deg, orient_conf, script, script_conf = (
        _apply_osd_rotation_if_confident(img, osd)
    )
    _log_ms(t2, "process.osd_apply")

    # 2) Prepare (deskew small angles, conditional enhance) + dynamic PSM
    t3 = _t()
    enhanced, psm, angle, metrics = prepare_image_for_ocr(img)
    _log_ms(t3, "process.prepare_total")
    logger.info(
        "Prep: deskew=%.2f°, blur=%.1f, contrast=%.1f; OSD rotate=%s, orient_conf=%s, script=%s(%.2f)",
        angle,
        metrics["blur"],
        metrics["contrast"],
        str(rotate_deg),
        f"{orient_conf:.2f}" if orient_conf is not None else "N/A",
        script or "N/A",
        script_conf or 0.0,
    )

    # 3) Choose languages from script
    # t4 = _t()
    # lang = _select_langs(script, script_conf)
    # _log_ms(t4, f"process.select_langs({lang})")

    # 4) OCR (first pass)
    t5 = _t()
    try:
        raw_text = _ocr_once(enhanced, lang=MIXED_LANGS, psm=psm)
    except pytesseract.TesseractError as e:
        _log_ms(t5, "process.ocr_firstpass(error)")
        logger.error(f"Tesseract failed to process the image: {e}")
        return ImageOcrResult(text=f"OCR failed: {e}")
    _log_ms(t5, "process.ocr_firstpass")

    # # 5) If result looks too weak and we trusted a single-script, try mixed-langs fallback once
    # if _text_strength(raw_text) < 40 and lang != MIXED_LANGS:
    #     t6 = _t()
    #     try:
    #         logger.info(
    #             "Weak OCR result with single-script; retrying with mixed languages."
    #         )
    #         raw_text = _ocr_once(enhanced, lang=MIXED_LANGS, psm=psm)
    #     except pytesseract.TesseractError:
    #         pass
    #     _log_ms(t6, "process.ocr_fallback_mixed")

    # 6) Clean and format the text
    t7 = _t()
    cleaned_text = text_formatting(raw_text)
    _log_ms(t7, "process.text_formatting")

    t8 = _t()
    final_text = _ensure_utf8(cleaned_text)
    _log_ms(t8, "process.ensure_utf8")

    # 7) Detect language of the extracted text (informational)
    t9 = _t()
    lang_info = language_detector(final_text) if final_text else {}
    language = lang_info.get("language")
    score = lang_info.get("score")
    _log_ms(t9, "process.language_detector")

    # 8) Assemble the final result object
    result = ImageOcrResult(
        text=final_text,
        language=language,
        language_score=float(score) if score is not None else None,
        encoding=script or "N/A",  # reuse field to surface script info
        encoding_conf=float(script_conf) if script_conf is not None else None,
    )

    _log_ms(t_total, "process.total")
    return result


def process_image_from_path(image_path: str) -> ImageOcrResult:
    """
    Opens an image from a file path and processes it.
    """
    logger.info(f"Processing image from path: {image_path}")
    t0 = _t()
    try:
        with Image.open(image_path) as pil_img:
            _log_ms(t0, "open_image.from_path")
            return process_image_from_pil(pil_img)
    except FileNotFoundError:
        _log_ms(t0, "open_image.from_path(error)")
        logger.error(f"Image file not found at path: {image_path}")
        raise
    except Exception as e:
        _log_ms(t0, "open_image.from_path(error)")
        logger.error(f"Failed to open or process image at {image_path}: {e}")
        raise


def process_image_from_bytes(image_bytes: bytes) -> ImageOcrResult:
    """
    Processes an image directly from a byte stream.
    """
    logger.info("Processing image from byte stream.")
    t0 = _t()
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            _log_ms(t0, "open_image.from_bytes")
            return process_image_from_pil(pil_img)
    except Exception as e:
        _log_ms(t0, "open_image.from_bytes(error)")
        logger.error(f"Failed to process image from bytes: {e}")
        raise
