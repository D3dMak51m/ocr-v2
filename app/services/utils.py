import logging
import os
import sys
import re
import time
from typing import Union, Tuple, Dict
from urllib.parse import urlparse

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from bs4 import BeautifulSoup

# Allow importing local detector if needed
sys.path.insert(0, "..")

logger = logging.getLogger(__name__)

# Toggle timing logs (default: on)
LOG_TIMINGS = os.getenv("OCR_LOG_TIMINGS", "1") not in ("0", "false", "False", "")

def _t() -> float:
    return time.perf_counter()

def _log_ms(start: float, label: str) -> None:
    if LOG_TIMINGS:
        logger.info("[TIME] %s: %.1f ms", label, (time.perf_counter() - start) * 1000.0)

files_tmp = "temp_files/"

image_extensions = ["jpeg/jpg", "jpeg", "jpg", "png", "gif", "bmp"]


def text_formatting(text: str) -> str:
    """
    Simple whitespace normalization for OCR text.
    """
    t0 = _t()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r" +", " ", text)
    _log_ms(t0, "text_formatting")
    return text


def get_file_extension(url: str) -> str:
    """
    Extract a lowercase extension from a URL or path.
    """
    t0 = _t()
    parsed_url = urlparse(url)
    ext = os.path.splitext(parsed_url.path)[1].lower()
    if ext.startswith("."):
        ext = ext[1:]
    _log_ms(t0, "get_file_extension")
    return ext


def xml_to_txt(xml_table: str) -> str:
    """
    Convert HTML/XML to plain text, then normalize whitespace.
    """
    t0 = _t()
    soup = BeautifulSoup(xml_table, "html.parser")
    text = text_formatting(soup.get_text())
    _log_ms(t0, "xml_to_txt")
    return text


def enhance_ocr_image_v1(
    image_input: Image.Image,
    scale_factor: float = 2.0,
    save_debug: bool = False,
    debug_prefix: str = "debug"
) -> Image.Image:
    """
    Enhance a PIL image for OCR: grayscale, denoise, threshold, sharpen, upscale.

    Returns:
        PIL.Image.Image: Enhanced image as a PIL image in mode 'L'.
    """
    t_total = _t()
    if not isinstance(image_input, Image.Image):
        raise TypeError("image_input must be a PIL.Image.Image object.")

    t0 = _t()
    img = np.array(image_input.convert("RGB"))[:, :, ::-1]  # RGB to BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _log_ms(t0, "enhance_v1.rgb2gray")

    t1 = _t()
    denoised = cv2.fastNlMeansDenoising(gray, h=25)
    _log_ms(t1, "enhance_v1.denoise")

    t2 = _t()
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    _log_ms(t2, "enhance_v1.adaptive_threshold")

    t3 = _t()
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    _log_ms(t3, "enhance_v1.sharpen")

    t4 = _t()
    resized = cv2.resize(
        sharpened, None,
        fx=scale_factor, fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )
    _log_ms(t4, "enhance_v1.resize")

    if save_debug:
        t5 = _t()
        cv2.imwrite(f"{debug_prefix}_gray.jpg", gray)
        cv2.imwrite(f"{debug_prefix}_denoised.jpg", denoised)
        cv2.imwrite(f"{debug_prefix}_thresh.jpg", thresh)
        cv2.imwrite(f"{debug_prefix}_sharpened.jpg", sharpened)
        cv2.imwrite(f"{debug_prefix}_resized.jpg", resized)
        _log_ms(t5, "enhance_v1.debug_save")

    _log_ms(t_total, "enhance_v1.total")
    return Image.fromarray(resized)


# ------------------- Faster, conditional pipeline ------------------- #

def _quality_metrics(gray: np.ndarray) -> Dict[str, float]:
    """
    Compute quick metrics to decide how much enhancement is needed.
    """
    t0 = _t()
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean, std = cv2.meanStdDev(gray)
    metrics = {
        "blur": float(blur),
        "brightness": float(mean[0][0]),
        "contrast": float(std[0][0]),
    }
    _log_ms(t0, "quality_metrics")
    return metrics


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate skew angle (degrees) with Hough on edges; robust and quick.
    Positive angle means image should be rotated counter-clockwise by that amount.
    """
    t0 = _t()
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if np.mean(thr) > 127:
        thr = 255 - thr
    edges = cv2.Canny(thr, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 200)
    angle = 0.0
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            a = np.degrees(theta) - 90.0
            if -45 < a < 45:
                angles.append(a)
        if angles:
            angle = float(np.median(angles))
    _log_ms(t0, "estimate_skew_angle")
    return angle


def _deskew_pil(image: Image.Image) -> Tuple[Image.Image, float]:
    """
    Deskew using Hough-based angle estimation.
    """
    t0 = _t()
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    t1 = _t()
    angle = _estimate_skew_angle(gray)
    _log_ms(t1, "deskew.estimate_angle")

    if abs(angle) < 0.5:
        _log_ms(t0, "deskew.total(noop)")
        return image, 0.0

    t2 = _t()
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    _log_ms(t2, "deskew.rotate")

    _log_ms(t0, "deskew.total")
    return Image.fromarray(rotated), angle


def enhance_ocr_image_fast(image_input: Image.Image, upscale_target: int = 1200) -> Tuple[Image.Image, Dict[str, float]]:
    """
    Conditional enhancement: CLAHE/unsharp only when needed; Otsu binarization; optional light upscale.
    Returns (enhanced_pil, metrics).
    """
    t_total = _t()
    if not isinstance(image_input, Image.Image):
        raise TypeError("image_input must be a PIL.Image.Image object.")

    t0 = _t()
    rgb = np.array(image_input.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _log_ms(t0, "enhance_fast.rgb2gray")

    t1 = _t()
    m = _quality_metrics(gray)
    _log_ms(t1, "enhance_fast.metrics")

    # Contrast boost only if low
    if m["contrast"] < 25:
        t2 = _t()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _log_ms(t2, "enhance_fast.clahe")

    # Light unsharp if blur is high (i.e., blur metric low)
    if m["blur"] < 120:
        t3 = _t()
        g = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        gray = cv2.addWeighted(gray, 1.5, g, -0.5, 0)
        _log_ms(t3, "enhance_fast.unsharp")

    t4 = _t()
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _log_ms(t4, "enhance_fast.otsu_threshold")

    # Upscale only if small (longest side below target)
    h, w = bin_img.shape
    longest = max(h, w)
    if longest < upscale_target:
        t5 = _t()
        scale = upscale_target / float(longest)
        bin_img = cv2.resize(bin_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        _log_ms(t5, "enhance_fast.resize")

    _log_ms(t_total, "enhance_fast.total")
    return Image.fromarray(bin_img), m


def choose_psm(enhanced: Image.Image, metrics: Dict[str, float]) -> int:
    """
    Heuristic PSM chooser based on aspect ratio and image quality.
    """
    t0 = _t()
    w, h = enhanced.size
    aspect = w / float(h)
    psm = 6
    if aspect > 4.0 or aspect < 0.25:
        psm = 7
    if metrics["contrast"] < 15 or metrics["blur"] < 80:
        psm = 11
    logger.info("choose_psm: psm=%d (aspect=%.2f, blur=%.1f, contrast=%.1f)", psm, aspect, metrics["blur"], metrics["contrast"])
    _log_ms(t0, "choose_psm")
    return psm


def prepare_image_for_ocr(pil_img: Image.Image) -> Tuple[Image.Image, int, float, Dict[str, float]]:
    """
    Full preparation pipeline:
      - EXIF orientation fix
      - Fast deskew
      - Conditional enhancement
      - Heuristic PSM selection
    Returns (enhanced_pil, psm, deskew_angle, metrics)
    """
    t_total = _t()
    t0 = _t()
    img = ImageOps.exif_transpose(pil_img)
    _log_ms(t0, "prepare.exif_transpose")

    t1 = _t()
    img, angle = _deskew_pil(img)
    _log_ms(t1, "prepare.deskew")

    t2 = _t()
    enhanced, metrics = enhance_ocr_image_fast(img)
    _log_ms(t2, "prepare.enhance_fast")

    t3 = _t()
    psm = choose_psm(enhanced, metrics)
    _log_ms(t3, "prepare.choose_psm")

    logger.info("prepare_image_for_ocr: angle=%.2f, blur=%.1f, contrast=%.1f", angle, metrics["blur"], metrics["contrast"])
    _log_ms(t_total, "prepare.total")
    return enhanced, psm, angle, metrics
