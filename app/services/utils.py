import logging
import os
import re
import time
from typing import Tuple, Dict
from urllib.parse import urlparse

import cv2
import numpy as np
from bs4 import BeautifulSoup

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


# ------------------- Image preprocessing for OCR ------------------- #

def quality_metrics(gray: np.ndarray) -> Dict[str, float]:
    """
    Compute quick metrics to decide how much enhancement is needed.
    - blur:       Laplacian variance (high = sharp, low = blurry)
    - brightness: mean pixel intensity
    - contrast:   std of pixel intensity
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


def deskew(cv_img: np.ndarray, skip_threshold: float = 0.5) -> Tuple[np.ndarray, float]:
    """
    Deskew a BGR image using Hough-based angle estimation.
    Skips rotation if |angle| < skip_threshold degrees.
    Returns (corrected_cv_img, angle_degrees).
    """
    t0 = _t()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    angle = _estimate_skew_angle(gray)

    if abs(angle) < skip_threshold:
        _log_ms(t0, "deskew.total(noop)")
        return cv_img, 0.0

    h, w = cv_img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    logger.info("Deskewed by %.2f°", angle)
    _log_ms(t0, "deskew.total")
    return rotated, angle


def conditional_enhance(cv_img: np.ndarray, metrics: Dict[str, float]) -> np.ndarray:
    """
    Conditional image enhancement for PaddleOCR:
    - CLAHE if contrast is low (< 25)
    - Light unsharp masking if image is blurry (blur metric < 120)

    NOTE: No binarization — PaddleOCR works better with grayscale/color input.
    Returns enhanced BGR image.
    """
    t0 = _t()
    enhanced = cv_img.copy()

    # Convert to LAB for CLAHE on lightness channel (preserves color info)
    needs_clahe = metrics["contrast"] < 25
    needs_unsharp = metrics["blur"] < 120

    if not needs_clahe and not needs_unsharp:
        _log_ms(t0, "conditional_enhance(noop)")
        return enhanced

    if needs_clahe:
        t1 = _t()
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        _log_ms(t1, "conditional_enhance.clahe")
        logger.info("Applied CLAHE (contrast=%.1f)", metrics["contrast"])

    if needs_unsharp:
        t2 = _t()
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        _log_ms(t2, "conditional_enhance.unsharp")
        logger.info("Applied unsharp mask (blur=%.1f)", metrics["blur"])

    _log_ms(t0, "conditional_enhance.total")
    return enhanced
