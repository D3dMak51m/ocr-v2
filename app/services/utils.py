from PIL import Image, ImageOps
import pytesseract
import os
from urllib.parse import urlparse
import sys
import re
from bs4 import BeautifulSoup

import cv2
import numpy as np
from typing import Union, Tuple, Dict

# Allow importing local detector if needed
sys.path.insert(0, "..")
from lang_detector import detector  # noqa: E402

files_tmp = "temp_files/"

image_extensions = ["jpeg/jpg", "jpeg", "jpg", "png", "gif", "bmp"]


def text_formatting(text: str) -> str:
    """
    Simple whitespace normalization for OCR text.
    """
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r" +", " ", text)
    return text


def get_file_extension(url: str) -> str:
    """
    Extract a lowercase extension from a URL or path.
    """
    parsed_url = urlparse(url)
    ext = os.path.splitext(parsed_url.path)[1].lower()
    if ext.startswith("."):
        ext = ext[1:]
    return ext


def extract_text_uzb(input_image: Union[str, Image.Image]) -> str:
    """
    Direct call to Tesseract for Uzbek Cyrillic/Uzbek/English bundle.
    """
    if isinstance(input_image, str):
        img = Image.open(input_image)
    elif isinstance(input_image, Image.Image):
        img = input_image
    else:
        raise ValueError("Input must be a file path (str) or a PIL Image object.")
    return pytesseract.image_to_string(
        image=img, lang="uzb_cyrl+uzb+eng", config="-c min_characters_to_try=5"
    )


def language_detector(text_uz: str):
    """
    Returns {'language': code, 'score': float} or defaults if none.
    """
    lang_list = detector.detect_lang(text_uz)
    print("language_list: ", str(lang_list))
    if len(lang_list) > 0:
        return lang_list[0]
    return {"language": None, "score": 0}


def xml_to_txt(xml_table: str) -> str:
    """
    Convert HTML/XML to plain text, then normalize whitespace.
    """
    soup = BeautifulSoup(xml_table, "html.parser")
    return text_formatting(soup.get_text())


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
    if not isinstance(image_input, Image.Image):
        raise TypeError("image_input must be a PIL.Image.Image object.")

    # Convert to BGR (OpenCV format)
    img = np.array(image_input.convert("RGB"))[:, :, ::-1]  # RGB to BGR

    # Step 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=25)

    # Step 3: Thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Step 4: Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)

    # Step 5: Resizing
    resized = cv2.resize(
        sharpened, None,
        fx=scale_factor, fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )

    # Save debug images if needed
    if save_debug:
        cv2.imwrite(f"{debug_prefix}_gray.jpg", gray)
        cv2.imwrite(f"{debug_prefix}_denoised.jpg", denoised)
        cv2.imwrite(f"{debug_prefix}_thresh.jpg", thresh)
        cv2.imwrite(f"{debug_prefix}_sharpened.jpg", sharpened)
        cv2.imwrite(f"{debug_prefix}_resized.jpg", resized)

    # Return final result as PIL image (mode "L" = grayscale)
    return Image.fromarray(resized)


# ------------------- Faster, conditional pipeline ------------------- #

def _quality_metrics(gray: np.ndarray) -> Dict[str, float]:
    """
    Compute quick metrics to decide how much enhancement is needed.
    """
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean, std = cv2.meanStdDev(gray)
    return {
        "blur": float(blur),
        "brightness": float(mean[0][0]),
        "contrast": float(std[0][0]),
    }


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate skew angle (degrees) with Hough on edges; robust and quick.
    Positive angle means image should be rotated counter-clockwise by that amount.
    """
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Make text dark for line detection if needed
    if np.mean(thr) > 127:
        thr = 255 - thr
    edges = cv2.Canny(thr, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 200)
    if lines is None:
        return 0.0
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90.0
        if -45 < angle < 45:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def _deskew_pil(image: Image.Image) -> Tuple[Image.Image, float]:
    """
    Deskew using Hough-based angle estimation.
    """
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    angle = _estimate_skew_angle(gray)
    if abs(angle) < 0.5:
        return image, 0.0
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated), angle


def enhance_ocr_image_fast(image_input: Image.Image, upscale_target: int = 1200) -> Tuple[Image.Image, Dict[str, float]]:
    """
    Conditional enhancement: CLAHE/unsharp only when needed; Otsu binarization; optional light upscale.
    Returns (enhanced_pil, metrics).
    """
    if not isinstance(image_input, Image.Image):
        raise TypeError("image_input must be a PIL.Image.Image object.")

    # PIL -> numpy (RGB -> GRAY)
    rgb = np.array(image_input.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    m = _quality_metrics(gray)

    # Contrast boost only if low
    if m["contrast"] < 25:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Light unsharp if blur is high (i.e., blur metric low)
    if m["blur"] < 120:
        g = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        gray = cv2.addWeighted(gray, 1.5, g, -0.5, 0)

    # Binarize (fast and usually best for paragraph text)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Upscale only if small (longest side below target)
    h, w = bin_img.shape
    longest = max(h, w)
    if longest < upscale_target:
        scale = upscale_target / float(longest)
        bin_img = cv2.resize(bin_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    return Image.fromarray(bin_img), m


def choose_psm(enhanced: Image.Image, metrics: Dict[str, float]) -> int:
    """
    Heuristic PSM chooser based on aspect ratio and image quality.
    """
    w, h = enhanced.size
    aspect = w / float(h)
    # Default: block/paragraph of text
    psm = 6
    # Very long or very tall -> single line
    if aspect > 4.0 or aspect < 0.25:
        psm = 7
    # Very sparse or weak contrast/blur -> sparse text
    if metrics["contrast"] < 15 or metrics["blur"] < 80:
        psm = 11  # sparse text
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
    # Cheap orientation fix from EXIF first
    img = ImageOps.exif_transpose(pil_img)

    # Fast deskew (skip OSD here; OSD is called upstream for 90° steps)
    img, angle = _deskew_pil(img)

    enhanced, metrics = enhance_ocr_image_fast(img)
    psm = choose_psm(enhanced, metrics)
    return enhanced, psm, angle, metrics
