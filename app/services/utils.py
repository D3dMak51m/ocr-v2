from PIL import Image
import pytesseract
import os
from urllib.parse import urlparse
import sys
import re
from bs4 import BeautifulSoup

import cv2
import numpy as np
from typing import Union


sys.path.insert(0, "..")
from lang_detector import detector

# import app.lang_detector.detector as detector

# from app.lang_detector import detector

files_tmp = "temp_files/"

image_extensions = ["jpeg/jpg", "jpeg", "jpg", "png", "gif", "bmp"]


def text_formatting(text):
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r" +", " ", text)
    return text


def get_file_extension(url: str) -> str:
    parsed_url = urlparse(url)
    ext = os.path.splitext(parsed_url.path)[1].lower()
    if ext.startswith("."):
        ext = ext[1:]
    return ext


def extract_text_uzb(input_image):
    if isinstance(input_image, str):
        img = Image.open(input_image)
    elif isinstance(input_image, Image.Image):
        img = input_image
    else:
        raise ValueError("Input must be a file path (str) or a PIL Image object.")
    return pytesseract.image_to_string(
        image=img, lang="uzb_cyrl+uzb+en", config="-c min_characters_to_try=5"
    )


def language_detector(text_uz: str):
    lang_list = detector.detect_lang(text_uz)
    print("language_list: ", str(lang_list))
    if len(lang_list) > 0:
        return lang_list[0]
    return {"language": None, "score": 0}


def xml_to_txt(xml_table):
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
    
    Args:
        image_input (PIL.Image.Image): Input image in RGB or L mode.
        scale_factor (float): Scaling factor for resizing the image.
        save_debug (bool): Save intermediate images for inspection.
        debug_prefix (str): Prefix for debug filenames if save_debug is True.

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
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

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


