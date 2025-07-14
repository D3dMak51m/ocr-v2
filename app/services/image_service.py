import requests
import tempfile

import pytesseract
import re

from services.utils import extract_text_uzb
from services.utils import language_detector
from ocr_models.bmodels import ImageOcrResult
from PIL import Image
import chardet


def ensure_utf8(text: str) -> str:
    detected = chardet.detect(text.encode())
    if detected["encoding"] and detected["encoding"].lower() != "utf-8":
        return text.encode(detected["encoding"], errors="ignore").decode("utf-8")
    return text


def detect_image_encoding(pil_image):
    try:
        osd = pytesseract.image_to_osd(
            pil_image, config="-c min_characters_to_try=5"
        )
        script = re.search("Script: ([a-zA-Z]+)\n", osd).group(1)
        conf = re.search("Script confidence: (\d+\.?(\d+)?)", osd).group(1)
        return script, conf
    except Exception as e:
        print(f"error: {e}")
        return None, 0.0


def runner_image_v1_with_pil(pil_image: Image.Image) -> ImageOcrResult:
    encoding, conf = detect_image_encoding(pil_image)
    print(f"encoding: {encoding}, conf: {conf}")
    raw_text = extract_text_uzb(pil_image)
    text = ensure_utf8(" ".join(raw_text.replace("\n", " ").split()))
    language_d = language_detector(text)
    language = language_d.get("language")
    score = language_d.get("score")
    res = ImageOcrResult(
        status="success",
        text=text,
        language=language,
        language_score=score,
        encoding=encoding,
        encoding_conf=conf,
    )
    return res


def runner_image_v1(tempimagepath) -> ImageOcrResult:
    with Image.open(tempimagepath) as pil_img:
        return runner_image_v1_with_pil(pil_img)


def runner_image_url(url):
    with tempfile.NamedTemporaryFile(delete=True) as temp_image:
        # Download the image and write to the temporary file
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 \
                    Safari/537.3"
        }
        response = requests.get(url, headers=headers)
        temp_image.write(response.content)
        # Ensure all data is written before closing the file
        temp_image.flush()

        return runner_image_v1(temp_image.name)
