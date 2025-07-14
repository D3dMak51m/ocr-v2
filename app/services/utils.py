from PIL import Image
import pytesseract
import os
from urllib.parse import urlparse
import sys
import re
from bs4 import BeautifulSoup

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
