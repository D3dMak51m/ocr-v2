import requests
import os
import tempfile
import zipfile
from services import image_service
from services.utils import text_formatting
from ocr_models.bmodels import DocOcrResult
from tika_client import TikaClient
from pathlib import Path
from ocr_models import extensions_wrapper
import fitz
from PIL import Image
import io

tika_server_url_root = "http://tika-server:9998"


tika_server_url_unpack = tika_server_url_root + "/unpack"
tika_server_url_tika = tika_server_url_root + "/tika"


# X-TIKA:embedded_resource_path
headers = {"Accept-Charset": "UTF-8"}  # Adjust content type as needed


def tika_get_content(filepath):
    with TikaClient(tika_url=tika_server_url_root) as client:
        path = Path(filepath)
        text = client.tika.as_text.from_file(path)
        # text = client.tika.as_text.from_file(filepath)
        print(f"text from tika client: {text}")
    if text is not None:
        return text.content
    return ""


def tika_text_from_images(filepath):
    with open(filepath, "rb") as f:
        response = requests.put(tika_server_url_unpack, data=f, headers=headers)

        image_response = []

        if response.status_code == 200:
            print("status 200")
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmpzip:
                tmpzip.write(response.content)
                tmpzip.seek(0)

                with tempfile.TemporaryDirectory() as temp_extract_folder:

                    with zipfile.ZipFile(tmpzip, "r") as zip_ref:
                        # Extract all contents to the temp directory
                        zip_ref.extractall(temp_extract_folder)

                        extracted_files = [
                            temp_extract_folder + "/" + item
                            for item in os.listdir(temp_extract_folder)
                        ]
                        print(f"Extracted files: {extracted_files}")

                        for exfile in extracted_files:
                            try:
                                res = image_service.runner_image_v1(exfile)
                                image_response.append(res)

                            except Exception as e:
                                print(e)

        else:
            print(f"status {response.status_code}")

        return image_response


def pdf_to_images_text(pdf_path):
    result_list = []
    with fitz.open(pdf_path) as pdf_file:
        for page_number in range(len(pdf_file)):
            page = pdf_file[page_number]

            image_list = page.get_images()

            for image_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                pil_image = Image.open(io.BytesIO(image_bytes))

                with tempfile.NamedTemporaryFile(
                    suffix=f".{image_ext}", delete=True
                ) as tempImg:
                    pil_image.save(tempImg)
                    data = image_service.runner_image_v1(tempImg)
                    data.filename = f"page_{page_number}_img_{image_index}.{image_ext}"
                    result_list.append(data)

    return result_list


def runner_tika(filepath, filetype):
    text = tika_get_content(filepath)
    print(f"text from tike get content: {text}")

    text = text_formatting(text)
    if filetype == extensions_wrapper.TYPE_PDF:
        images = pdf_to_images_text(filepath)
    else:
        images = tika_text_from_images(filepath)

    docResult = DocOcrResult(text=text, images=images, service="tika")
    return docResult
