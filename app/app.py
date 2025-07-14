import requests
import os
import tempfile
from contextlib import contextmanager
from ocr_models.bmodels import TextractRequest
from ocr_models import extensions_wrapper


from services.utils import get_file_extension
import services.image_service as image_service
import services.tika_service as tika_service


def get_extension_from_url(url):
    """Extract the file extension from the URL."""
    ext = os.path.splitext(url)[-1]  # Get extension from the URL
    if ext:
        return ext.lower()
    return ".jpg"  # Default extension if none found or unsupported


def get_file_type(ext):
    if ext is None:
        return None
    for key in extensions_wrapper.extensions:
        if ext in extensions_wrapper.extensions[key]:
            return key
    return extensions_wrapper.TYPE_TIKA


def runner_file_path(filepath):
    print("filepath: ", filepath)
    extension = get_file_extension(filepath)
    file_type = get_file_type(extension)

    print(extension, file_type)

    # if file_type == TYPE_PDF:
    #     return pdf_service.runner_pdf_file(filepath)

    # if file_type == TYPE_DOC:
    #     return doc_service.runner_doc_file(filepath)

    if file_type == extensions_wrapper.TYPE_IMG:
        return image_service.runner_image_v1(filepath)

    if (
        file_type == extensions_wrapper.TYPE_EXCEL
        or file_type == extensions_wrapper.TYPE_PPT
        or file_type == extensions_wrapper.TYPE_RTF
        or file_type == extensions_wrapper.TYPE_PDF
        or file_type == extensions_wrapper.TYPE_DOC
        or file_type == extensions_wrapper.TYPE_TIKA
    ):
        result = tika_service.runner_tika(filepath, file_type)
        return result

    if file_type == extensions_wrapper.TYPE_TIKA:
        return tika_service.runner_tika(filepath)
    return {
        "status": "error",
        "message": "Unexpected error",
    }


def download_file(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Use a named temporary file for storage
        extension = get_extension_from_url(url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    else:
        raise Exception("Failed to download file")


@contextmanager
def temporary_file(url):
    """Context manager for downloading a file and automatically cleaning up."""
    filename = download_file(url)
    try:
        yield filename
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def runner_file(fileUpload):
    filename = fileUpload.filename
    ext = get_file_extension(filename)
    file_type = get_file_type(ext)

    if file_type is None:
        return {
            "status": "error",
            "message": "Unsupported media type",
            "filtype": ext,
            "supported-types": extensions_wrapper,
        }

    file_data = fileUpload.file.read()

    with tempfile.NamedTemporaryFile(delete=True, suffix=f".{ext}") as temp_file:
        temp_file.write(file_data)
        temp_file.flush()  # Ensure all data is written before processing

        return runner_file_path(temp_file.name)


def runner_url(url):
    ext = get_file_extension(url)

    file_type = get_file_type(ext)

    if file_type is None:
        return {
            "status": "error",
            "message": "Unsupported media type",
            "filtype": ext,
            "supported-types": extensions_wrapper,
        }
    print(f"file type: {file_type}")
    with temporary_file(url) as filename:
        print(f"filename temp: {filename}")
        return runner_file_path(filename)


def runner(request: TextractRequest):
    if request.url:
        return runner_url(request.url)
    if request.local_path:
        return runner_file_path(request.local_path)
    raise Exception("url and local path are both empty")
