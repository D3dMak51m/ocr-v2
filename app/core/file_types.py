from typing import Optional

TYPE_IMG = "image"
TYPE_PDF = "pdf"
TYPE_DOC = "doc"
TYPE_PPT = "ppt"
TYPE_TIKA = "tika"
TYPE_EXCEL = "excel"
TYPE_RTF = "rtf"

# Mapping of file extensions to their type
EXTENSION_TO_TYPE = {ext: TYPE_IMG for ext in ["jpeg", "jpg", "png", "gif",
                                               "bmp"]}
EXTENSION_TO_TYPE.update({ext: TYPE_DOC for ext in ["doc", "docx"]})
EXTENSION_TO_TYPE.update({ext: TYPE_PDF for ext in ["pdf"]})
EXTENSION_TO_TYPE.update({ext: TYPE_PPT for ext in ["ppt", "pptx"]})
EXTENSION_TO_TYPE.update({ext: TYPE_EXCEL for ext in ["xls", "xlsx"]})
EXTENSION_TO_TYPE.update({ext: TYPE_RTF for ext in ["rtf"]})

# File types processed by Tika, including those that might have embedded images
TIKA_FILE_TYPES = {TYPE_DOC, TYPE_PDF, TYPE_PPT,
                   TYPE_EXCEL, TYPE_RTF, TYPE_TIKA}


def get_file_type(extension: str) -> Optional[str]:
    """
    Determines the file type from its extension.
    Returns the file type string or None if not directly supported.
    """
    return EXTENSION_TO_TYPE.get(
        extension.lower(), TYPE_TIKA
    )  # Default to Tika for unknown types
