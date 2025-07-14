TYPE_IMG = "image"
TYPE_PDF = "pdf"
TYPE_DOC = "doc"
TYPE_PPT = "ppt"
TYPE_TIKA = "tika"  # default file type
TYPE_EXCEL = "excel"
TYPE_RTF = "rtf"

image_extensions = ["jpeg/jpg", "jpeg", "jpg", "png", "gif", "bmp"]

docs_extensions = ["doc", "docx"]

pdf_extensions = ["pdf"]
ppt_extensions = ["ppt", "pptx"]
excel_extensions = ["xls", "xlsx"]
rtf_extensions = ["rtf"]

extensions = {
    TYPE_IMG: image_extensions,  #
    TYPE_DOC: docs_extensions,  #
    TYPE_PDF: pdf_extensions,  #
    TYPE_PPT: ppt_extensions,  # tika
    TYPE_RTF: rtf_extensions,  # ?
    TYPE_EXCEL: excel_extensions,  # tika ?
}
