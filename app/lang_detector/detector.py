import lang_detector.cld2_utils.cld2_api as cld2_api


def detect_lang(text: str) -> list:
    result = cld2_api.detect_lang_cld2(text)
    return result


def detect_lang_cld2(text: str):
    return detect_lang(text, "cld2")


def detect_lang_fasttext(text: str):
    return detect_lang(text, "fasttext")
