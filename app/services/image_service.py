import io
import logging
import numpy as np
import torch
import os
from typing import Optional, List, Tuple
from PIL import Image, ImageOps

# 1. Memory Conflict Protection in C++
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
torch.set_num_threads(1)

import paddleocr.paddleocr as p_ocr  # Импортируем сам модуль для патча
from paddleocr import PaddleOCR, PPStructure
from core.schemas import ImageOcrResult
from services.stamp_detector import stamp_detector

logger = logging.getLogger(__name__)

# ==============================================================================
# 2. HOT FIX: Bypass PaddleOCR's hard limitation on 'en' and 'ch' languages. Use *args and **kwargs to ensure the patch works.
if not hasattr(p_ocr, '_original_get_model_config'):
    p_ocr._original_get_model_config = p_ocr.get_model_config


    def patched_get_model_config(*args, **kwargs):
        new_args = list(args)
        if len(new_args) >= 4:
            model_type = new_args[2]
            lang = new_args[3]
            if model_type in ['layout', 'table'] and lang not in ['en', 'ch']:
                new_args[3] = 'en'

        if 'model_type' in kwargs and 'lang' in kwargs:
            if kwargs['model_type'] in ['layout', 'table'] and kwargs['lang'] not in ['en', 'ch']:
                kwargs['lang'] = 'en'

        return p_ocr._original_get_model_config(*new_args, **kwargs)


    p_ocr.get_model_config = patched_get_model_config
# ==============================================================================

_ocr_engine = None
_table_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Initializing PaddleOCR engine in worker...")
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="cyrillic", show_log=False)
    return _ocr_engine


def get_table_engine():
    global _table_engine
    if _table_engine is None:
        logger.info("Initializing PPStructure engine in worker...")
        _table_engine = PPStructure(lang="ru", show_log=False)
    return _table_engine


def process_image_from_pil(pil_image: Image.Image) -> ImageOcrResult:
    img = ImageOps.exif_transpose(pil_image)
    cv_img = np.array(img.convert('RGB'))
    cv_img = cv_img[:, :, ::-1].copy()

    ocr_engine = get_ocr_engine()
    table_engine = get_table_engine()

    detected_stamps = stamp_detector.detect(img)

    full_text = ""
    try:
        ocr_result = ocr_engine.ocr(cv_img, cls=True)
        if ocr_result and ocr_result[0]:
            lines = [line[1][0] for line in ocr_result[0]]
            full_text = " ".join(lines)
    except Exception as e:
        logger.error(f"PaddleOCR error: {e}")

    tables_html = []
    try:
        structure_res = table_engine(cv_img)
        for region in structure_res:
            if region['type'] == 'table':
                tables_html.append(region['res']['html'])
    except Exception as e:
        logger.error(f"PPStructure error: {e}")

    return ImageOcrResult(text=full_text, stamps=detected_stamps, tables_html=tables_html)


def process_image_from_path(image_path: str) -> ImageOcrResult:
    try:
        with Image.open(image_path) as pil_img:
            return process_image_from_pil(pil_img)
    except Exception as e:
        raise


def process_image_from_bytes(image_bytes: bytes) -> ImageOcrResult:
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            return process_image_from_pil(pil_img)
    except Exception as e:
        raise