import io
import logging

import cv2
import numpy as np
import torch
import os
from PIL import Image, ImageOps

# 1. Memory Conflict Protection in C++
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
torch.set_num_threads(1)

import paddleocr.paddleocr as p_ocr  # Импортируем сам модуль для патча
from paddleocr import PaddleOCR, PPStructure
from core.schemas import ImageOcrResult
from services.stamp_detector import stamp_detector
from finetuned.uz_recognizer import UzbekRecognizer
from services import utils

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
_uz_recognizer = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Initializing PaddleOCR detector...")
        _ocr_engine = PaddleOCR(
            det_model_dir='/root/.paddleocr/whl/det/ml/Multilingual_PP-OCRv3_det_infer',
            rec_model_dir='/root/.paddleocr/whl/rec/cyrillic/cyrillic_PP-OCRv3_rec_infer',
            use_gpu=False,
            use_angle_cls=False,
        )
    return _ocr_engine


def get_uz_recognizer():
    global _uz_recognizer
    if _uz_recognizer is None:
        logger.info("Initializing UzbekRecognizer...")
        _uz_recognizer = UzbekRecognizer(
            model_dir='./finetuned/inference',
            char_dict_path='./finetuned/ideal_ocr_charset.txt',
        )
    return _uz_recognizer


def get_table_engine():
    global _table_engine
    if _table_engine is None:
        import paddle
        paddle.set_device('gpu:0')
        logger.info("Initializing PPStructure engine in worker...")
        _table_engine = PPStructure(
            lang="ru",
            show_log=False,
            use_gpu=True,
        )
    return _table_engine


def isolate_stamp_color(cv_img: np.ndarray) -> np.ndarray:
    """
    Выделяет синий текст штампа, создавая чистый чёрно-белый вывод.
    Использует вычитание каналов (B - max(R,G)) для сохранения деталей букв.
    """
    b, g, r = cv2.split(cv_img)

    # «Синесть» — чем сильнее синий канал превосходит красный/зелёный,
    # тем темнее будет пиксель. Это сохраняет тонкие штрихи букв.
    max_rg = cv2.max(r, g)
    raw_blue = cv2.subtract(b, max_rg)

    # Усиливаем контраст
    if raw_blue.max() > 0:
        raw_blue = cv2.normalize(raw_blue, None, 0, 255, cv2.NORM_MINMAX)

    # Дополнительно: HSV-маска для подавления не-синих пикселей
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 20, 20]), np.array([160, 255, 255]))

    # Комбинируем: оставляем только пиксели в синем диапазоне
    raw_blue[blue_mask == 0] = 0

    # Адаптивный порог → чистый чёрно-белый
    _, binary = cv2.threshold(raw_blue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Инвертируем: чёрный текст на белом фоне
    binary = cv2.bitwise_not(binary)

    return cv2.merge([binary, binary, binary])


def _detect_table_grid(table_crop_bgr):
    """
    Обнаруживает горизонтальные и вертикальные линии таблицы
    через морфологические операции. Возвращает (h_lines, v_lines).
    """
    gray = cv2.cvtColor(table_crop_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape

    # Горизонтальные линии
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 12, 30), 1))
    horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)

    # Вертикальные линии
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 12, 30)))
    vert = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)

    # Извлекаем Y-позиции горизонтальных линий
    horiz_proj = np.sum(horiz, axis=1)
    h_lines = _extract_line_positions(horiz_proj, w * 0.08)

    # Извлекаем X-позиции вертикальных линий
    vert_proj = np.sum(vert, axis=0)
    v_lines = _extract_line_positions(vert_proj, h * 0.08)

    # Добавляем границы если их нет
    if not h_lines or h_lines[0] > 15:
        h_lines.insert(0, 0)
    if not h_lines or h_lines[-1] < h - 15:
        h_lines.append(h)
    if not v_lines or v_lines[0] > 15:
        v_lines.insert(0, 0)
    if not v_lines or v_lines[-1] < w - 15:
        v_lines.append(w)

    return h_lines, v_lines


def _extract_line_positions(projection, threshold):
    """Из проекционного профиля извлекает позиции линий."""
    lines = []
    in_line = False
    line_start = 0
    for i, val in enumerate(projection):
        if val > threshold:
            if not in_line:
                line_start = i
                in_line = True
        else:
            if in_line:
                lines.append((line_start + i) // 2)
                in_line = False
    if in_line:
        lines.append((line_start + len(projection)) // 2)
    return lines


def _build_table_html_with_grid(ocr_lines, table_crop_bgr):
    """
    Строит HTML-таблицу, используя реальные линии сетки из изображения.
    Каждая ячейка определяется пересечением обнаруженных линий.
    Тексты внутри одной ячейки объединяются.
    """
    if not ocr_lines:
        return "<html><body><table><tbody><tr><td></td></tr></tbody></table></body></html>"

    h_lines, v_lines = _detect_table_grid(table_crop_bgr)

    num_rows = len(h_lines) - 1
    num_cols = len(v_lines) - 1

    logger.info(f"Table grid: {num_rows} rows x {num_cols} cols "
                f"(h_lines={h_lines}, v_lines={v_lines})")

    # Если линии не обнаружены — fallback на простую кластеризацию
    if num_rows < 1 or num_cols < 1:
        return _build_table_html_fallback(ocr_lines)

    # Инициализируем сетку ячеек
    grid = [[[] for _ in range(num_cols)] for _ in range(num_rows)]

    # Маппим каждый OCR-текст в соответствующую ячейку
    for line in ocr_lines:
        box = line[0]
        text = line[1][0]
        cx = (box[0][0] + box[2][0]) / 2
        cy = (box[0][1] + box[2][1]) / 2

        row = -1
        for r in range(num_rows):
            if h_lines[r] <= cy < h_lines[r + 1]:
                row = r
                break

        col = -1
        for c in range(num_cols):
            if v_lines[c] <= cx < v_lines[c + 1]:
                col = c
                break

        if row >= 0 and col >= 0:
            grid[row][col].append((cx, text))

    # Строим HTML, объединяя тексты внутри каждой ячейки
    html_parts = ["<html><body><table>"]
    for r in range(num_rows):
        if r == 0:
            html_parts.append("<thead><tr>")
        else:
            if r == 1:
                html_parts.append("<tbody>")
            html_parts.append("<tr>")

        for c in range(num_cols):
            cell_texts = sorted(grid[r][c], key=lambda x: x[0])
            cell_text = " ".join(t for _, t in cell_texts)
            html_parts.append(f"<td>{cell_text}</td>")

        if r == 0:
            html_parts.append("</tr></thead>")
        else:
            html_parts.append("</tr>")

    if num_rows > 1:
        html_parts.append("</tbody>")
    html_parts.append("</table></body></html>")

    return "".join(html_parts)


def _build_table_html_fallback(ocr_lines):
    """Fallback: если линии сетки не обнаружены, кластеризуем по Y/X."""
    entries = []
    for line in ocr_lines:
        box = line[0]
        text = line[1][0]
        cx = (box[0][0] + box[2][0]) / 2
        cy = (box[0][1] + box[2][1]) / 2
        h = max(p[1] for p in box) - min(p[1] for p in box)
        entries.append({'text': text, 'cx': cx, 'cy': cy, 'h': h})

    entries.sort(key=lambda e: e['cy'])
    avg_h = sum(e['h'] for e in entries) / len(entries)
    threshold = max(avg_h * 0.6, 8)

    rows = []
    current_row = [entries[0]]
    for entry in entries[1:]:
        if abs(entry['cy'] - current_row[-1]['cy']) < threshold:
            current_row.append(entry)
        else:
            rows.append(sorted(current_row, key=lambda e: e['cx']))
            current_row = [entry]
    rows.append(sorted(current_row, key=lambda e: e['cx']))

    html_parts = ["<html><body><table>"]
    for i, row in enumerate(rows):
        tag_open = "<thead><tr>" if i == 0 else ("<tbody>" if i == 1 else "") + "<tr>"
        html_parts.append(tag_open)
        for cell in row:
            html_parts.append(f"<td>{cell['text']}</td>")
        tag_close = "</tr></thead>" if i == 0 else "</tr>"
        html_parts.append(tag_close)
    if len(rows) > 1:
        html_parts.append("</tbody>")
    html_parts.append("</table></body></html>")
    return "".join(html_parts)


def _extract_circular_text(stamp_bgr, filtered_stamp, ocr_engine):
    """
    Извлекает круговой текст из штампа через polar unwrap.
    Находит круг, разворачивает кольцо в полосу, OCR по полосе.
    """
    gray = cv2.cvtColor(filtered_stamp, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Ищем круг штампа
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    min_r = min(h, w) // 5
    max_r = min(h, w) // 2

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(h, w) // 2,
        param1=80, param2=25,
        minRadius=min_r, maxRadius=max_r
    )

    if circles is None:
        logger.info("No circle found in stamp for polar unwrap")
        return ""

    circle = circles[0][0]
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    logger.info(f"Stamp circle: center=({cx:.0f},{cy:.0f}), radius={r:.0f}")

    # Разворачиваем кольцо: polar → cartesian
    # Ширина = окружность, Высота = радиус
    circumference = int(2 * np.pi * r)
    out_h = int(r)

    if circumference < 50 or out_h < 10:
        return ""

    unwrapped = cv2.warpPolar(
        filtered_stamp,
        (circumference, out_h),
        (cx, cy),
        r,
        cv2.WARP_POLAR_LINEAR
    )

    # Кольцо текста: обычно 55-95% радиуса от центра
    inner_row = int(out_h * 0.55)
    outer_row = int(out_h * 0.95)
    ring_strip = unwrapped[inner_row:outer_row, :]

    if ring_strip.size == 0:
        return ""

    # Масштабируем тонкую полосу для лучшего OCR
    strip_h = ring_strip.shape[0]
    if strip_h < 40:
        scale_y = 40.0 / strip_h
        ring_strip = cv2.resize(
            ring_strip, None, fx=1, fy=scale_y,
            interpolation=cv2.INTER_CUBIC
        )

    # OCR развёрнутой полосы (cls=True для перевёрнутого текста)
    text_parts = []
    try:
        result = ocr_engine.ocr(ring_strip, cls=True)
        if result and result[0]:
            text_parts = [line[1][0] for line in result[0]]
    except Exception as e:
        logger.warning(f"Circular OCR failed: {e}")

    return " ".join(text_parts)


def _extract_full_stamp_text(stamp_bgr, ocr_engine):
    """
    Полное извлечение текста из штампа:
    1. Горизонтальный текст в центре (обычный OCR)
    2. Круговой текст вокруг штампа (polar unwrap + OCR)
    Возвращает объединённый текст через ' | '.
    """
    # Фильтруем синий цвет штампа
    filtered = isolate_stamp_color(stamp_bgr)

    # 1. Горизонтальный текст
    center_text = ""
    try:
        res = ocr_engine.ocr(filtered, cls=True)
        if res and res[0]:
            center_text = " ".join(line[1][0] for line in res[0])
    except Exception as e:
        logger.warning(f"Center stamp OCR failed: {e}")

    # Если фильтр не дал — пробуем без фильтра
    if not center_text.strip():
        try:
            res = ocr_engine.ocr(stamp_bgr, cls=True)
            if res and res[0]:
                center_text = " ".join(line[1][0] for line in res[0])
        except Exception:
            pass

    # 2. Круговой текст
    circular_text = _extract_circular_text(stamp_bgr, filtered, ocr_engine)

    # Объединяем
    parts = []
    if circular_text.strip():
        parts.append(circular_text.strip())
    if center_text.strip():
        parts.append(center_text.strip())

    return " | ".join(parts) if parts else ""


def _count_orientation(boxes):
    """Подсчитывает горизонтальные и вертикальные текстовые блоки."""
    horiz = 0
    vert = 0
    for box in boxes:
        w = np.linalg.norm(np.array(box[1]) - np.array(box[0]))
        h = np.linalg.norm(np.array(box[3]) - np.array(box[0]))
        if w > h * 1.3:
            horiz += 1
        elif h > w * 1.3:
            vert += 1
    return horiz, vert


def _auto_orient(cv_img: np.ndarray, ocr_engine) -> tuple:
    """
    Определяет ориентацию страницы по форме текстовых боксов.
    Если текст преимущественно вертикальный — страница повёрнута на 90°.
    Возвращает (cv_img_corrected, pil_img_corrected).
    """
    try:
        # Быстрый detection-only проход (без распознавания текста)
        det_result = ocr_engine.ocr(cv_img, cls=False, rec=False)
        if not det_result or not det_result[0]:
            pil_img = Image.fromarray(cv_img[:, :, ::-1])
            return cv_img, pil_img

        horiz, vert = _count_orientation(det_result[0])
        logger.info(f"Orientation check: {horiz} horizontal, {vert} vertical text boxes")

        if vert > horiz:
            # Пробуем оба направления вращения, выбираем лучшее
            cw = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
            ccw = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            det_cw = ocr_engine.ocr(cw, cls=False, rec=False)
            det_ccw = ocr_engine.ocr(ccw, cls=False, rec=False)

            h_cw = _count_orientation(det_cw[0])[0] if det_cw and det_cw[0] else 0
            h_ccw = _count_orientation(det_ccw[0])[0] if det_ccw and det_ccw[0] else 0

            if h_cw >= h_ccw:
                cv_img = cw
                logger.info(f"Auto-rotated 90° CW ({h_cw} horiz boxes)")
            else:
                cv_img = ccw
                logger.info(f"Auto-rotated 90° CCW ({h_ccw} horiz boxes)")
    except Exception as e:
        logger.warning(f"Orientation detection failed: {e}")

    pil_img = Image.fromarray(cv_img[:, :, ::-1])  # BGR → RGB → PIL
    return cv_img, pil_img


def process_image_from_pil(pil_image: Image.Image) -> ImageOcrResult:
    img = ImageOps.exif_transpose(pil_image)
    cv_img = np.array(img.convert('RGB'))
    cv_img = cv_img[:, :, ::-1].copy()  # Конвертация RGB в BGR для OpenCV

    ocr_engine = get_ocr_engine()
    uz_rec = get_uz_recognizer()
    table_engine = get_table_engine()

    # 0. Авто-определение ориентации через быстрый detection-only проход
    cv_img, img = _auto_orient(cv_img, ocr_engine)

    # 0.1 Выравнивание перекоса (1-15°, Hough-based)
    cv_img, skew_angle = utils.deskew(cv_img)
    if skew_angle != 0.0:
        img = Image.fromarray(cv_img[:, :, ::-1])  # Обновляем PIL-версию

    # 0.2 Метрики качества + условная предобработка
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    metrics = utils.quality_metrics(gray)
    logger.info("Image quality: blur=%.1f, brightness=%.1f, contrast=%.1f",
                metrics["blur"], metrics["brightness"], metrics["contrast"])
    cv_img = utils.conditional_enhance(cv_img, metrics)

    img_height, img_width = cv_img.shape[:2]

    # 1. Ищем штампы
    detected_stamps = stamp_detector.detect(img)

    # 2. Извлекаем текст из каждого найденного штампа
    for stamp in detected_stamps:
        try:
            # Вырезаем область штампа с запасом (+15 пикселей для лучшего контекста)
            x1 = max(0, stamp.box.x1 - 15)
            y1 = max(0, stamp.box.y1 - 15)
            x2 = min(cv_img.shape[1], stamp.box.x2 + 15)
            y2 = min(cv_img.shape[0], stamp.box.y2 + 15)

            cropped_stamp_cv = cv_img[y1:y2, x1:x2]

            # Масштабируем мелкие штампы для лучшего распознавания
            h_stamp, w_stamp = cropped_stamp_cv.shape[:2]
            min_dim = 600  # Минимум для хорошего OCR + polar unwrap
            if max(h_stamp, w_stamp) < min_dim:
                scale = min_dim / max(h_stamp, w_stamp)
                cropped_stamp_cv = cv2.resize(
                    cropped_stamp_cv, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )
                logger.info(f"Upscaled stamp from {w_stamp}x{h_stamp} by {scale:.1f}x")

            # Извлекаем текст штампа (горизонтальный + круговой)
            stamp.text = _extract_full_stamp_text(cropped_stamp_cv, ocr_engine)
        except Exception as e:
            logger.warning(f"Failed to extract text from stamp: {e}")

    # 3. Ищем таблицы через PPStructure (получаем bounding box'ы для разделения текста)
    table_bboxes = []  # список [x1, y1, x2, y2] для каждой таблицы
    try:
        rgb_img = cv_img[:, :, ::-1].copy()  # BGR → RGB для PPStructure
        structure_res = table_engine(rgb_img)
        for region in structure_res:
            if region['type'] == 'table':
                table_bboxes.append(region['bbox'])
                logger.info(f"Detected table region: {region['bbox']}")
    except Exception as e:
        logger.error(f"PPStructure error: {e}")

    # 4. Детекция текстовых боксов (без распознавания)
    full_text = ""
    tables_html = []
    try:
        boxes = ocr_engine.ocr(cv_img, rec=False, cls=False)

        if boxes and boxes[0]:
            free_text_lines = []  # текст ВНЕ таблиц
            table_lines_map = {i: [] for i in range(len(table_bboxes))}  # текст ВНУТРИ каждой таблицы

            for box in boxes[0]:
                pts = np.array(box, dtype=np.int32)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)

                # Центр текстовой строки
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Recognition через кастомную модель
                crop = cv_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                text = uz_rec.predict(crop)
                if not text.strip():
                    continue

                # Определяем, попадает ли строка в одну из таблиц
                assigned_to_table = False
                for idx, tbbox in enumerate(table_bboxes):
                    if (tbbox[0] <= center_x <= tbbox[2] and
                            tbbox[1] <= center_y <= tbbox[3]):
                        # Сдвигаем координаты относительно области таблицы
                        shifted_box = [
                            [p[0] - tbbox[0], p[1] - tbbox[1]] for p in box
                        ]
                        table_lines_map[idx].append((shifted_box, (text, 1.0)))
                        assigned_to_table = True
                        break

                if not assigned_to_table:
                    free_text_lines.append((center_y, center_x, text))

            # Sort by Y (top-to-bottom), then X (left-to-right)
            free_text_lines.sort(key=lambda t: (t[0], t[1]))
            full_text = " ".join(t[2] for t in free_text_lines)

            # 5. Строим HTML-таблицы из собранных OCR-строк
            for idx in range(len(table_bboxes)):
                table_ocr_lines = table_lines_map[idx]
                if table_ocr_lines:
                    tbbox = table_bboxes[idx]
                    # Кропим область таблицы из изображения для детекции линий сетки
                    tx1, ty1 = int(tbbox[0]), int(tbbox[1])
                    tx2, ty2 = int(tbbox[2]), int(tbbox[3])
                    table_crop = cv_img[ty1:ty2, tx1:tx2]
                    html = _build_table_html_with_grid(table_ocr_lines, table_crop)
                    tables_html.append(html)
                    logger.info(f"Built table HTML with {len(table_ocr_lines)} text lines")
        else:
            # Если детекция ничего не нашла, но таблицы есть — оставляем пустые
            for _ in table_bboxes:
                tables_html.append(
                    "<html><body><table><tbody><tr><td></td></tr></tbody></table></body></html>"
                )
    except Exception as e:
        logger.exception(f"PaddleOCR error: {e}")

    return ImageOcrResult(text=full_text, stamps=detected_stamps, tables_html=tables_html)


def process_image_from_path(image_path: str) -> ImageOcrResult:
    with Image.open(image_path) as pil_img:
        return process_image_from_pil(pil_img)


def process_image_from_bytes(image_bytes: bytes) -> ImageOcrResult:
    with Image.open(io.BytesIO(image_bytes)) as pil_img:
        return process_image_from_pil(pil_img)
