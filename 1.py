#!/usr/bin/env python3
"""Помощник для проверки тайпа манхвы.

Алгоритм:
1) Загружает все изображения страниц из папки.
2) Находит русский текст через OCR (Tesseract, язык rus).
3) Собирает строки и их координаты (учитывает только слова, где есть буквы).
4) Группирует строки по OCR-блокам/абзацам.
5) Для каждой строки сравнивает её центр с центром своего текстового блока.
6) Если модуль смещения центра больше tolerance, помечает строку красным.
7) Сохраняет *_fix.png в ту же папку.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Допустимое отклонение центра строки от центра текстового блока (пиксели)
TOLERANCE_PX = 25

# При необходимости переопределите локальный путь к tesseract.exe.
# Например: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")
TEXT_CHAR_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]")


@dataclass
class LineBox:
    text: str
    left: int
    top: int
    right: int
    bottom: int
    block_num: int
    par_num: int
    page_num: int
    bubble_id: int | None = None

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)


@dataclass
class BubbleBox:
    bubble_id: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def center_x(self) -> int:
        return (self.left + self.right) // 2


def has_letters(text: str) -> bool:
    return bool(text and LETTER_RE.search(text))




def estimate_text_horizontal_bounds(line: LineBox) -> Tuple[int, int]:
    """Оценивает границы только текстовой части строки (без крайних знаков)."""
    raw = line.text.strip()
    if not raw:
        return line.left, line.right

    left_trim = 0
    while left_trim < len(raw) and not TEXT_CHAR_RE.search(raw[left_trim]):
        left_trim += 1

    right_trim = 0
    while right_trim < len(raw) and not TEXT_CHAR_RE.search(raw[len(raw) - 1 - right_trim]):
        right_trim += 1

    width = line.right - line.left
    if width <= 0:
        return line.left, line.right

    scale = width / max(len(raw), 1)
    adj_left = int(round(line.left + left_trim * scale))
    adj_right = int(round(line.right - right_trim * scale))

    if adj_right <= adj_left:
        return line.left, line.right
    return adj_left, adj_right


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.stem.endswith("_fix"):
            yield path


def extract_lines(image: np.ndarray, language: str) -> List[LineBox]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, lang=language, output_type=Output.DICT)

    grouped: Dict[Tuple[int, int, int, int], List[int]] = {}
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not has_letters(text):
            continue

        conf_raw = data["conf"][i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1
        if conf < 40:
            continue

        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
            int(data["page_num"][i]),
        )
        grouped.setdefault(key, []).append(i)

    lines: List[LineBox] = []
    for idxs in grouped.values():
        left = min(int(data["left"][i]) for i in idxs)
        top = min(int(data["top"][i]) for i in idxs)
        right = max(int(data["left"][i]) + int(data["width"][i]) for i in idxs)
        bottom = max(int(data["top"][i]) + int(data["height"][i]) for i in idxs)
        text = " ".join(data["text"][i].strip() for i in idxs if data["text"][i].strip())
        sample_idx = idxs[0]
        lines.append(
            LineBox(
                text=text,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                block_num=int(data["block_num"][sample_idx]),
                par_num=int(data["par_num"][sample_idx]),
                page_num=int(data["page_num"][sample_idx]),
            )
        )

    # Нормализуем порядок строк (сверху вниз, затем слева направо)
    # для детерминированной обработки и отладки.
    lines.sort(key=lambda l: (l.top, l.left))
    return lines


def detect_bubbles(image: np.ndarray, min_area: int = 500) -> Dict[int, BubbleBox]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Светлые области чаще всего соответствуют баблам.
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        2,
    )

    # Смыкаем границы баблов в более цельные области.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    bubbles: Dict[int, BubbleBox] = {}
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        aspect_ratio = w / h if h != 0 else 0
        if area < min_area or aspect_ratio > 5 or aspect_ratio < 0.2:
            continue
        bubbles[label] = BubbleBox(
            bubble_id=label,
            left=int(x),
            top=int(y),
            right=int(x + w),
            bottom=int(y + h),
        )
    return bubbles


def assign_lines_to_bubbles(lines: List[LineBox], bubbles: Dict[int, BubbleBox]) -> None:
    for line in lines:
        cx, cy = line.center
        chosen = None
        for bubble in bubbles.values():
            if bubble.left <= cx <= bubble.right and bubble.top <= cy <= bubble.bottom:
                chosen = bubble.bubble_id
                break
        line.bubble_id = chosen


def find_suspect_lines(lines: List[LineBox], bubbles: Dict[int, BubbleBox], tolerance: int) -> List[LineBox]:
    """Помечает строки, смещённые влево/вправо.

    1) Если строка привязана к баблу: сравниваем центр текста (без крайних знаков)
       с центром бабла, в том числе для одиночных строк.
    2) Если бабл не найден: fallback на центр OCR-блока (только для групп >= 2 строк).
    """
    suspects: List[LineBox] = []
    suspect_keys: set[Tuple[int, int, int, int]] = set()

    # Основная проверка: относительно центра бабла, включая одиночные строки.
    for line in lines:
        if line.bubble_id is None or line.bubble_id not in bubbles:
            continue

        bubble = bubbles[line.bubble_id]
        text_left, text_right = estimate_text_horizontal_bounds(line)
        text_center_x = (text_left + text_right) // 2
        bubble_center_x = (bubble.left + bubble.right) // 2

        if abs(text_center_x - bubble_center_x) > tolerance:
            key = (line.left, line.top, line.right, line.bottom)
            if key not in suspect_keys:
                suspects.append(line)
                suspect_keys.add(key)

    # Fallback для текста без бабла.
    grouped: DefaultDict[Tuple[int, int, int], List[LineBox]] = defaultdict(list)
    for line in lines:
        if line.bubble_id is None or line.bubble_id not in bubbles:
            grouped[(line.page_num, line.block_num, line.par_num)].append(line)

    for group_lines in grouped.values():
        if len(group_lines) < 2:
            continue

        centers = sorted((estimate_text_horizontal_bounds(line)[0] + estimate_text_horizontal_bounds(line)[1]) // 2 for line in group_lines)
        median_center = centers[len(centers) // 2]

        for line in group_lines:
            text_left, text_right = estimate_text_horizontal_bounds(line)
            text_center_x = (text_left + text_right) // 2
            if abs(text_center_x - median_center) > tolerance:
                key = (line.left, line.top, line.right, line.bottom)
                if key not in suspect_keys:
                    suspects.append(line)
                    suspect_keys.add(key)

    return suspects


def draw_marks(image: np.ndarray, lines: List[LineBox]) -> np.ndarray:
    output = image.copy()
    for line in lines:
        cv2.rectangle(output, (line.left, line.top), (line.right, line.bottom), (0, 0, 255), 2)
    return output


def process_image(path: Path, lang: str, tolerance: int) -> Tuple[Path, int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")

    lines = extract_lines(image, language=lang)
    # Детектор баблов оставлен как вспомогательный (на случай будущих эвристик),
    # но проверка смещения работает независимо от формы/наличия бабла.
    _bubbles = detect_bubbles(image)
    assign_lines_to_bubbles(lines, _bubbles)
    suspects = find_suspect_lines(lines, _bubbles, tolerance=tolerance)

    marked = draw_marks(image, suspects)
    out_path = path.with_name(f"{path.stem}_fix.png")
    cv2.imwrite(str(out_path), marked)

    return out_path, len(lines), len(suspects)


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка тайпа манхвы по смещению строк относительно центра текстового блока.")
    parser.add_argument("folder", type=Path, help="Папка со сканами.")
    parser.add_argument("--lang", default="rus", help="Язык Tesseract (по умолчанию: rus).")
    parser.add_argument("--tolerance", type=int, default=TOLERANCE_PX, help="Допуск отклонения центра строки от центра блока в пикселях.")
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Папка не найдена: {folder}")

    images = list(iter_images(folder))
    if not images:
        raise SystemExit("В папке нет поддерживаемых изображений.")

    print(f"Найдено изображений: {len(images)}")
    for image_path in images:
        out_path, total_lines, suspect_lines = process_image(image_path, args.lang, args.tolerance)
        print(f"{image_path.name}: строк={total_lines}, подозрительных={suspect_lines}, сохранено -> {out_path.name}")


if __name__ == "__main__":
    main()